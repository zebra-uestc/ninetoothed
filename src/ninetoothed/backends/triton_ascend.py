"""Triton language emission with an independent Ascend schedule policy."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import Artifact, Backend, Capability, Target
from ninetoothed.backends.emitters.triton import emit as emit_triton
from ninetoothed.compiler.passes import Context, OptimizeSchedule, ScheduleCandidate
from ninetoothed.ir import ssa

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


def _operation_map(block: ssa.Block) -> dict[str, ssa.Operation]:
    result: dict[str, ssa.Operation] = {}

    for operation in block.operations:
        result.update((value.name, operation) for value in operation.results)

        for region in operation.regions:
            result.update(_operation_map(region))

    return result


def _all_value_names(program: ssa.Program) -> set[str]:
    names = {value.name for value in (*program.inputs, *program.outputs)}

    def visit(block: ssa.Block) -> None:
        names.update(value.name for value in block.args)

        for operation in block.operations:
            names.update(operation.operands)
            names.update(value.name for value in operation.results)

            for region in operation.regions:
                visit(region)

    for block in program.blocks:
        visit(block)

    return names


def _new_value_name(used: set[str], prefix: str) -> str:
    index = 0

    while True:
        name = f"%{prefix}_{index}"

        if name not in used:
            used.add(name)

            return name

        index += 1


def _is_tail_predicate(name: str, operations: Mapping[str, ssa.Operation]) -> bool:
    comparison = operations.get(name)

    if comparison is None or comparison.opcode != "cmp.lt":
        return False

    if len(comparison.operands) != 2:
        return False

    offset = operations.get(comparison.operands[0])
    limit = operations.get(comparison.operands[1])

    return (
        offset is not None
        and offset.opcode == "index.offset"
        and limit is not None
        and limit.opcode == "shape.dim"
        and bool(limit.attrs.get("source"))
    )


def _is_negative_infinity(name: str, operations: Mapping[str, ssa.Operation]) -> bool:
    operation = operations.get(name)

    return (
        operation is not None
        and operation.opcode == "arith.constant"
        and str(operation.attrs.get("value")).lower() in {"-inf", "-infinity"}
    )


def _replace_operands(
    operation: ssa.Operation,
    replacements: Mapping[str, str],
) -> ssa.Operation:
    operands = tuple(
        replacements.get(operand, operand) for operand in operation.operands
    )
    regions = tuple(
        _replace_block_operands(region, replacements) for region in operation.regions
    )

    return replace(operation, operands=operands, regions=regions)


def _replace_block_operands(
    block: ssa.Block,
    replacements: Mapping[str, str],
) -> ssa.Block:
    return replace(
        block,
        operations=tuple(
            _replace_operands(operation, replacements) for operation in block.operations
        ),
    )


def _tail_mask_value(
    operation: ssa.Operation,
    operations: Mapping[str, ssa.Operation],
) -> tuple[str, str] | None:
    if operation.opcode != "select.where" or len(operation.operands) != 3:
        return None

    predicate, selected, rejected = operation.operands

    if not _is_tail_predicate(predicate, operations):
        return None

    if not _is_negative_infinity(rejected, operations):
        return None

    selected_operation = operations.get(selected)

    if selected_operation is None or selected_operation.opcode != "linalg.dot":
        return None

    return predicate, operation.results[0].name


def _rewrite_attention_loop(
    operation: ssa.Operation,
    used: set[str],
) -> ssa.Operation:
    if operation.opcode != "scf.for" or len(operation.regions) != 1:
        return operation

    region = operation.regions[0]
    body = list(region.operations)
    definitions = _operation_map(region)
    key_mask = None
    qk_value = None
    qk_select_index = None

    for index, candidate in enumerate(body):
        match = _tail_mask_value(candidate, definitions)

        if match is not None:
            key_mask, qk_value = match
            qk_select_index = index
            break

    if key_mask is None or qk_value is None or qk_select_index is None:
        return operation

    # Find p = exp2(qk - m), excluding alpha = exp2(m_i - m).
    p_index = None
    p_operation = None

    for index, candidate in enumerate(body):
        if candidate.opcode != "math.exp2" or not candidate.operands:
            continue

        subtraction = definitions.get(candidate.operands[0])

        if subtraction is None or subtraction.opcode != "arith.sub":
            continue

        candidate_qk = subtraction.operands[0] if subtraction.operands else None
        conditional = next(
            (
                condition_index
                for condition_index in range(qk_select_index + 1, len(body))
                if body[condition_index].opcode == "scf.if"
                and body[condition_index].results
                and body[condition_index].results[0].name == candidate_qk
            ),
            None,
        )

        if candidate_qk == qk_value or conditional is not None:
            p_index = index
            p_operation = candidate
            break

    if p_index is None or p_operation is None:
        return replace(operation, regions=(replace(region, operations=tuple(body)),))

    zero_name = _new_value_name(used, "ascend_sdpa_zero")
    masked_name = _new_value_name(used, "ascend_sdpa_prob")
    p_type = p_operation.results[0].type
    zero = ssa.Value(
        name=zero_name,
        type=ssa.Type(kind="scalar", dtype=p_type.dtype or "float32"),
    )
    masked = ssa.Value(name=masked_name, type=p_type)
    zero_operation = ssa.Operation(
        opcode="arith.constant",
        results=(zero,),
        attrs={"value": 0.0},
    )
    mask_operation = ssa.Operation(
        opcode="select.where",
        operands=(key_mask, p_operation.results[0].name, zero_name),
        results=(masked,),
    )

    # Ascend's dynamic tail path can retain invalid probability values after
    # lowering.  Reapply the structurally identified key mask after exp2.
    body[p_index + 1 : p_index + 1] = [zero_operation, mask_operation]
    replacements = {p_operation.results[0].name: masked_name}

    for index in range(p_index + 3, len(body)):
        body[index] = _replace_operands(body[index], replacements)

    return replace(operation, regions=(replace(region, operations=tuple(body)),))


def _rewrite_sdpa_tail_masks(program: ssa.Program) -> ssa.Program:
    used = _all_value_names(program)

    def rewrite_block(block: ssa.Block) -> ssa.Block:
        operations = []

        for operation in block.operations:
            if operation.opcode == "scf.for":
                operation = _rewrite_attention_loop(operation, used)
            elif operation.regions:
                operation = replace(
                    operation,
                    regions=tuple(
                        rewrite_block(region) for region in operation.regions
                    ),
                )

            operations.append(operation)

        return replace(block, operations=tuple(operations))

    return replace(
        program,
        blocks=tuple(rewrite_block(block) for block in program.blocks),
    )


def _rewrite_clamp_operations(program: ssa.Program) -> ssa.Program:
    """Lower clamp to min(max(x, lower), upper) for Triton-Ascend."""
    used = _all_value_names(program)

    def rewrite_block(block: ssa.Block) -> ssa.Block:
        operations = []

        for operation in block.operations:
            if operation.regions:
                operation = replace(
                    operation,
                    regions=tuple(
                        rewrite_block(region) for region in operation.regions
                    ),
                )

            if operation.opcode != "call.clamp" or len(operation.operands) != 3:
                operations.append(operation)
                continue

            if len(operation.results) != 1:
                operations.append(operation)
                continue

            value, lower, upper = operation.operands
            result = operation.results[0]
            inner_name = _new_value_name(used, "ascend_clamp_lower")
            inner = ssa.Value(name=inner_name, type=result.type)
            operations.extend(
                (
                    ssa.Operation(
                        opcode="arith.maximum",
                        operands=(value, lower),
                        results=(inner,),
                    ),
                    ssa.Operation(
                        opcode="arith.minimum",
                        operands=(inner_name, upper),
                        results=(result,),
                    ),
                )
            )

        return replace(block, operations=tuple(operations))

    return replace(
        program,
        blocks=tuple(rewrite_block(block) for block in program.blocks),
    )


class TritonAscendOptimizeSchedule(OptimizeSchedule):
    """Generate Ascend candidates using the existing schedule pass contract."""

    name = "ssa.triton-ascend.optimize_schedule"
    supported_backends = (Target.TRITON_ASCEND,)

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        scheduled = super().run(program, context)
        scheduled = _rewrite_clamp_operations(scheduled)

        return _rewrite_sdpa_tail_masks(scheduled)

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        del analysis
        profile = context.resolved_target.platform.name
        b4 = profile == "ascend-910b4"
        granularity = schedule.get("granularity")

        if granularity == "layout-transfer":
            transfer = schedule.get("layout_transfer")

            if transfer is None or not transfer.requires_tiling:
                return ()

            return tuple(
                ScheduleCandidate(
                    name=f"ascend-transpose-{size}x{size}",
                    schedule={
                        "tile": {"block_m": size, "block_n": size},
                        "num_warps": 4 if size == 16 else 8,
                        "num_stages": 1,
                    },
                    tags=("layout-transfer",),
                )
                for size in ((16, 32) if b4 else (16,))
            )

        if granularity == "blocked-linalg":
            choices = ((32, 4, 2), (16, 2, 1))

            if b4:
                choices += ((64, 8, 3),)

            return tuple(
                ScheduleCandidate(
                    name=f"ascend-dot-{size}x{size}x32",
                    schedule={
                        "tile": {"block_m": size, "block_n": size, "block_k": 32},
                        "num_warps": warps,
                        "num_stages": stages,
                    },
                    tags=("dot",),
                )
                for size, warps, stages in choices
            )

        reduction = schedule.get("reduction", {})

        if (
            granularity == "parallel-reduction"
            and reduction.get("mode") == "row-vector"
        ):
            return tuple(
                ScheduleCandidate(
                    name=f"ascend-reduction-warps-{warps}",
                    schedule={"num_warps": warps, "num_stages": 1},
                    tags=("reduction",),
                )
                for warps in ((4, 8) if b4 else (4,))
            )

        return ()

    def optimization_policy(
        self,
        backend: Target,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend
        granularity = schedule.get("granularity")

        if granularity == "blocked-linalg":
            return {
                "preserve_linalg": bool(
                    analysis.get("dot_supports_low_precision_intrinsic", False)
                )
            }

        if granularity in {
            "parallel-reduction",
            "layout-transfer",
            "exp-reduction-dot-region",
        }:
            return {
                "schedule": {
                    "num_warps": schedule.get("num_warps", 4),
                    "num_stages": schedule.get("num_stages", 1),
                }
            }

        return {"schedule": {"num_warps": 4}}


class TritonAscendBackend(Backend):
    name = Target.TRITON_ASCEND
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=("Triton source is materialized by the Ascend JIT driver.",),
    )

    def emit(self, kernel) -> Artifact:
        return emit_triton(
            kernel,
            backend=Target.TRITON_ASCEND,
            source_route="ssa-unified-triton-ascend-emitter",
        )


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.TRITON_ASCEND,
        optimize_schedule=TritonAscendOptimizeSchedule,
    )


__all__ = [
    "TritonAscendBackend",
    "TritonAscendOptimizeSchedule",
    "register_ssa_passes",
]
