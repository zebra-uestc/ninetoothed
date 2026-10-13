"""Triton language emission with an independent Ascend schedule policy."""

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import Artifact, Backend, Capability, Target
from ninetoothed.backends.emitters.triton import emit as emit_triton
from ninetoothed.compiler.passes import Context, OptimizeSchedule, ScheduleCandidate

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class TritonAscendOptimizeSchedule(OptimizeSchedule):
    """Generate Ascend candidates using the existing schedule pass contract."""

    name = "ssa.triton-ascend.optimize_schedule"
    supported_backends = (Target.TRITON_ASCEND,)

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
