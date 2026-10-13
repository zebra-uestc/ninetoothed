"""Shared Triton-language JIT materialization primitives.

This module owns source import, Launch ABI validation, tuning-cache setup, and
the common runtime handle.  Device-family validation is injected by the
caller so this layer never assumes CUDA or an Ascend runtime.
"""

import functools
import os
import shutil
import types
from dataclasses import dataclass
from pathlib import Path

from ninetoothed.compiler.cache import (
    TRITON_CACHE_DIR,
    compilation_cache_key,
    write_source,
)
from ninetoothed.compiler.layout_runtime import (
    build_layout_transfer_validator,
    memory_spans_overlap,
    tensor_memory_span,
)
from ninetoothed.targets import runtime_device_types


def materialize_triton_jit(
    compilation,
    *,
    environment_validator=None,
    prepare_environment=True,
):
    """Materialize Triton source through a caller-selected runtime driver."""
    from ninetoothed.compiler.runtime import (
        Handle,
        _runtime_wrapper,
        _verified_runtime_launch,
        import_python_module,
    )

    runtime_metadata = {}

    if environment_validator is not None:
        runtime_metadata = dict(environment_validator(compilation) or {})

    if prepare_environment:
        _ensure_c_compiler()

    artifact = compilation.artifact
    cache_key = compilation_cache_key(compilation)
    source = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        "triton.py",
        cache_key=cache_key,
    )
    TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE_DIR))
    module = import_python_module(source)
    launch = getattr(module, artifact.entrypoint)
    kernel = getattr(module, f"{artifact.kernel_name}_kernel", None)
    candidates = tuple(compilation.launch_plan.tuning_candidates)
    validate_bindings = _runtime_binding_validator(compilation)
    candidate_launches = tuple(
        _candidate_launch(compilation, launch, kernel, candidate, validate_bindings)
        for candidate in candidates
    )
    tuner = None

    if len(candidate_launches) > 1:
        from ninetoothed.auto_tuner import AutoTuner

        class TritonAutoTuner(AutoTuner):
            @staticmethod
            def _make_arg_key(args, kwargs):
                return _triton_specialization_key(compilation, args, kwargs)

        tuner = TritonAutoTuner(
            candidate_launches,
            tuple((cache_key, candidate["id"]) for candidate in candidates),
            cache_namespace=f"jit_{cache_key}",
            validator=_runtime_validator(compilation, validate_bindings),
        )
        wrapped = tuner
    elif candidate_launches:
        wrapped = _verified_runtime_launch(candidate_launches[0])
    else:
        wrapped = _verified_runtime_launch(
            _runtime_wrapper(
                launch,
                compilation.launch_abi,
                specs=compilation.kernel.tensors,
                device_types=runtime_device_types(compilation),
                prepare_invocation=_triton_prepare_invocation(launch, kernel),
                validate_bindings=validate_bindings,
            )
        )

    handle = Handle(compilation, kernel, wrapped, source)
    handle._tuner = tuner
    handle._selected_tuning_candidate = candidates[0] if len(candidates) == 1 else None
    handle._runtime_metadata = runtime_metadata

    if tuner is not None:
        by_launch = dict(zip(candidate_launches, candidates))
        handle._launch = _tuned_runtime_launch(
            tuner, by_launch, handle, compilation, validate_bindings
        )

    return handle


def _candidate_launch(compilation, launch, kernel, candidate, validate_bindings):
    from ninetoothed.compiler.runtime import _runtime_wrapper

    bindings = {binding.name: binding for binding in compilation.launch_abi.kernel_args}
    meta_values = {
        str(bindings[name].source): value
        for name, value in dict(candidate.get("meta_parameters", {})).items()
    }
    private_meta = {
        f"_ninetoothed_{name.lower()}": value
        for name, value in dict(candidate.get("private_meta_parameters", {})).items()
    }
    function = functools.partial(
        launch,
        **private_meta,
        _ninetoothed_num_warps=int(candidate["num_warps"]),
        _ninetoothed_num_stages=int(candidate["num_stages"]),
    )

    return _runtime_wrapper(
        function,
        compilation.launch_abi,
        specs=compilation.kernel.tensors,
        device_types=runtime_device_types(compilation),
        binding_overrides=meta_values,
        prepare_invocation=_triton_prepare_invocation(function, kernel),
        validate_bindings=validate_bindings,
    )


def _triton_prepare_invocation(function, kernel):
    from ninetoothed.compiler.runtime import _is_cacheable_runtime_literal

    if kernel is None:
        return None

    target = getattr(function, "func", function)

    if not isinstance(target, types.FunctionType):
        return None

    kernel_names = tuple(
        name
        for name in target.__code__.co_names
        if target.__globals__.get(name) is kernel
    )

    if len(kernel_names) != 1:
        return None

    kernel_name = kernel_names[0]
    partial_args = getattr(function, "args", ())
    partial_keywords = getattr(function, "keywords", None) or {}

    @dataclass(frozen=True)
    class ValueRef:
        index: int

        def resolve(self, values, _args, _kwargs):
            return values[self.index]

    @dataclass(frozen=True)
    class CallRef:
        kind: str
        key: object

        def resolve(self, _values, args, kwargs):
            return args[self.key] if self.kind == "positional" else kwargs[self.key]

    @dataclass(frozen=True)
    class Literal:
        value: object

        def resolve(self, _values, _args, _kwargs):
            return self.value

    @dataclass(frozen=True)
    class BoundCall:
        function: object
        args: tuple
        kwargs: tuple

        def invoke(self, values, args, kwargs):
            self.function(
                *(value.resolve(values, args, kwargs) for value in self.args),
                **{
                    name: value.resolve(values, args, kwargs)
                    for name, value in self.kwargs
                },
            )

    @dataclass(frozen=True)
    class InvocationPlan:
        call: object
        requires_values: bool

        def __call__(self, values, args, kwargs):
            if self.requires_values:
                self.call.invoke(values, args, kwargs)
            else:
                self.call(args, kwargs)

    def plan_value(value, values, static_values, call_sources):
        for index, candidate in enumerate(values):
            if value is candidate:
                if call_sources is not None and call_sources[index] is not None:
                    return CallRef(*call_sources[index])

                return (
                    Literal(static_values[index])
                    if static_values is not None
                    else ValueRef(index)
                )

        if hasattr(value, "data_ptr") or (
            hasattr(value, "shape") and hasattr(value, "dtype")
        ):
            return None

        if _is_cacheable_runtime_literal(value):
            return Literal(value)

        return None

    def build_direct_invocation(function, args, kwargs):
        namespace = {"_function": function}
        expressions = []

        def expression(value):
            if isinstance(value, CallRef):
                container = "_args" if value.kind == "positional" else "_kwargs"

                return f"{container}[{value.key!r}]"

            name = f"_value_{len(namespace)}"
            namespace[name] = value.value

            return name

        expressions.extend(expression(value) for value in args)
        expressions.extend(f"{name}={expression(value)}" for name, value in kwargs)
        source = (
            "def invoke(_args, _kwargs):\n    _function(" + ", ".join(expressions) + ")"
        )
        exec(compile(source, "<ninetoothed-triton-invocation>", "exec"), namespace)

        return namespace["invoke"]

    def prepare(values, static_values=None, call_sources=None):
        calls = []

        class KernelProxy:
            def __getitem__(self, grid):
                bound_kernel = kernel[grid]

                def capture(*args, **kwargs):
                    calls.append((bound_kernel, args, kwargs))

                return capture

        globals_copy = dict(target.__globals__)
        globals_copy[kernel_name] = KernelProxy()
        dry_launch = types.FunctionType(
            target.__code__,
            globals_copy,
            target.__name__,
            target.__defaults__,
            target.__closure__,
        )
        dry_launch.__kwdefaults__ = target.__kwdefaults__
        functools.partial(
            dry_launch,
            *partial_args,
            **partial_keywords,
        )(*values)

        if len(calls) != 1:
            return None

        bound_kernel, args, kwargs = calls[0]
        planned_args = tuple(
            plan_value(value, values, static_values, call_sources) for value in args
        )
        planned_kwargs = tuple(
            (
                name,
                plan_value(value, values, static_values, call_sources),
            )
            for name, value in kwargs.items()
        )

        if any(value is None for value in planned_args) or any(
            value is None for _name, value in planned_kwargs
        ):
            return None

        planned_call = BoundCall(bound_kernel, planned_args, planned_kwargs)

        requires_values = any(
            isinstance(value, ValueRef) for value in planned_args
        ) or any(isinstance(value, ValueRef) for _name, value in planned_kwargs)

        if requires_values:
            return InvocationPlan(planned_call, True)

        direct_call = build_direct_invocation(
            planned_call.function,
            planned_call.args,
            planned_call.kwargs,
        )

        return InvocationPlan(direct_call, False)

    return prepare


def _runtime_alias_signature(abi, public):
    from ninetoothed.compiler.runtime import _binding_value

    output_names = set(abi.outputs)
    physical_tensors = {}
    aliases = []

    for binding in abi.kernel_args:
        if binding.source not in public or binding.kind not in {
            "tensor",
            "scalar",
            "constexpr",
            "jagged_values",
            "jagged_offsets",
        }:
            continue

        value = _binding_value(binding, public)

        if binding.kind in {"scalar", "constexpr"} and not all(
            hasattr(value, attribute) for attribute in ("data_ptr", "shape", "stride")
        ):
            continue

        identity = (binding.source, binding.kind)
        access = binding.access or (
            "read"
            if binding.kind in {"scalar", "constexpr", "jagged_offsets"}
            else "read_write"
            if binding.source in output_names
            else "read"
        )
        previous = physical_tensors.get(identity)

        if previous is not None and previous[2] != access:
            access = "read_write"

        physical_tensors[identity] = (
            *identity,
            access,
            value,
            tensor_memory_span(value),
        )

    writers = tuple(
        tensor
        for tensor in physical_tensors.values()
        if tensor[2] in {"write", "read_write"}
    )
    readers = tuple(
        tensor
        for tensor in physical_tensors.values()
        if tensor[2] in {"read", "read_write"}
    )

    for writer_name, writer_kind, _access, writer, writer_span in writers:
        for reader_name, reader_kind, _access, reader, reader_span in readers:
            overlaps = writer is reader

            if not overlaps:
                overlaps = memory_spans_overlap(writer_span, reader_span)

            if overlaps:
                aliases.append((writer_name, writer_kind, reader_name, reader_kind))

    return tuple(aliases)


def _tuned_runtime_launch(
    tuner,
    candidates_by_launch,
    handle,
    compilation,
    validate_bindings=None,
):
    from ninetoothed.compiler.runtime import (
        _arm_prepared_runtime_launch,
        _empty_launch,
        _first_output,
        _public_values,
        _remember_verified_runtime_call,
        _runtime_call_identity,
    )

    active_identity = None
    active = None
    prepared_calls = {}

    def evict(identity, token):
        nonlocal active, active_identity
        cached = prepared_calls.get(identity)

        if cached is not None and cached[2].cache_token is token:
            prepared_calls.pop(identity, None)

        if active is not None and active[2].cache_token is token:
            active = None
            active_identity = None

    def activate(identity, entry):
        nonlocal active, active_identity
        active_identity = identity
        active = entry
        record(entry[1])

    def record(selected):
        handle._selected_tuning_candidate = candidates_by_launch[selected]

    def remember(identity, selection_key, selected, prepared):
        token = object()

        def collected(_reference):
            evict(identity, token)

        prepared = _arm_prepared_runtime_launch(prepared, collected, token)

        if prepared is None:
            return None

        entry = (selection_key, selected, prepared)
        _remember_verified_runtime_call(prepared_calls, identity, entry)
        activate(identity, entry)

        return prepared

    def remember_best_effort(identity, selection_key, selected, prepared):
        try:
            return remember(identity, selection_key, selected, prepared)
        except TypeError:
            return None

    def launch(*args, **kwargs):
        active_snapshot = active
        active_identity_snapshot = active_identity
        identity = _runtime_call_identity(args, kwargs)

        if (
            active_snapshot is not None
            and identity == active_identity_snapshot
            and active_snapshot[2].matches(
                args,
                kwargs,
                identity_verified=True,
            )
        ):
            return active_snapshot[1]._ninetoothed_invoke_prepared(
                active_snapshot[2],
                args,
                kwargs,
            )

        cached = prepared_calls.pop(identity, None)

        if cached is not None and cached[2].matches(
            args,
            kwargs,
            identity_verified=True,
        ):
            prepared_calls[identity] = cached
            activate(identity, cached)

            return cached[1]._ninetoothed_invoke_prepared(cached[2], args, kwargs)

        public = _public_values(
            compilation.launch_abi,
            args,
            kwargs,
            specs=compilation.kernel.tensors,
            device_types=runtime_device_types(compilation),
        )

        if _empty_launch(compilation.launch_abi, public):
            if validate_bindings is not None:
                validate_bindings(public)

            return _first_output(compilation.launch_abi, public)

        key = tuner._make_arg_key(args, kwargs)
        alias_signature = _runtime_alias_signature(compilation.launch_abi, public)
        selection_key = (key, alias_signature)
        selected = next(
            (
                entry[1]
                for entry in reversed(tuple(prepared_calls.values()))
                if entry[0] == selection_key
            ),
            None,
        )

        if selected is None and cached is not None and cached[0] == selection_key:
            selected = cached[1]

        if selected is None and alias_signature:
            selected = tuner._funcs[0]

            try:
                prepared = selected._ninetoothed_prepare(
                    args,
                    kwargs,
                    public=public,
                )
            except Exception:  # noqa: BLE001
                result = selected(*args, **kwargs)
            else:
                result = selected._ninetoothed_invoke_prepared(
                    prepared,
                    args,
                    kwargs,
                )
                remember_best_effort(
                    identity,
                    selection_key,
                    selected,
                    prepared,
                )

            record(selected)

            return result

        if selected is None:
            result = tuner(*args, **kwargs)
            selected = tuner._best_func[key]
            record(selected)

            try:
                prepared = selected._ninetoothed_prepare(args, kwargs, public=public)
            except Exception:  # noqa: BLE001
                return result

            remember_best_effort(identity, selection_key, selected, prepared)

            return result

        try:
            prepared = selected._ninetoothed_prepare(args, kwargs, public=public)
        except Exception:  # noqa: BLE001
            result = selected(*args, **kwargs)
            record(selected)

            return result

        result = selected._ninetoothed_invoke_prepared(
            prepared,
            args,
            kwargs,
        )
        record(selected)
        remember_best_effort(identity, selection_key, selected, prepared)

        return result

    return launch


def _triton_specialization_key(compilation, args, kwargs):
    from ninetoothed.auto_tuner import AutoTuner

    public = dict(zip(compilation.launch_abi.public_args, args))
    public.update(kwargs)
    scalar_values = []

    for binding in compilation.launch_abi.kernel_args:
        if (
            binding.kind not in {"scalar", "constexpr", "meta"}
            or binding.source not in public
        ):
            continue

        value = public[binding.source]
        shape = getattr(value, "shape", None)

        if shape is not None and not tuple(shape) and hasattr(value, "item"):
            value = value.item()

        scalar_values.append((binding.source, type(value).__name__, repr(value)))

    base = AutoTuner._make_arg_key(args, kwargs)

    return f"{base}, specialization={tuple(scalar_values)!r}"


def _runtime_validator(compilation, validate_bindings):
    from ninetoothed.compiler.runtime import _public_values

    def validate(args, kwargs):
        public = _public_values(
            compilation.launch_abi,
            args,
            kwargs,
            specs=compilation.kernel.tensors,
            device_types=runtime_device_types(compilation),
        )

        if validate_bindings is not None:
            validate_bindings(public)

    return validate


def _runtime_binding_validator(compilation):
    validators = tuple(
        validator
        for validator in (
            _runtime_vector_validator(compilation),
            build_layout_transfer_validator(
                getattr(compilation.artifact, "metadata", {}),
                compilation.launch_abi,
            ),
        )
        if validator is not None
    )

    if not validators:
        return None

    def validate(public):
        for validator in validators:
            validator(public)

    return validate


def _runtime_vector_validator(compilation):
    metadata = getattr(compilation.artifact, "metadata", {})
    limit = metadata.get("vector_numel_limit")
    schedule = metadata.get("ssa_schedule", {})
    reduction = schedule.get("reduction", {})

    if reduction.get("mode") != "row-vector":
        return None

    import sympy

    extent_expression = sympy.sympify(str(reduction["extent"]))
    program_constraints = tuple(
        (
            tuple(sympy.sympify(str(value)) for value in actual),
            tuple(sympy.sympify(str(value)) for value in expected),
        )
        for actual, expected in reduction.get("program_constraints", ())
    )
    expressions = (
        extent_expression,
        *(
            expression
            for constraint in program_constraints
            for shape in constraint
            for expression in shape
        ),
    )
    bindings = {binding.name: binding for binding in compilation.launch_abi.kernel_args}

    try:
        symbols = tuple(
            (symbol, bindings[str(symbol)])
            for symbol in sorted(
                set().union(*(expression.free_symbols for expression in expressions)),
                key=str,
            )
        )
    except KeyError as exc:
        raise ValueError(
            f"Triton reduction extent references unknown symbol `{exc.args[0]}`."
        ) from exc

    from ninetoothed.compiler.runtime import _binding_value

    last_values = None

    def validate(public):
        nonlocal last_values
        values = tuple(_binding_value(binding, public) for _symbol, binding in symbols)

        if values == last_values:
            return

        substitutions = {
            symbol: value for (symbol, _binding), value in zip(symbols, values)
        }

        def resolve(expression):
            resolved = expression.subs(substitutions)

            if resolved.free_symbols:
                names = ", ".join(sorted(map(str, resolved.free_symbols)))
                raise ValueError(f"Unresolved Triton row-vector symbols: {names}.")
            return int(resolved)

        extent = resolve(extent_expression)

        if limit is not None and extent > limit:
            block = 1 << (extent - 1).bit_length()
            raise ValueError(
                f"Triton row-vector reduction extent {extent} requires "
                f"BLOCK={block}, exceeding the backend tensor numel limit "
                f"{limit}; hierarchical reduction is not implemented."
            )

        for actual, expected in program_constraints:
            actual_total = 1
            expected_total = 1

            for expression in actual:
                actual_total *= resolve(expression)

            for expression in expected:
                expected_total *= resolve(expression)

            if actual_total != expected_total:
                raise ValueError(
                    "Triton row-vector reduction operand and output program "
                    "domains do not match; separate kernels are required."
                )

        last_values = values

    return validate


def _ensure_c_compiler() -> None:
    """Choose a host C compiler for Triton Python JIT setup."""
    if os.environ.get("CC"):
        return

    candidates = (
        shutil.which("cc"),
        shutil.which("gcc"),
        "/usr/bin/cc",
        "/usr/bin/gcc",
    )

    for candidate in candidates:
        if candidate is not None and Path(candidate).is_file():
            os.environ["CC"] = candidate

            return


__all__ = ["materialize_triton_jit", "_ensure_c_compiler"]
