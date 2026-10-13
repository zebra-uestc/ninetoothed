"""Triton artifact materialization."""

import ctypes
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.backends.emitters.expressions import replace_symbols
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.backends.materializers.triton_jit import (
    _ensure_c_compiler,
)
from ninetoothed.backends.materializers.triton_jit import (
    materialize_triton_jit as _shared_materialize_triton_jit,
)
from ninetoothed.backends.toolchain import find_nvcc
from ninetoothed.compiler.cache import (
    TRITON_CACHE_DIR,
    artifact_directory,
    atomic_write_bytes,
    cache_lock,
    compilation_cache_key,
    read_manifest,
    write_manifest,
    write_source,
)
from ninetoothed.compiler.layout_runtime import build_layout_transfer_validator
from ninetoothed.dtype import normalize_dtype

_TRITON_AOT_LAUNCHER_SCHEMA = 2
_TRITON_MODULE_PATTERN = re.compile(
    r"^CUmodule ([A-Za-z_][A-Za-z0-9_]*)_mod = NULL;$", re.MULTILINE
)
_TRITON_FUNCTION_PATTERN = re.compile(
    r"^CUfunction ([A-Za-z_][A-Za-z0-9_]*)_func = NULL;$", re.MULTILINE
)
_TRITON_LOADER_PATTERN = re.compile(
    r"^void load_([A-Za-z_][A-Za-z0-9_]*)\(\) \{$", re.MULTILINE
)


class TritonMaterializer(Materializer):
    target = Target.TRITON

    def jit_materialize(self, compilation, *, output_dir=None):
        del output_dir

        return _shared_materialize_triton_jit(compilation)

    def aot_build(self, compilation, *, output_dir: str | Path):
        return _aot_materialize(compilation, output_dir=output_dir)

    def load_built_artifact(self, built: BuiltArtifact):
        if built.binary_path is None:
            raise ValueError("Triton built artifact does not contain an AOT binary.")

        from ninetoothed.compiler.runtime import (
            _launch_abi_from_dict,
            _runtime_specs,
        )

        _, function, enter, leave = _load_aot_exports(
            built.binary_path,
            built.source.kernel_name,
        )
        specs = _runtime_specs(built.source)

        abi = _launch_abi_from_dict(built.abi)

        return _aot_wrapper(
            function,
            enter,
            leave,
            abi,
            specs,
            validate_bindings=build_layout_transfer_validator(
                built.source.metadata,
                abi,
            ),
        )


def _aot_materialize(compilation, *, output_dir):
    from ninetoothed.compiler.runtime import Handle, _publish_library

    compilation = _select_aot_candidate(compilation)
    artifact = compilation.artifact
    cache_key = compilation_cache_key(compilation)
    source = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        "triton.py",
        cache_key=cache_key,
    )
    cache_library = artifact_directory(cache_key) / f"{artifact.kernel_name}.triton.so"

    with cache_lock(cache_library):
        _ensure_aot_library(compilation, source, cache_library)

        write_manifest(
            cache_library.with_suffix(".manifest.json"),
            _triton_aot_manifest(compilation, cache_key, source, cache_library),
        )

    library_path = _publish_library(
        cache_library,
        output_dir,
        f"{artifact.kernel_name}.triton.so",
    )
    _, function, enter, leave = _load_aot_exports(
        library_path,
        artifact.kernel_name,
    )
    wrapped = _aot_wrapper(
        function,
        enter,
        leave,
        compilation.launch_abi,
        compilation.kernel.tensors,
        validate_bindings=build_layout_transfer_validator(
            getattr(artifact, "metadata", {}),
            compilation.launch_abi,
        ),
    )
    handle = Handle(compilation, function, wrapped, source, library_path)
    write_manifest(
        handle._built_artifact.manifest_path,
        _triton_aot_manifest(compilation, cache_key, source, library_path),
    )

    return handle


def _select_aot_candidate(compilation):
    launch_plan = getattr(compilation, "launch_plan", None)

    if launch_plan is None:
        return compilation

    candidates = tuple(launch_plan.tuning_candidates)

    if len(candidates) <= 1:
        return compilation

    metadata = getattr(compilation.artifact, "metadata", {})

    if not metadata.get("layout_transfer"):
        raise ValueError(
            "Triton AOT accepts one launch configuration; use build() to "
            "benchmark and package multiple explicit configurations."
        )

    return replace(
        compilation,
        launch_plan=replace(
            compilation.launch_plan,
            tuning_candidates=(candidates[0],),
        ),
    )


def _ensure_aot_library(compilation, source: Path, library: Path) -> None:
    manifest = read_manifest(library.with_suffix(".manifest.json"))
    schema = None if manifest is None else manifest.get("triton_aot_launcher_schema")

    if (
        not library.is_file()
        or schema != _TRITON_AOT_LAUNCHER_SCHEMA
        or _cpp_source_bundle(library) is None
    ):
        _compile_aot_library(compilation, source, library)


def _triton_aot_manifest(compilation, cache_key, source, library):
    from ninetoothed.compiler.runtime import _built_manifest

    return dict(_built_manifest(compilation, cache_key, source, library)) | {
        "triton_aot_launcher_schema": _TRITON_AOT_LAUNCHER_SCHEMA,
    }


def _load_aot_exports(library_path, kernel_name):
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, f"{kernel_name}_kernel_default")

    try:
        enter = library.ninetoothed_triton_enter
        leave = library.ninetoothed_triton_leave
    except AttributeError as exc:
        raise RuntimeError(
            "Triton AOT artifact is missing the CUDA context guard exports; "
            "rebuild the artifact with the current NineToothed version."
        ) from exc

    function.restype = ctypes.c_int
    enter.argtypes = []
    enter.restype = ctypes.c_int
    leave.argtypes = []
    leave.restype = None

    return library, function, enter, leave


def _compile_aot_library(compilation, source: Path, library: Path) -> None:
    _ensure_c_compiler()
    artifact = compilation.artifact
    kernel_name = f"{artifact.kernel_name}_kernel"
    signature = _compile_signature(compilation)
    grid = _compile_grid(compilation)
    num_warps, num_stages = _compile_schedule(compilation)
    library.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    environment["TRITON_CACHE_DIR"] = str(TRITON_CACHE_DIR)

    with tempfile.TemporaryDirectory(dir=library.parent) as temporary_dir:
        temporary = Path(temporary_dir)
        compiled = temporary / "compiled"
        linked = temporary / "linked"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "triton.tools.compile",
                str(source),
                "--kernel-name",
                kernel_name,
                "--signature",
                signature,
                "--grid",
                grid,
                "--num-warps",
                str(num_warps),
                "--num-stages",
                str(num_stages),
                "--out-name",
                kernel_name,
                "--out-path",
                str(compiled),
            ],
            check=True,
            env=environment,
        )
        headers = tuple(temporary.glob("compiled.*.h"))
        sources = tuple(temporary.glob("compiled.*.c"))

        if not headers or not sources:
            raise RuntimeError("Triton AOT compiler did not produce C artifacts.")

        kernel_names = _triton_aot_kernel_names(sources)
        context_guard = temporary / "ninetoothed_triton_context_guard.cu"
        context_guard.write_text(
            _triton_context_guard_source(kernel_names),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "triton.tools.link",
                *(str(path) for path in headers),
                "--out",
                str(linked),
            ],
            check=True,
            env=environment,
        )
        output = temporary / library.name
        subprocess.run(
            [
                find_nvcc(),
                "-shared",
                "-std=c++17",
                "-Xcompiler",
                "-fPIC",
                "-Xcompiler",
                "-pthread",
                "-O3",
                *(str(path) for path in sources),
                str(linked.with_suffix(".c")),
                str(context_guard),
                "-lcuda",
                "-Xlinker",
                "-z",
                "-Xlinker",
                "defs",
                "-o",
                str(output),
            ],
            check=True,
        )
        os.replace(output, library)
        _write_cpp_sources(compilation, sources, linked, context_guard, library)


def _write_cpp_sources(compilation, sources, linked, context_guard, library):
    name = compilation.artifact.kernel_name
    enter = f"{name}_triton_enter"
    leave = f"{name}_triton_leave"
    files = {
        f"{name}.triton.{path.stem}.cpp": (
            '#include <cstdlib>\n#include <cuda.h>\nextern "C" {\n'
            + path.read_text(encoding="utf-8")
            + "\n}\n"
        )
        for path in (*sources, linked.with_suffix(".c"))
    }
    guard = context_guard.read_text(encoding="utf-8")
    guard = guard.replace("ninetoothed_triton_enter", enter)
    guard = guard.replace("ninetoothed_triton_leave", leave)
    files[f"{name}.triton.launch.cpp"] = guard
    write_manifest(library.with_suffix(".sources.json"), files)


def _cpp_source_bundle(library):
    sources = read_manifest(library.with_suffix(".sources.json"))

    if not sources or not all(
        Path(name).name == name and isinstance(source, str)
        for name, source in sources.items()
    ):
        return None

    if not all(
        any(name.endswith(suffix) for name in sources)
        for suffix in (".triton.launch.cpp", ".triton.linked.cpp")
    ):
        return None
    return sources


def publish_cpp_sources(compilation, output_dir):
    """Publish standalone sources for a concrete Triton build variant."""
    from ninetoothed.backends.materializers.cpp import wrapper_source, write_header

    name = compilation.artifact.kernel_name
    library = (
        artifact_directory(compilation_cache_key(compilation)) / f"{name}.triton.so"
    )
    output_dir = Path(output_dir)
    sources = _cpp_source_bundle(library)

    if sources is None:
        raise RuntimeError("Triton AOT artifact is missing its C++ sources.")

    sources = dict(sources)
    sources[f"{name}.triton.launch.cpp"] += wrapper_source(
        compilation, f"{name}_triton_enter", f"{name}_triton_leave"
    )
    write_header(output_dir)

    for filename, source in sources.items():
        atomic_write_bytes(output_dir / filename, source.encode("utf-8"))


def _triton_aot_kernel_names(sources: tuple[Path, ...]) -> tuple[str, ...]:
    """Return low-level kernels with matching module, function, and loader symbols."""
    if not sources:
        raise ValueError("Unsupported Triton AOT source format: no C sources found.")

    kernel_names = []

    for source in sources:
        text = source.read_text(encoding="utf-8")
        modules = _TRITON_MODULE_PATTERN.findall(text)
        functions = _TRITON_FUNCTION_PATTERN.findall(text)
        loaders = _TRITON_LOADER_PATTERN.findall(text)
        module_names = set(modules)

        if (
            not modules
            or len(modules) != len(module_names)
            or len(functions) != len(set(functions))
            or len(loaders) != len(set(loaders))
            or module_names != set(functions)
            or module_names != set(loaders)
        ):
            raise ValueError(
                f"Unsupported Triton AOT source format in `{source}`: expected "
                "one matching `CUmodule`, `CUfunction`, and `load_*` symbol per "
                "low-level kernel."
            )

        kernel_names.extend(modules)

    if len(kernel_names) != len(set(kernel_names)):
        raise ValueError(
            "Unsupported Triton AOT source format: duplicate low-level kernel symbols."
        )
    return tuple(kernel_names)


def _triton_context_guard_source(kernel_names: tuple[str, ...]) -> str:
    """Generate the CUDA-context guard linked beside Triton's AOT launchers."""
    declarations = "\n".join(
        (
            f'extern "C" CUmodule {name}_mod;\n'
            f'extern "C" CUfunction {name}_func;\n'
            f'extern "C" void load_{name}(void);'
        )
        for name in kernel_names
    )
    resets_and_loads = "\n".join(
        (
            f"        {name}_mod = nullptr;\n"
            f"        {name}_func = nullptr;\n"
            f"        load_{name}();\n"
            f"        if ({name}_mod == nullptr || {name}_func == nullptr) {{\n"
            "            launch_mutex.unlock();\n"
            "            return CUDA_ERROR_INVALID_HANDLE;\n"
            "        }"
        )
        for name in kernel_names
    )
    stores = "\n".join(
        (
            f"        state.modules[{index}] = {name}_mod;\n"
            f"        state.functions[{index}] = {name}_func;"
        )
        for index, name in enumerate(kernel_names)
    )
    restores = "\n".join(
        (
            f"        {name}_mod = found->second.modules[{index}];\n"
            f"        {name}_func = found->second.functions[{index}];"
        )
        for index, name in enumerate(kernel_names)
    )

    return f"""#include <array>
#include <cuda.h>
#include <mutex>
#include <unordered_map>

{declarations}

namespace {{
struct State {{
    std::array<CUmodule, {len(kernel_names)}> modules{{}};
    std::array<CUfunction, {len(kernel_names)}> functions{{}};
}};

std::mutex launch_mutex;
std::unordered_map<CUcontext, State> context_states;
}}

extern "C" CUresult ninetoothed_triton_enter(void) {{
    launch_mutex.lock();

    CUcontext context = nullptr;
    CUresult result = cuCtxGetCurrent(&context);

    if (result != CUDA_SUCCESS) {{
        launch_mutex.unlock();
        return result;
    }}

    if (context == nullptr) {{
        launch_mutex.unlock();
        return CUDA_ERROR_INVALID_CONTEXT;
    }}

    try {{
        auto found = context_states.find(context);

        if (found == context_states.end()) {{
{resets_and_loads}

            State state{{}};
{stores}
            context_states.emplace(context, state);
        }} else {{
{restores}
        }}
    }} catch (...) {{
        launch_mutex.unlock();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }}

    return CUDA_SUCCESS;
}}

extern "C" void ninetoothed_triton_leave(void) {{
    launch_mutex.unlock();
}}
"""


def _compile_signature(compilation) -> str:
    specs = {spec.name: spec for spec in compilation.kernel.tensors}
    values = []

    for binding in compilation.launch_abi.kernel_args:
        if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}:
            values.append(f"*{_triton_dtype(specs[binding.source].dtype)}")
        elif binding.kind == "scalar":
            values.append(_triton_scalar_dtype(specs[binding.source].dtype))
        elif binding.kind in {"constexpr", "meta"}:
            if binding.value is None:
                raise ValueError(f"Triton AOT requires a value for `{binding.name}`.")

            values.append(str(binding.value))
        else:
            values.append("i64")

    private_meta = _private_meta_values(compilation)
    values.extend(
        str(private_meta[name])
        for name in dict(compilation.artifact.metadata.get("layout_transfer", {})).get(
            "private_meta_parameters", ()
        )
    )
    values.append(str(_compile_block(compilation)))

    return ",".join(values)


def _triton_dtype(dtype) -> str:
    name = normalize_dtype(str(dtype))
    aliases = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float64": "fp64",
        "bool": "i1",
    }

    if name in aliases:
        return aliases[name]

    if name.startswith(("int", "uint")):
        prefix = "i" if name.startswith("int") else "u"

        return prefix + "".join(character for character in name if character.isdigit())

    raise TypeError(f"Unsupported Triton AOT dtype: {dtype!r}.")


def _triton_scalar_dtype(dtype) -> str:
    name = normalize_dtype(str(dtype))

    if name in {
        "float16",
        "bfloat16",
        "float32",
        "float64",
    }:
        return "fp64"
    return _triton_dtype(dtype)


def _compile_block(compilation) -> int:
    mode = dict(compilation.artifact.metadata.get("program_mode", {}))
    reduction = dict(
        compilation.artifact.metadata.get("ssa_schedule", {}).get("reduction", {})
    )

    if mode.get("block") or mode.get("scalar"):
        return 1

    if mode.get("vector"):
        total = (
            _constant_reduction_extent(compilation, reduction)
            if reduction.get("mode") == "row-vector"
            else _constant_grid_total(compilation)
        )
        block = 1 << max(0, (total - 1).bit_length())
        limit = compilation.artifact.metadata.get("vector_numel_limit")

        if limit is not None and block > limit:
            raise ValueError(
                f"Triton row-vector reduction extent {total} requires "
                f"BLOCK={block}, exceeding the backend tensor numel limit "
                f"{limit}; hierarchical reduction is not implemented."
            )
        return block
    return 256


def _constant_reduction_extent(compilation, reduction) -> int:
    expression = replace_symbols(
        str(reduction["extent"]),
        {
            binding.name: str(binding.value)
            for binding in compilation.launch_abi.kernel_args
            if binding.kind in {"meta", "constexpr"} and binding.value is not None
        },
    )

    try:
        import sympy

        value = sympy.sympify(expression)

        if value.free_symbols:
            raise ValueError
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Triton AOT row-vector reductions require a statically "
            "specialized reduction extent."
        ) from exc


def _constant_grid_total(compilation) -> int:
    expression = _specialized_grid_total(compilation)

    try:
        import sympy

        value = sympy.sympify(expression)

        if value.free_symbols:
            raise ValueError
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Triton AOT vector programs require a statically specialized domain."
        ) from exc


def _compile_grid(compilation) -> str:
    total = _specialized_grid_total(compilation).replace("//", "/")
    mode = dict(compilation.artifact.metadata.get("program_mode", {}))

    if not any(mode.get(name) for name in ("block", "scalar", "vector")):
        block = _compile_block(compilation)
        total = f"((({total}) + {block - 1}) / {block})"
    return f"{total},1,1"


def _specialized_grid_total(compilation) -> str:
    expression = compilation.launch_plan.grid[0].render()
    replacements = {
        binding.name: str(binding.value)
        for binding in compilation.launch_abi.kernel_args
        if binding.kind in {"meta", "constexpr"} and binding.value is not None
    }
    replacements.update(
        {
            f"_ninetoothed_{name.lower()}": str(value)
            for name, value in _private_meta_values(compilation).items()
        }
    )

    return replace_symbols(expression, replacements)


def _private_meta_values(compilation) -> dict[str, int]:
    candidates = tuple(compilation.launch_plan.tuning_candidates)

    if not candidates:
        return {}

    return {
        str(name): int(value)
        for name, value in dict(
            candidates[0].get("private_meta_parameters", {})
        ).items()
    }


def _compile_schedule(compilation) -> tuple[int, int]:
    schedule = dict(compilation.artifact.metadata.get("ssa_schedule", {}))
    candidates = tuple(compilation.launch_plan.tuning_candidates)
    selected = dict(candidates[0]) if candidates else {}
    warps = (
        compilation.request.num_warps
        or selected.get("num_warps")
        or schedule.get("num_warps")
        or 4
    )
    stages = (
        compilation.request.num_stages
        or selected.get("num_stages")
        or schedule.get("num_stages")
        or 3
    )

    if isinstance(warps, tuple):
        warps = warps[0]

    if isinstance(stages, tuple):
        stages = stages[0]
    return int(warps), int(stages)


def _aot_wrapper(
    function,
    enter,
    leave,
    abi,
    tensor_specs,
    *,
    validate_bindings=None,
):
    from ninetoothed.compiler.runtime import (
        KernelLaunchError,
        _bound_values,
        _empty_launch,
        _first_output,
        _public_values,
    )

    specs = {spec.name: spec for spec in tensor_specs}
    runtime_bindings = tuple(
        binding
        for binding in abi.kernel_args
        if binding.kind not in {"constexpr", "meta"}
    )
    function.argtypes = [
        ctypes.c_void_p,
        *(_triton_aot_ctype(binding, specs) for binding in runtime_bindings),
    ]

    def launch(*args, **kwargs):
        import torch

        public = _public_values(abi, args, kwargs, specs=tensor_specs)
        _validate_aot_constants(abi, public)

        if validate_bindings is not None:
            validate_bindings(public)

        if _empty_launch(abi, public):
            return _first_output(abi, public)

        runtime_abi = replace(
            abi,
            kernel_args=runtime_bindings,
        )
        values, keepalive = _bound_values(
            runtime_abi,
            public,
            scalar_mode="cuda",
            specs=specs,
            cuda_scalar=_triton_aot_scalar,
        )
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        enter_result = enter()

        if enter_result != 0:
            raise KernelLaunchError(enter_result)

        try:
            result = function(stream, *values)
        finally:
            leave()

        del keepalive

        if result != 0:
            raise KernelLaunchError(result)
        return _first_output(abi, public)

    return launch


def _triton_aot_scalar(value, dtype):
    name = normalize_dtype(str(dtype))

    if hasattr(value, "item"):
        value = value.item()

    if name in {"float16", "bfloat16", "float32", "float64"}:
        return ctypes.c_double(float(value))

    ctype = {
        "bool": ctypes.c_int8,
        "int8": ctypes.c_int8,
        "uint8": ctypes.c_uint8,
        "int16": ctypes.c_int16,
        "uint16": ctypes.c_uint16,
        "int32": ctypes.c_int32,
        "uint32": ctypes.c_uint32,
        "int64": ctypes.c_int64,
        "uint64": ctypes.c_uint64,
    }.get(name)

    if ctype is None:
        raise TypeError(f"Unsupported Triton AOT scalar dtype: {dtype!r}.")
    return ctype(value)


def _triton_aot_ctype(binding, specs):
    if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}:
        return ctypes.c_void_p

    if binding.kind == "scalar":
        name = normalize_dtype(str(specs[binding.source].dtype))

        if name in {"float16", "bfloat16", "float32", "float64"}:
            return ctypes.c_double

        ctype = {
            "bool": ctypes.c_int8,
            "int8": ctypes.c_int8,
            "uint8": ctypes.c_uint8,
            "int16": ctypes.c_int16,
            "uint16": ctypes.c_uint16,
            "int32": ctypes.c_int32,
            "uint32": ctypes.c_uint32,
            "int64": ctypes.c_int64,
            "uint64": ctypes.c_uint64,
        }.get(name)

        if ctype is None:
            raise TypeError(f"Unsupported Triton AOT scalar dtype: {name!r}.")
        return ctype
    return ctypes.c_int64


def _validate_aot_constants(abi, public) -> None:
    for binding in abi.kernel_args:
        if binding.kind not in {"constexpr", "meta"} or binding.value is None:
            continue

        if binding.source not in public:
            continue

        actual = public[binding.source]

        if hasattr(actual, "item"):
            actual = actual.item()

        if actual != binding.value:
            raise ValueError(
                f"Kernel argument `{binding.source}` is specialized to "
                f"{binding.value!r}, but received {actual!r}."
            )


__all__ = ["TritonMaterializer"]
