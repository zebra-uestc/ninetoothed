"""JIT materialization for Triton source executed by the Ascend driver."""

import importlib
import importlib.util
import re
from pathlib import Path

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.backends.materializers.triton_jit import materialize_triton_jit


class TritonAscendMaterializer(Materializer):
    target = Target.TRITON_ASCEND

    def jit_materialize(self, compilation, *, output_dir=None):
        del output_dir

        return materialize_triton_jit(
            compilation,
            environment_validator=_validate_ascend_jit_environment,
            prepare_environment=False,
        )

    def aot_build(self, compilation, *, output_dir: str | Path):
        del compilation, output_dir
        raise ValueError(
            "Triton-Ascend AOT materialization is not supported; use JIT mode."
        )

    def load_built_artifact(self, built: BuiltArtifact):
        del built
        raise ValueError(
            "Triton-Ascend built artifact reload is not supported in JIT-only mode."
        )


def _validate_ascend_jit_environment(compilation):
    """Validate the NPU driver boundary before importing generated source."""
    target = compilation.target

    if target is None or target.backend is not Target.TRITON_ASCEND:
        raise ValueError("Triton-Ascend materialization requires target triton-ascend.")

    if target.platform.accelerator_vendor != "ascend":
        raise ValueError(
            f"Triton-Ascend JIT requires an Ascend profile; got `{target.platform.name}`."
        )

    if target.platform.name not in {"ascend-910b3", "ascend-910b4"}:
        raise RuntimeError(
            "Triton-Ascend JIT requires a supported Ascend profile "
            "(`ascend-910b3` or `ascend-910b4`)."
        )

    try:
        import torch

        npu = getattr(torch, "npu", None)
        available = npu is not None and bool(npu.is_available())
    except (ImportError, RuntimeError, AttributeError) as exc:
        raise RuntimeError(
            "Triton-Ascend JIT requires torch-npu and an available NPU; "
            "source-only lowering remains available without materialization."
        ) from exc

    if not available:
        raise RuntimeError(
            "Triton-Ascend JIT requires an available NPU; source-only lowering "
            "remains available without materialization."
        )

    try:
        driver_spec = importlib.util.find_spec("triton.backends.ascend")
    except (ImportError, ModuleNotFoundError, ValueError):
        driver_spec = None

    if driver_spec is None:
        raise RuntimeError(
            "Triton-Ascend JIT driver is not installed; source-only lowering "
            "remains available without materialization."
        )

    try:
        importlib.import_module("triton.backends.ascend")
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        raise RuntimeError(
            "Triton-Ascend JIT driver could not be imported; source-only lowering "
            "remains available without materialization."
        ) from exc

    try:
        identity = _query_npu_identity(npu)
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise RuntimeError(
            "Triton-Ascend JIT could not query the active NPU/SoC; "
            "source-only lowering remains available without materialization."
        ) from exc

    device = identity["device_index"]
    actual_soc = identity.get("soc") or identity.get("device_name")

    if actual_soc is None:
        raise RuntimeError(
            "Triton-Ascend JIT could not determine the actual NPU SoC for "
            f"profile `{target.platform.name}`."
        )

    if _normalize_soc(actual_soc) != _normalize_soc(target.platform.accelerator_name):
        raise RuntimeError(
            "Triton-Ascend JIT profile/SoC mismatch: "
            f"profile `{target.platform.name}` expects `{target.platform.accelerator_name}`, "
            f"but device reports `{actual_soc}`."
        )

    try:
        stream = npu.current_stream(device)
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise RuntimeError(
            "Triton-Ascend JIT could not obtain the current NPU stream."
        ) from exc

    runtime = {
        "runtime": "triton-ascend",
        "device_type": "npu",
        "device_index": device,
        "stream": getattr(stream, "stream_id", getattr(stream, "cuda_stream", None)),
        "driver": str(driver_spec.origin or driver_spec.submodule_search_locations),
        "soc": str(actual_soc),
        "profile": target.platform.name,
    }

    return runtime


def _query_npu_identity(npu):
    """Read the current device and SoC for JIT profile validation."""
    index = int(npu.current_device())
    values = []

    for name in ("get_soc_version", "get_device_name"):
        getter = getattr(npu, name, None)

        if getter is not None:
            try:
                values.append(getter(index))
            except (AttributeError, RuntimeError, TypeError):
                pass

    getter = getattr(npu, "get_device_properties", None)

    if getter is not None:
        try:
            properties = getter(index)
            values.extend(
                getattr(properties, name, None)
                for name in ("soc_version", "name", "device_name")
            )
        except (AttributeError, RuntimeError, TypeError):
            pass

    values = [str(value) for value in values if value not in (None, "")]

    return {
        "device_index": index,
        "soc": values[0] if values else None,
        "device_name": values[1]
        if len(values) > 1
        else (values[0] if values else None),
    }


def _normalize_soc(value) -> str:
    normalized = "".join(
        character for character in str(value).lower() if character.isalnum()
    )
    normalized = normalized.removeprefix("ascend")
    match = re.search(r"910b[34]", normalized)

    return match.group(0) if match else normalized


__all__ = ["TritonAscendMaterializer"]
