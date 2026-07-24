"""Calculate the model-transfer floor for the NVIDIA GPU in this machine."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Sequence


MODEL_WEIGHT_SHARDS_BYTES = (
    4_792_272_488,
    4_798_702_184,
    4_170_342_232,
)
MODEL_WEIGHTS_BYTES = sum(MODEL_WEIGHT_SHARDS_BYTES)
BYTES_PER_GB = 1_000_000_000
PCIE_GT_PER_SECOND = {1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0, 5: 32.0}


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    uuid: str
    driver_version: str
    memory_mib: int
    current_pcie_generation: int
    current_pcie_width: int
    max_pcie_generation: int
    max_pcie_width: int


@dataclass(frozen=True)
class SpeedOfLight:
    gpu: GpuInfo
    weight_bytes: int
    max_pcie_gb_per_second: float
    minimum_h2d_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "gpu": asdict(self.gpu),
            "weight_bytes": self.weight_bytes,
            "weight_gb": self.weight_bytes / BYTES_PER_GB,
            "max_pcie_gb_per_second": self.max_pcie_gb_per_second,
            "minimum_h2d_seconds": self.minimum_h2d_seconds,
        }


def pcie_payload_gb_per_second(generation: int, width: int) -> float:
    """Return the theoretical one-way PCIe payload rate in decimal GB/s."""

    if generation not in PCIE_GT_PER_SECOND:
        raise ValueError(f"unsupported PCIe generation: {generation}")
    if width <= 0:
        raise ValueError("PCIe link width must be positive")

    encoding_efficiency = 0.8 if generation <= 2 else 128 / 130
    return PCIE_GT_PER_SECOND[generation] * encoding_efficiency * width / 8


def detect_gpu(index: int = 0) -> GpuInfo:
    """Read the selected GPU and its PCIe capabilities from nvidia-smi."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi was not found; an NVIDIA GPU is required")

    fields = (
        "index,name,uuid,driver_version,memory.total,"
        "pcie.link.gen.current,pcie.link.width.current,"
        "pcie.link.gen.max,pcie.link.width.max"
    )
    try:
        result = subprocess.run(
            [
                executable,
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
                f"--id={index}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or "nvidia-smi failed"
        raise RuntimeError(message) from error

    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != 9:
        raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout!r}")

    return GpuInfo(
        index=int(values[0]),
        name=values[1],
        uuid=values[2],
        driver_version=values[3],
        memory_mib=int(values[4]),
        current_pcie_generation=int(values[5]),
        current_pcie_width=int(values[6]),
        max_pcie_generation=int(values[7]),
        max_pcie_width=int(values[8]),
    )


def calculate_speed_of_light(
    gpu: GpuInfo, weight_bytes: int = MODEL_WEIGHTS_BYTES
) -> SpeedOfLight:
    """Calculate this GPU's theoretical host-RAM-to-VRAM weight-copy floor."""

    if weight_bytes <= 0:
        raise ValueError("weight_bytes must be positive")
    bandwidth = pcie_payload_gb_per_second(
        gpu.max_pcie_generation, gpu.max_pcie_width
    )
    return SpeedOfLight(
        gpu=gpu,
        weight_bytes=weight_bytes,
        max_pcie_gb_per_second=bandwidth,
        minimum_h2d_seconds=weight_bytes / (bandwidth * BYTES_PER_GB),
    )


def print_speed_of_light(estimate: SpeedOfLight) -> None:
    gpu = estimate.gpu
    print("GPU speed-of-light calculation")
    print(f"  GPU: {gpu.name} ({gpu.memory_mib / 1024:.1f} GiB)")
    print(f"  Driver: {gpu.driver_version}")
    print(f"  PCIe capability: Gen {gpu.max_pcie_generation} x{gpu.max_pcie_width}")
    if (
        gpu.current_pcie_generation != gpu.max_pcie_generation
        or gpu.current_pcie_width != gpu.max_pcie_width
    ):
        print(
            "  PCIe currently negotiated: "
            f"Gen {gpu.current_pcie_generation} x{gpu.current_pcie_width} "
            "(the link normally downclocks while idle)"
        )
    print(f"  Indexed model weights: {estimate.weight_bytes / BYTES_PER_GB:.3f} GB")
    print(
        "  Theoretical one-way PCIe payload: "
        f"{estimate.max_pcie_gb_per_second:.3f} GB/s"
    )
    print(f"  Absolute host-RAM -> GPU floor: {estimate.minimum_h2d_seconds:.3f} s")
    print(
        "  Excludes disk/network reads, deserialization, allocation, compilation, "
        "graph capture, and server startup."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        estimate = calculate_speed_of_light(detect_gpu(args.gpu_index))
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(estimate.as_dict(), indent=2, sort_keys=True))
    else:
        print_speed_of_light(estimate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
