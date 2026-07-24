import subprocess
import unittest
from unittest.mock import patch

import gpu_speed_of_light as sol
import serve


def test_gpu() -> sol.GpuInfo:
    return sol.GpuInfo(
        index=0,
        name="NVIDIA RTX 6000 Ada Generation",
        uuid="GPU-test",
        driver_version="580.126.09",
        memory_mib=46068,
        current_pcie_generation=1,
        current_pcie_width=16,
        max_pcie_generation=4,
        max_pcie_width=16,
    )


class SpeedOfLightTests(unittest.TestCase):
    def test_repository_weight_size_is_sum_of_indexed_shards(self):
        self.assertEqual(sol.MODEL_WEIGHTS_BYTES, 13_761_316_904)

    def test_gen4_x16_payload_rate(self):
        self.assertAlmostEqual(sol.pcie_payload_gb_per_second(4, 16), 31.5076923077)

    def test_current_machine_floor_uses_max_not_idle_link(self):
        estimate = sol.calculate_speed_of_light(test_gpu())

        self.assertAlmostEqual(estimate.max_pcie_gb_per_second, 31.5076923077)
        self.assertAlmostEqual(estimate.minimum_h2d_seconds, 0.43676, places=5)

    @patch("gpu_speed_of_light.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("gpu_speed_of_light.subprocess.run")
    def test_gpu_detection(self, run, _which):
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "0, NVIDIA RTX 6000 Ada Generation, GPU-test, 580.126.09, "
                "46068, 1, 16, 4, 16\n"
            ),
            stderr="",
        )

        gpu = sol.detect_gpu()
        self.assertEqual(gpu.name, "NVIDIA RTX 6000 Ada Generation")
        self.assertEqual(gpu.max_pcie_generation, 4)


class CommandTests(unittest.TestCase):
    def test_basic_command_is_pinned_and_single_gpu(self):
        command = serve.build_command(serve.parse_args(["--dry-run"]))

        self.assertEqual(command[:3], ["vllm", "serve", serve.MODEL_ID])
        self.assertIn(serve.MODEL_REVISION, command)
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")

    def test_extra_vllm_arguments_are_forwarded(self):
        args = serve.parse_args(["--dry-run", "--", "--disable-log-stats"])
        self.assertEqual(serve.build_command(args)[-1], "--disable-log-stats")

    @patch("serve.os.execvpe")
    @patch("serve.shutil.which", return_value="/venv/bin/vllm")
    def test_server_disables_usage_stats_and_compile_cache(self, _which, execvpe):
        self.assertEqual(serve.main([]), 0)

        environment = execvpe.call_args.args[2]
        self.assertEqual(environment["VLLM_NO_USAGE_STATS"], "1")
        self.assertEqual(environment["VLLM_DISABLE_COMPILE_CACHE"], "1")
        self.assertEqual(environment["VLLM_USE_FLASHINFER_SAMPLER"], "0")


if __name__ == "__main__":
    unittest.main()
