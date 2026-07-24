import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import serve_01_compile_cache as cached
import serve_03_instanttensor as instant


class InstantTensorServerTests(unittest.TestCase):
    def test_only_command_change_is_instanttensor_loader(self):
        cached_command = cached.build_command(cached.parse_args(["--dry-run"]))
        instant_command = instant.build_command(instant.parse_args(["--dry-run"]))

        self.assertEqual(
            instant_command,
            cached_command + ["--load-format", "instanttensor"],
        )

    @patch("serve_03_instanttensor.os.execvpe")
    @patch("serve_03_instanttensor.shutil.which", return_value="/venv/bin/vllm")
    def test_cache_and_compatibility_settings_are_preserved(self, _which, execvpe):
        with tempfile.TemporaryDirectory() as cache_root:
            with patch.dict(os.environ, {"VLLM_DISABLE_COMPILE_CACHE": "1"}):
                self.assertEqual(instant.main(["--cache-root", cache_root]), 0)

        environment = execvpe.call_args.args[2]
        self.assertNotIn("VLLM_DISABLE_COMPILE_CACHE", environment)
        self.assertEqual(environment["VLLM_CACHE_ROOT"], str(Path(cache_root).resolve()))
        self.assertEqual(environment["VLLM_NO_USAGE_STATS"], "1")
        self.assertEqual(environment["VLLM_USE_FLASHINFER_SAMPLER"], "0")


if __name__ == "__main__":
    unittest.main()
