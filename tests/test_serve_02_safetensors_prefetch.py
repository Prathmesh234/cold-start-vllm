import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import serve_01_compile_cache as cached
import serve_02_safetensors_prefetch as prefetched


class SafetensorsPrefetchServerTests(unittest.TestCase):
    def test_only_command_change_is_prefetch_strategy(self):
        cached_command = cached.build_command(cached.parse_args(["--dry-run"]))
        prefetch_command = prefetched.build_command(
            prefetched.parse_args(["--dry-run"])
        )

        self.assertEqual(
            prefetch_command,
            cached_command + ["--safetensors-load-strategy", "prefetch"],
        )

    @patch("serve_02_safetensors_prefetch.os.execvpe")
    @patch(
        "serve_02_safetensors_prefetch.shutil.which",
        return_value="/venv/bin/vllm",
    )
    def test_cache_and_compatibility_settings_are_preserved(self, _which, execvpe):
        with tempfile.TemporaryDirectory() as cache_root:
            with patch.dict(os.environ, {"VLLM_DISABLE_COMPILE_CACHE": "1"}):
                self.assertEqual(prefetched.main(["--cache-root", cache_root]), 0)

        environment = execvpe.call_args.args[2]
        self.assertNotIn("VLLM_DISABLE_COMPILE_CACHE", environment)
        self.assertEqual(environment["VLLM_CACHE_ROOT"], str(Path(cache_root).resolve()))
        self.assertEqual(environment["VLLM_NO_USAGE_STATS"], "1")
        self.assertEqual(environment["VLLM_USE_FLASHINFER_SAMPLER"], "0")


if __name__ == "__main__":
    unittest.main()
