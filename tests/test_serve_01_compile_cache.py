import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import serve
import serve_01_compile_cache as cached


class CompileCacheServerTests(unittest.TestCase):
    def test_engine_command_matches_baseline(self):
        baseline_command = serve.build_command(serve.parse_args(["--dry-run"]))
        cached_command = cached.build_command(cached.parse_args(["--dry-run"]))

        self.assertEqual(cached_command, baseline_command)

    @patch("serve_01_compile_cache.os.execvpe")
    @patch("serve_01_compile_cache.shutil.which", return_value="/venv/bin/vllm")
    def test_compile_cache_is_enabled_and_persistent(self, _which, execvpe):
        with tempfile.TemporaryDirectory() as cache_root:
            with patch.dict(os.environ, {"VLLM_DISABLE_COMPILE_CACHE": "1"}):
                self.assertEqual(cached.main(["--cache-root", cache_root]), 0)

        environment = execvpe.call_args.args[2]
        self.assertNotIn("VLLM_DISABLE_COMPILE_CACHE", environment)
        self.assertEqual(environment["VLLM_CACHE_ROOT"], str(Path(cache_root).resolve()))
        self.assertEqual(environment["VLLM_NO_USAGE_STATS"], "1")
        self.assertEqual(environment["VLLM_USE_FLASHINFER_SAMPLER"], "0")


if __name__ == "__main__":
    unittest.main()
