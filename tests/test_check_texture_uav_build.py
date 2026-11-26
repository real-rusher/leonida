"""check.py's cs_texture_uav case: a build without the Windows runtime skips it, a full build still fails without it.

d3d12_texture_uav_test is built only with the Windows runtime. A build of the standalone Vulkan tools
(LEONIDA_BUILD_WINDOWS_RUNTIME=OFF, docs/testing.md) and build_runtime.ps1's cl.exe fallback cannot have it, so
"missing d3d12_texture_uav_test.exe" was a failure in every such run (#2204). A full build that lost the program
next to its other D3D12 tests still fails.

    python -I tests/test_check_texture_uav_build.py
"""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
import check  # noqa: E402


def touch(folder, name):
    with open(os.path.join(folder, name), "wb"):
        pass


class TextureUavBuild(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tools = self._tmp.name
        self.spv = os.path.join(self.tools, "cs_texture_uav.spv")
        touch(self.tools, "vk_compute_test.exe")   # what a standalone build has

    def tearDown(self):
        self._tmp.cleanup()

    def on_windows(self, **run_result):
        """texture_uav_test as on Windows; os.name is only switched for this call."""
        run = mock.patch.object(check, "run", return_value=run_result.get("returns", (0, "")))
        with mock.patch.object(check, "EXE", ".exe"), run as fake_run, mock.patch.object(check.os, "name", "nt"):
            problem = check.texture_uav_test(self.tools, self.spv)
        return problem, fake_run

    def test_a_build_without_the_windows_runtime_skips(self):
        problem, fake_run = self.on_windows()
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("this build has no d3d12_*_test programs", problem)
        fake_run.assert_not_called()

    def test_the_cl_fallback_build_skips_too(self):
        for name in ("root_sig_test.exe", "device_proxy_test.exe", "gta6rt.exe"):
            touch(self.tools, name)
        problem, _ = self.on_windows()
        self.assertIsInstance(problem, check.Skip)

    def test_a_full_build_without_the_program_still_fails(self):
        touch(self.tools, "d3d12_uav_clear_test.exe")
        problem, fake_run = self.on_windows()
        self.assertNotIsInstance(problem, check.Skip)
        self.assertEqual(problem, "missing " + os.path.join(self.tools, "d3d12_texture_uav_test.exe"))
        fake_run.assert_not_called()

    def test_a_present_program_is_run_and_judged(self):
        touch(self.tools, "d3d12_texture_uav_test.exe")
        problem, fake_run = self.on_windows(returns=(0, "ok"))
        self.assertEqual(problem, "")
        fake_run.assert_called_once_with(os.path.join(self.tools, "d3d12_texture_uav_test.exe"), self.spv)
        problem, _ = self.on_windows(returns=(check.UNSUPPORTED, "SKIP: no WARP adapter"))
        self.assertIsInstance(problem, check.Skip)
        self.assertEqual(problem, "SKIP: no WARP adapter")

    def test_other_systems_skip_as_before(self):
        with mock.patch.object(check.os, "name", "posix"):
            problem = check.texture_uav_test(self.tools, self.spv)
        self.assertEqual(problem, "SKIP (needs the Windows D3D12 runtime)")
        self.assertIsInstance(problem, check.Skip)


if __name__ == "__main__":
    unittest.main()
