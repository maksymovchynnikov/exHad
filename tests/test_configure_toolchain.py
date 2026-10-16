"""Installed Pythia may require ABI macros and non-default include directories."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('configure', Path(__file__).resolve().parents[1] / 'tools/configure.py')
configure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configure)


class ToolchainTests(unittest.TestCase):
    def test_installed_pythia_flags_reach_every_helper(self):
        replies = {'--cxxflags': '-I/custom/include -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11',
                   '--ldflags': '-L/custom/lib -lpythia8 -Wl,-rpath,/custom/lib',
                   '--includedir': '/custom/include\n'}
        with patch.object(configure.subprocess, 'check_output', side_effect=lambda args, **kw: replies[args[1]]):
            compiler, includes, linker = configure.toolchain({'config': '/custom/bin/pythia8-config'})
        self.assertIn('-D_GLIBCXX_USE_CXX11_ABI=0', compiler)
        self.assertGreater(compiler.index('-std=c++17'), compiler.index('-std=c++11'))
        self.assertIn('-I/custom/include', includes)
        self.assertIn('-Wl,-rpath,/custom/lib', linker)
        self.assertIn('-pthread', linker)

    def test_prefix_only_installation(self):
        compiler, includes, linker = configure.toolchain({'prefix': '/prefix with spaces', 'libdir': '/prefix with spaces/lib'})
        self.assertIn('-I/prefix with spaces/include', compiler)
        self.assertIn('-I/prefix with spaces/include', includes)
        self.assertIn('-L/prefix with spaces/lib', linker)


if __name__ == '__main__':
    unittest.main()
