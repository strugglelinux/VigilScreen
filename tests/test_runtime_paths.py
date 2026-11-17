import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import runtime_paths
from initialization import reset_project_data


class RuntimePathsTests(unittest.TestCase):
    def test_frozen_worker_relaunches_bundled_executable(self):
        with patch.object(sys,'frozen',True,create=True):
            self.assertEqual(runtime_paths.worker_command('train','--epochs',1),
                             [sys.executable,'--identity-worker','train','--epochs','1'])

    def test_source_worker_uses_real_source_script(self):
        with patch.object(sys,'frozen',False,create=True):
            command=runtime_paths.worker_command('validate')
        self.assertEqual(command[:2],[sys.executable,'-u'])
        self.assertTrue(Path(command[2]).is_file())
        self.assertEqual(command[3],'validate')

    def test_first_launch_relocates_seed_and_reset_never_restores_it(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();resources=base/'resources';target=base/'user data'
            profile=resources/'seed/checkpoints/identities/active_profile.json'
            profile.parent.mkdir(parents=True)
            profile.write_text('{"source":"__VIGILSCREEN_DATA__/data/labeled"}')
            with patch.object(sys,'frozen',True,create=True),\
                 patch.object(runtime_paths,'RESOURCE_ROOT',resources),\
                 patch.dict(os.environ,{'VIGILSCREEN_DATA_DIR':str(target)}):
                self.assertEqual(runtime_paths.initialize_app_data(),target)
                saved=target/'checkpoints/identities/active_profile.json'
                self.assertIn(str(target),saved.read_text())
                reset_project_data(target)
                runtime_paths.initialize_app_data()
                self.assertFalse(saved.exists())
                self.assertTrue((target/'.seed_initialized').exists())


if __name__=='__main__':
    unittest.main()
