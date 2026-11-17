from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from initialization import reset_project_data


class InitializationTests(unittest.TestCase):
    def test_clears_training_identity_and_captures_but_preserves_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for relative in ('data/labeled/chi/face.png','data/labeled/.labels.json',
                             'checkpoints/identities/active_profile.json',
                             'checkpoints/identities/runs/run/gallery.npz',
                             'screenshots/hit.jpg','face_models/sface.onnx','main.py'):
                path=root/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('sample')
            reset_project_data(root)
            for name in ('data','checkpoints','screenshots'):
                self.assertFalse([p for p in (root/name).rglob('*') if p.is_file()])
            for relative in ('data/labeled','checkpoints/identities','screenshots'):
                self.assertTrue((root/relative).is_dir())
            self.assertEqual((root/'face_models/sface.onnx').read_text(),'sample')
            self.assertTrue((root/'main.py').is_file())
            reset_project_data(root)

    def test_top_level_and_nested_links_never_delete_external_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'project';root.mkdir()
            external=Path(directory)/'external';external.mkdir()
            original=external/'photo.png';original.write_text('keep')
            (root/'data').symlink_to(external,target_is_directory=True)
            (root/'checkpoints').mkdir()
            (root/'checkpoints'/'outside').symlink_to(external,target_is_directory=True)
            reset_project_data(root)
            self.assertEqual(original.read_text(),'keep')
            self.assertFalse((root/'data').is_symlink())

    def test_cleanup_failure_is_reported_instead_of_claiming_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data').mkdir()
            with patch('initialization.shutil.rmtree',side_effect=PermissionError('denied')):
                with self.assertRaises(PermissionError):
                    reset_project_data(root)


if __name__=='__main__':
    unittest.main()
