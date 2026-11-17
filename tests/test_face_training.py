if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from prepare_face_training import prepare
from test_classifier import make_images


class FaceTrainingTests(unittest.TestCase):
    def test_split_before_crop_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, output = root/'original', root/'faces'
            make_images(source, per_class=4)
            before = {p: p.read_bytes() for p in source.rglob('*.png')}
            with patch('prepare_face_training.FaceFinder') as finder:
                finder.return_value.find.return_value = [(0, 0, 20, 20)]
                report = prepare(source, output, root/'fake.onnx')
            train_sources = {s['source'] for s in report['samples'] if s['split'] == 'train'}
            val_sources = {s['source'] for s in report['samples'] if s['split'] == 'val'}
            self.assertFalse(train_sources & val_sources)
            self.assertEqual(len(report['samples']), 8)
            self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
            self.assertEqual(report['counts']['val'], {'cat': 1, 'dog': 1})
            with self.assertRaises(ValueError):
                prepare(source, output, root/'fake.onnx')


if __name__=='__main__':
    unittest.main()
