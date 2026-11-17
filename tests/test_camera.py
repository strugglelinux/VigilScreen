"""离线验证摄像头采集、资源释放以及标记数据可训练。"""
if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from PIL import Image
from camera import CameraStream, save_capture, selection_box, prepare_capture
from data import build_datasets
from labeling import LabelStore


class CameraTests(unittest.TestCase):
    def test_selection_maps_letterbox_reverse_and_clamps(self):
        self.assertEqual(selection_box((150, 100), (350, 200), (50, 50, 400, 200), (800, 400)),
                         (200, 100, 600, 300))
        self.assertEqual(selection_box((350, 200), (150, 100), (50, 50, 400, 200), (800, 400)),
                         (200, 100, 600, 300))
        self.assertEqual(selection_box((-20, -20), (900, 900), (50, 50, 400, 200), (800, 400)),
                         (0, 0, 800, 400))
        with self.assertRaises(ValueError):
            selection_box((100, 100), (100, 100), (0, 0, 400, 200), (800, 400))

    def test_crop_resize_and_save_excludes_background(self):
        picture = Image.new('RGB', (800, 400), 'blue')
        picture.paste('red', (200, 100, 600, 300))
        with tempfile.TemporaryDirectory() as directory:
            target = save_capture(picture, Path(directory) / 'labels', 'target',
                                  box=(200, 100, 600, 300), max_side=128)
            with Image.open(target) as saved:
                self.assertEqual(saved.size, (128, 64))
                self.assertEqual(saved.getpixel((0, 0)), (255, 0, 0))
            raw = next((Path(directory) / 'labels_captures').glob('*.png'))
            with Image.open(raw) as saved:
                self.assertEqual(saved.size, (128, 64))
        self.assertEqual(picture.size, (800, 400))
        self.assertEqual(prepare_capture(picture, max_side=1600).size, picture.size)
        with self.assertRaises(ValueError):
            prepare_capture(picture, box=(-1, 0, 100, 100))
        with self.assertRaises(ValueError):
            prepare_capture(picture, max_side=-1)

    def test_capture_saved_as_dataset_and_can_be_relabeled(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'labeled'
            for category in ('cat', 'dog'):
                for _ in range(2):
                    target = save_capture(Image.new('RGB', (40, 30), 'red'), output, category)
                    self.assertTrue(target.is_file())
            store = LabelStore(output)
            self.assertEqual(len(store.labels), 4)
            raw = output.parent / 'labeled_captures'
            self.assertEqual(len(list(raw.glob('*.png'))), 4)
            bundle = build_datasets('folder', output)
            self.assertEqual(bundle.classes, ['cat', 'dog'])
            self.assertEqual(len(bundle.train), 2)
            self.assertEqual(len(bundle.val), 2)
            source = Path(next(iter(store.labels)))
            store.save(source, 'dog')
            self.assertEqual(LabelStore(output).labels[str(source)], 'dog')
            with self.assertRaises(ValueError):
                save_capture(Image.new('RGB', (4, 4)), output, '../bad')
            self.assertEqual(len(list(raw.glob('*.png'))), 4)

    def test_frames_are_rgb_and_camera_released_on_stop(self):
        stream = CameraStream()
        released = []
        def read():
            stream.stop()
            return True, np.array([[[0, 0, 255]]], dtype=np.uint8)
        capture = SimpleNamespace(isOpened=lambda: True, read=read,
                                  release=lambda: released.append(True))
        fake = SimpleNamespace(VideoCapture=lambda index: capture, COLOR_BGR2RGB=1,
                               cvtColor=lambda frame, code: frame[:, :, ::-1].copy())
        with patch.dict('sys.modules', {'cv2': fake}):
            stream.run()
        kind, frame = stream.events.get_nowait()
        self.assertEqual(kind, 'frame')
        self.assertEqual(frame.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(released, [True])

    def test_unavailable_camera_reports_error_and_releases(self):
        released = []
        capture = SimpleNamespace(isOpened=lambda: False, release=lambda: released.append(True))
        fake = SimpleNamespace(VideoCapture=lambda index: capture)
        with patch.dict('sys.modules', {'cv2': fake}):
            stream = CameraStream()
            stream.run()
        kind, error = stream.events.get_nowait()
        self.assertEqual(kind, 'error')
        self.assertIn('无法打开摄像头', error)
        self.assertEqual(released, [True])


if __name__=='__main__':
    unittest.main()
