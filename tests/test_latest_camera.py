"""验证推理消费最新帧、无重复确认、采集错误和资源释放。"""
if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from latest_camera import LatestCamera
from chi_camera import HitGate, build_parser


class LatestCameraTests(unittest.TestCase):
    def test_old_frames_are_discarded_and_same_frame_is_never_reused(self):
        camera = LatestCamera(0)
        for index in range(100):
            camera.publish(Image.new('RGB',(2,2),(index,0,0)), float(index))
        picture, timestamp, sequence = camera.read()
        self.assertEqual((timestamp,sequence),(99.,100))
        self.assertEqual(picture.getpixel((0,0)),(99,0,0))
        with self.assertRaisesRegex(RuntimeError,'超时'):
            camera.read(after=sequence,timeout=0)
        camera.publish(Image.new('RGB',(2,2),'blue'),100.)
        self.assertEqual(camera.read(after=sequence)[2],101)

    def test_capture_failure_is_reported_and_device_released(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False,None)
        fake = SimpleNamespace(VideoCapture=Mock(return_value=capture),CAP_PROP_BUFFERSIZE=38)
        camera = LatestCamera(0)
        with patch.dict('sys.modules',{'cv2':fake}):
            camera.start()
            with self.assertRaisesRegex(RuntimeError,'读取失败'):
                camera.read(timeout=1)
            camera.close()
        capture.release.assert_called_once()
        self.assertFalse(camera.thread.is_alive())

    def test_frames_are_rgb_and_capture_is_released_by_reader(self):
        capture = Mock()
        capture.isOpened.return_value = True
        camera = LatestCamera(0)
        def read():
            camera.stopped.set()
            return True,np.array([[[0,0,255]]],dtype=np.uint8)
        capture.read.side_effect = read
        fake = SimpleNamespace(VideoCapture=Mock(return_value=capture),CAP_PROP_BUFFERSIZE=38,
                               COLOR_BGR2RGB=1,cvtColor=lambda frame,code:frame[:,:,::-1].copy())
        with patch.dict('sys.modules',{'cv2':fake}):
            camera.run()
        self.assertEqual(camera.latest[0].getpixel((0,0)),(255,0,0))
        capture.release.assert_called_once()
        self.assertIsNone(camera.read())

    def test_two_current_positive_frames_hit_but_negative_resets_confirmation(self):
        self.assertEqual(build_parser().parse_args([]).confirmations,1)
        gate = HitGate(confirmations=2)
        for valid, expected in [(True,False),(False,False),(True,False),(True,True)]:
            self.assertEqual(gate.update(valid,0),expected)

    def test_default_hits_first_valid_frame_and_rearms_without_ten_second_delay(self):
        args = build_parser().parse_args(['--cooldown','0'])
        self.assertEqual(args.cooldown,0)
        gate = HitGate(args.confirmations,args.cooldown)
        self.assertFalse(gate.update(False,0))
        self.assertTrue(gate.update(True,.01))
        self.assertFalse(gate.update(True,.02))
        for now in (.03,.04,.05):
            self.assertFalse(gate.update(False,now))
        self.assertTrue(gate.update(True,.06))


if __name__=='__main__':
    unittest.main()
