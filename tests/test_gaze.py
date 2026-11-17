if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from PIL import Image
from gaze import GazeEstimator, GazeResult, evaluate_eyes


def eye_points(horizontal=.5, vertical=.5, closed=False):
    points=np.full((478,2),.5)
    for left,right,upper,lower,iris,x in [(33,133,159,145,468,.1),(362,263,386,374,473,.6)]:
        height=.01 if closed else .08
        points[left]=(x,.5)
        points[right]=(x+.2,.5)
        points[upper]=(x+.1,.5-height/2)
        points[lower]=(x+.1,.5+height/2)
        points[iris]=(x+.2*horizontal,.5-height/2+height*vertical)
    return points


class GazeTests(unittest.TestCase):
    def test_screen_and_camera_references_are_separate_and_closed_eyes_never_pass(self):
        estimator = GazeEstimator.__new__(GazeEstimator)
        estimator.reference = None
        estimator.screen_reference = None
        estimator.horizontal,estimator.vertical,estimator.min_eye_width = .12,.25,12
        estimator.mp = SimpleNamespace(Image=Mock(side_effect=lambda **kwargs:kwargs['data']),
                                       ImageFormat=SimpleNamespace(SRGB=1))
        estimator.landmarker = Mock()
        def set_points(closed=False):
            estimator.landmarker.detect.return_value = SimpleNamespace(face_landmarks=[
                [SimpleNamespace(x=x,y=y) for x,y in eye_points(.65,.5,closed)]])
        set_points()
        face = np.array([0,0,200,200,40,100,140,100,90,130,60,160,120,160,.99])
        picture = Image.new('RGB',(200,200))
        self.assertFalse(estimator.measure(picture,face).looking)
        estimator.calibrate([GazeResult(False,'test',((.65,.5),(.65,.5)))]*5,kind='screen')
        self.assertIsNone(estimator.reference)
        self.assertTrue(estimator.measure(picture,face).looking)
        self.assertEqual(estimator.measure(picture,face).reason,'视线接近已校准屏幕')
        estimator.calibrate([GazeResult(True,'test',((.5,.5),(.5,.5)))]*5)
        np.testing.assert_allclose(estimator.reference,((.5,.5),(.5,.5)))
        np.testing.assert_allclose(estimator.screen_reference,((.65,.5),(.65,.5)))
        set_points(closed=True)
        self.assertFalse(estimator.measure(picture,face).looking)

    def test_large_roi_resize_preserves_original_geometry_and_eye_width(self):
        estimator = GazeEstimator.__new__(GazeEstimator)
        estimator.reference = None
        estimator.horizontal,estimator.vertical,estimator.min_eye_width = .12,.25,200
        estimator.mp = SimpleNamespace(Image=Mock(side_effect=lambda **kwargs:kwargs['data']),
                                       ImageFormat=SimpleNamespace(SRGB=1))
        points = [SimpleNamespace(x=x,y=y) for x,y in eye_points()]
        estimator.landmarker = Mock()
        estimator.landmarker.detect.return_value = SimpleNamespace(face_landmarks=[points])
        face = np.array([200,100,800,800,248,570,868,570,558,700,400,850,700,850,.99])
        result = estimator.measure(Image.new('RGB',(1600,1200)),face)
        self.assertTrue(result.looking)
        network_input = estimator.landmarker.detect.call_args.args[0]
        self.assertEqual(max(network_input.shape[:2]),512)

    def test_centered_both_eyes_pass(self):
        result=evaluate_eyes(eye_points(),(200,200))
        self.assertTrue(result.looking)
        self.assertIsNotNone(result.ratios)

    def test_looking_left_right_up_down_and_closed_do_not_pass(self):
        for x,y,closed in [(.15,.5,False),(.85,.5,False),(.5,.1,False),(.5,.9,False),(.5,.5,True)]:
            with self.subTest(x=x,y=y,closed=closed):
                self.assertFalse(evaluate_eyes(eye_points(x,y,closed),(200,200)).looking)
        points=eye_points()
        points[468,0]=.12
        self.assertFalse(evaluate_eyes(points,(200,200)).looking)

    def test_unclear_and_tiny_eyes_do_not_pass(self):
        self.assertFalse(evaluate_eyes(np.zeros((468,2)),(200,200)).looking)
        self.assertFalse(evaluate_eyes(eye_points(),(30,30)).looking)
        points=eye_points();points[468]=float('nan')
        self.assertFalse(evaluate_eyes(points,(200,200)).looking)

    def test_reference_offsets_are_applied_to_both_eyes(self):
        self.assertFalse(evaluate_eyes(eye_points(.65,.5),(200,200)).looking)
        self.assertTrue(evaluate_eyes(eye_points(.65,.5),(200,200),reference=((.65,.5),(.65,.5))).looking)


if __name__=='__main__':
    unittest.main()
