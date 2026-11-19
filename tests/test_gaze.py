if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from PIL import Image
from gaze import (GazeEstimator,GazeResult,evaluate_eyes,head_pose,closest_reference,
                  matches_reference,calibration_location_matches)


def eye_points(horizontal=.5, vertical=.5, closed=False):
    points=np.full((478,2),.5)
    for left,right,upper,lower,iris,x in [(33,133,159,145,468,.1),(362,263,386,374,473,.6)]:
        height=.01 if closed else .08
        points[left]=(x,.5)
        points[right]=(x+.2,.5)
        pairs=[(160,144),(159,145),(158,153)] if iris==468 else [(385,380),(386,374),(387,373)]
        for position,(u,l) in zip((.07,.1,.13),pairs):
            points[u]=(x+position,.5-height/2)
            points[l]=(x+position,.5+height/2)
        center=np.array((x+.2*horizontal,.5-height/2+height*vertical))
        points[iris]=center
        for index,offset in enumerate(((.03,0),(0,.03),(-.03,0),(0,-.03)),iris+1):
            points[index]=center+offset
    return points


class GazeTests(unittest.TestCase):
    def estimator(self):
        estimator=GazeEstimator.__new__(GazeEstimator)
        estimator.horizontal,estimator.vertical,estimator.min_eye_width=.12,.25,12
        estimator.reset_calibration()
        return estimator

    def samples(self,x=.5,y=.5,pose=None,box=None):
        return [GazeResult(False,'sample',((x,y),(x,y)),pose,box) for _ in range(5)]

    def test_head_geometry_and_eye_features_tolerate_roll(self):
        landmarks=np.array([[40,50],[140,50],[90,90],[60,120],[120,120]],dtype=float)
        for angle in (-30,0,30):
            radians=np.deg2rad(angle)
            rotation=np.array([[np.cos(radians),-np.sin(radians)],
                               [np.sin(radians),np.cos(radians)]])
            rotated=(landmarks-[90,80])@rotation.T+[90,80]
            self.assertIsNotNone(head_pose(rotated))
            points=(eye_points()-.5)@rotation.T+.5
            self.assertTrue(evaluate_eyes(points,(200,200)).looking)
        landmarks[2,0]=145
        self.assertIsNone(head_pose(landmarks))
        self.assertIsNone(head_pose(np.full((5,2),np.nan)))

    def test_vertical_ratio_does_not_change_with_eyelid_aperture(self):
        points=eye_points(vertical=.65)
        original=evaluate_eyes(points,(200,200))
        for upper,lower in [(160,144),(159,145),(158,153),(385,380),(386,374),(387,373)]:
            points[upper,1]=.475;points[lower,1]=.525
        smaller=evaluate_eyes(points,(200,200))
        np.testing.assert_allclose(smaller.ratios,original.ratios)
        self.assertTrue(smaller.looking)

    def test_single_bad_eyelid_point_does_not_reverse_open_eye_result(self):
        points=eye_points();points[159,1]=.55
        self.assertTrue(evaluate_eyes(points,(200,200)).looking)

    def test_opposite_eye_residuals_and_unstable_iris_ring_never_pass(self):
        points=eye_points(horizontal=.4)
        points[473:478,0]+=.04
        result=evaluate_eyes(points,(200,200))
        self.assertFalse(result.looking)
        self.assertIsNotNone(result.ratios)
        points=eye_points();points[469:473]=points[468]
        self.assertIsNone(evaluate_eyes(points,(200,200)).ratios)

    def test_network_resolution_is_checked_without_upscaling_small_eyes(self):
        self.assertFalse(evaluate_eyes(eye_points(),(200,200),processing_size=(30,30)).looking)
        self.assertTrue(evaluate_eyes(eye_points(),(200,200),processing_size=(100,100)).looking)

    def test_screen_reference_interpolates_same_weights_for_both_eyes_without_extrapolation(self):
        refs=np.array([[[x,y],[x+.02,y-.01]] for x,y in ((.3,.2),(.7,.2),(.3,.8),(.7,.8))])
        for x,y in ((.3,.2),(.7,.8),(.4,.4),(.65,.6)):
            ratios=((x,y),(x+.02,y-.01))
            reference,weights=closest_reference(ratios,refs,.06,.125)
            self.assertTrue(matches_reference(ratios,reference,.06,.125))
            self.assertTrue(np.all(weights>=0))
            self.assertAlmostEqual(weights.sum(),1)
        away=((.9,.5),(.92,.49))
        reference,_=closest_reference(away,refs,.06,.125)
        self.assertFalse(matches_reference(away,reference,.06,.125))

    def test_calibration_rejects_unstable_samples_and_preserves_previous_reference(self):
        estimator=self.estimator();estimator.calibrate(self.samples())
        previous=estimator.reference.copy()
        bad=[self.samples(x)[0] for x in (.2,.35,.5,.65,.8)]
        with self.assertRaises(ValueError):estimator.calibrate(bad)
        np.testing.assert_allclose(estimator.reference,previous)
        with self.assertRaises(ValueError):estimator.calibrate([GazeResult(False,'闭眼')]*5)
        with self.assertRaises(ValueError):estimator.calibrate(self.samples()[:3])
        one_outlier=self.samples(.6);one_outlier[-1]=self.samples(.8)[0]
        estimator.calibrate(one_outlier)
        np.testing.assert_allclose(estimator.reference,((.6,.5),(.6,.5)))

    def test_screen_extension_requires_center_deduplicates_and_resets(self):
        estimator=self.estimator()
        with self.assertRaises(ValueError):estimator.calibrate(self.samples(.65),kind='screen_extend')
        estimator.calibrate(self.samples(.65),kind='screen')
        estimator.calibrate(self.samples(.65),kind='screen_extend')
        self.assertEqual(len(estimator.screen_samples),1)
        estimator.calibrate(self.samples(.35),kind='screen_extend')
        self.assertEqual(len(estimator.screen_samples),2)
        self.assertIsNone(estimator.reference)
        estimator.reset_calibration()
        self.assertIsNone(estimator.screen_reference)
        self.assertEqual(estimator.screen_samples,[])

    def test_calibration_rejects_pose_changes_or_person_position_changes(self):
        estimator=self.estimator()
        rows=self.samples(pose=(0,.4,0),box=(.1,.1,.2,.3))
        rows[-1]=self.samples(pose=(.2,.4,0),box=(.1,.1,.2,.3))[0]
        with self.assertRaises(ValueError):estimator.calibrate(rows)
        rows=self.samples(box=(.1,.1,.2,.3))
        rows[-1]=self.samples(box=(.6,.1,.2,.3))[0]
        with self.assertRaises(ValueError):estimator.calibrate(rows)
        self.assertTrue(calibration_location_matches((.12,.1,.2,.3),(.1,.1,.2,.3)))
        self.assertFalse(calibration_location_matches((.6,.1,.2,.3),(.1,.1,.2,.3)))

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

    def test_another_face_cannot_use_screen_calibration_from_different_location(self):
        estimator=self.estimator()
        estimator.mp=SimpleNamespace(Image=Mock(side_effect=lambda **kwargs:kwargs['data']),
                                     ImageFormat=SimpleNamespace(SRGB=1))
        points=[SimpleNamespace(x=x,y=y) for x,y in eye_points(.65)]
        estimator.landmarker=Mock(detect=Mock(return_value=SimpleNamespace(face_landmarks=[points])))
        estimator.calibrate(self.samples(.65,box=(.1,.1,.2,.2)),kind='screen')
        face=np.array([0,0,200,200,40,100,140,100,90,130,60,160,120,160,.99])
        self.assertFalse(estimator.measure(Image.new('RGB',(200,200)),face).looking)

    def test_nearby_mesh_with_wrong_eye_separation_is_rejected(self):
        estimator=self.estimator()
        estimator.mp=SimpleNamespace(Image=Mock(side_effect=lambda **kwargs:kwargs['data']),
                                     ImageFormat=SimpleNamespace(SRGB=1))
        points=[SimpleNamespace(x=x,y=y) for x,y in eye_points()]
        estimator.landmarker=Mock(detect=Mock(return_value=SimpleNamespace(face_landmarks=[points])))
        face=np.array([0,0,200,200,60,100,120,100,90,130,70,160,110,160,.99])
        self.assertFalse(estimator.measure(Image.new('RGB',(200,200)),face).looking)

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

    def test_small_target_zoom_preserves_original_pixel_gate(self):
        estimator=self.estimator()
        estimator.mp=SimpleNamespace(Image=Mock(side_effect=lambda **kwargs:kwargs['data']),
                                     ImageFormat=SimpleNamespace(SRGB=1))
        points=[SimpleNamespace(x=x,y=y) for x,y in eye_points()]
        estimator.landmarker=Mock(detect=Mock(return_value=SimpleNamespace(face_landmarks=[points])))
        base=np.array([0,0,200,200,40,100,140,100,90,130,60,160,120,160,.99])
        for size,expected in [(100,True),(30,False)]:
            with self.subTest(size=size):
                face=base.copy()
                face[:14]*=size/200
                result=estimator.measure(Image.new('RGB',(size,size)),face)
                self.assertEqual(result.looking,expected)
                self.assertEqual(max(estimator.landmarker.detect.call_args.args[0].shape[:2]),512)
                if not expected:
                    self.assertIn('视线无法确认',result.reason)

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
