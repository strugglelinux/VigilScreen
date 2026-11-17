if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

import tempfile
import queue
import threading
import time
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import Mock,patch

import numpy as np
from PIL import Image

from chi_camera import ChiCamera,HitGate,build_parser,recognize_faces,save_hit
from gaze import GazeResult
from tracking import PersonHits


def face(x=0,sideways=False):
    return np.array([x,0,30,40,x+5,10,x+25,10,x+(25 if sideways else 15),22,x+8,32,x+22,32,.99],dtype=np.float32)


def row(x,looking):
    return (x,10,30,40,'人脸',0.,looking,0.,True,GazeResult(looking,'test'))


class MonitoringTests(unittest.TestCase):
    def test_all_mode_exclusions_skip_gaze_but_keep_other_and_unknown_faces(self):
        verifier=Mock(gallery={'chi':[], 'me':[]})
        verifier.identify.side_effect=[('chi',.9),('me',.9),('unknown',.2)]
        gaze=Mock(measure=Mock(return_value=GazeResult(True,'镜头')))
        rows=recognize_faces(Image.new('RGB',(120,40)),[face(x) for x in (0,40,80)],
                             verifier=verifier,gaze=gaze,excluded={'chi'})
        self.assertEqual([r[6] for r in rows],[False,True,True])
        self.assertEqual(gaze.measure.call_count,2)
        self.assertIn('已过滤',rows[0][9].reason)
        self.assertEqual([int(call.args[1][0]) for call in gaze.measure.call_args_list],[40,80])

    def test_exclusion_overrides_selected_identity_and_requires_model(self):
        verifier=Mock(gallery={'chi':[]})
        verifier.identify.return_value=('chi',.9)
        gaze=Mock()
        rows=recognize_faces(Image.new('RGB',(40,40)),[face()],verifier=verifier,
                             target='chi',excluded={'chi'},gaze=gaze)
        self.assertFalse(rows[0][6])
        gaze.measure.assert_not_called()
        with self.assertRaises(ValueError):
            recognize_faces(Image.new('RGB',(40,40)),[face()],excluded={'chi'})

    def test_switch_cancels_remaining_faces_without_waiting_for_all_gaze_calls(self):
        cancelled=threading.Event()
        def measure(*args):
            cancelled.set()
            return GazeResult(True,'旧目标')
        gaze=Mock(measure=Mock(side_effect=measure))
        rows=recognize_faces(Image.new('RGB',(120,40)),[face(x) for x in (0,40,80)],
                             gaze=gaze,cancelled=cancelled.is_set)
        self.assertEqual(rows,[])
        gaze.measure.assert_called_once()

    def test_loading_profile_without_selected_person_never_silently_enables_all_people(self):
        app = ChiCamera.__new__(ChiCamera)
        app.args = SimpleNamespace(profile=Path('old.json'),target='alice')
        app.pause_monitor,app.reload_targets,app.status = Mock(),Mock(),Mock()
        with patch('identity_training.read_profile',return_value=(Path('new.json'),{'classes':['bob']})):
            app.set_profile(Path('new.json'))
        self.assertEqual(app.args.target,'alice')
        app.pause_monitor.assert_called_once()
    def make_worker(self,output):
        app = ChiCamera.__new__(ChiCamera)
        app.args = build_parser().parse_args([])
        app.args.profile = Path(output)/'does-not-exist.json'
        app.args.output_dir = Path(output)/'screenshots'
        app.events = queue.Queue()
        app.frames = queue.Queue(maxsize=1)
        app.stop_event = threading.Event()
        app.refresh_request = threading.Event()
        app.calibration_request = threading.Event()
        app.active_camera = None
        return app

    def test_complete_all_mode_worker_without_profile_saves_two_distinct_person_events(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_worker(directory)
            capture = Mock()
            picture = Image.new('RGB',(120,80),'white')
            capture.read.side_effect = [(picture,time.monotonic(),index) for index in (1,2,3)]+[None]
            gaze = Mock()
            gaze.measure.side_effect = [GazeResult(value,'test') for value in [True,False,True,True,True,True]]
            with patch('latest_camera.LatestCamera',return_value=capture),\
                 patch('chi_camera.FaceFinder') as finder,\
                 patch('chi_camera.GazeEstimator',return_value=gaze),\
                 patch('identity_training.load_profile',side_effect=AssertionError('不应加载身份模型')):
                finder.return_value.detect.return_value = [face(),face(60)]
                app.run_session()
            events = list(app.events.queue)
            self.assertFalse([value for kind,value in events if kind=='error'])
            hits = [value for kind,value in events if kind=='hit']
            self.assertEqual(len(hits),2)
            self.assertEqual([count for path,count in hits],[1,1])
            self.assertEqual(len(list(app.args.output_dir.glob('*.jpg'))),2)
            capture.close.assert_called_once()
            gaze.close.assert_called_once()

    def test_stop_during_gaze_does_not_publish_a_stale_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_worker(directory)
            capture = Mock()
            capture.read.return_value = (Image.new('RGB',(60,60)),time.monotonic(),1)
            def stop_while_measuring(*args):
                app.stop_event.set()
                app.events = queue.Queue()
                return GazeResult(True,'旧结果')
            gaze = Mock(measure=Mock(side_effect=stop_while_measuring))
            with patch('latest_camera.LatestCamera',return_value=capture),\
                 patch('chi_camera.FaceFinder') as finder,patch('chi_camera.GazeEstimator',return_value=gaze):
                finder.return_value.detect.return_value = [face()]
                app.run_session()
            self.assertTrue(app.events.empty())
            self.assertFalse(app.args.output_dir.exists())
            capture.close.assert_called_once()

    def test_default_all_mode_without_identity_model_checks_every_front_face(self):
        self.assertIsNone(build_parser().parse_args([]).target)
        gaze = Mock()
        gaze.measure.side_effect = [GazeResult(True,'镜头'),GazeResult(True,'屏幕')]
        verifier = Mock()
        verifier.identify.side_effect = AssertionError('全员模式不能要求身份匹配')
        rows = recognize_faces(Image.new('RGB',(80,40)),[face(),face(40)],verifier=verifier,gaze=gaze)
        self.assertEqual([r[4] for r in rows],['人脸','人脸'])
        self.assertEqual([r[6] for r in rows],[True,True])
        verifier.identify.assert_not_called()
        self.assertEqual(gaze.measure.call_count,2)

    def test_away_gaze_closed_or_unavailable_and_sideways_never_hit(self):
        for eyes in [None,Mock(measure=Mock(return_value=GazeResult(False,'闭眼'))),
                     Mock(measure=Mock(side_effect=RuntimeError('不可见')))]:
            rows = recognize_faces(Image.new('RGB',(80,40)),[face(),face(40,True)],gaze=eyes)
            self.assertFalse(any(r[6] for r in rows))

    def test_selected_identity_only_checks_that_person(self):
        verifier = Mock()
        verifier.gallery = {'chi':[],'me':[]}
        verifier.identify.side_effect = [('me',.9),('chi',.9),('unknown',.2)]
        gaze = Mock(measure=Mock(return_value=GazeResult(True,'镜头')))
        rows = recognize_faces(Image.new('RGB',(120,40)),[face(x) for x in (0,40,80)],
                               verifier=verifier,target='chi',gaze=gaze)
        self.assertEqual([r[6] for r in rows],[False,True,False])
        gaze.measure.assert_called_once()
        with self.assertRaises(ValueError):
            recognize_faces(Image.new('RGB',(40,40)),[face()],target='chi')

    def test_each_person_has_own_latch_and_face_order_does_not_matter(self):
        tracker = PersonHits(HitGate)
        hits,ids = tracker.update([row(0,True),row(80,False)],0)
        self.assertEqual(len(hits),1)
        first,second = ids
        hits,_ = tracker.update([row(0,True),row(80,True)],.1)
        self.assertEqual(hits,[second])
        hits,ids = tracker.update([row(80,True),row(0,True)],.2)
        self.assertEqual(hits,[])
        self.assertEqual(ids,[second,first])
        for now in (.3,.4,.5):
            tracker.update([row(0,False),row(80,True)],now)
        hits,_ = tracker.update([row(0,True),row(80,True)],.6)
        self.assertEqual(hits,[first])

    def test_simultaneous_people_both_hit_and_screenshot_has_both_red_boxes(self):
        tracker = PersonHits(HitGate)
        rows = [row(10,True),row(80,True)]
        self.assertEqual(len(tracker.update(rows,0)[0]),2)
        picture = Image.new('RGB',(140,80),'white')
        with tempfile.TemporaryDirectory() as directory:
            path = save_hit(picture,Path(directory),[r[:4] for r in rows if r[6]])
            with Image.open(path) as saved:
                for x in (11,81):
                    r,g,b = saved.getpixel((x,30))
                    self.assertGreater(r,180)
                    self.assertLess(g,90)
        self.assertEqual(picture.getpixel((11,30)),(255,255,255))


if __name__=='__main__':
    unittest.main()
