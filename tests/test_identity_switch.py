if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock,patch
import numpy as np
from PIL import Image
from chi_camera import ALL_PEOPLE,ChiCamera,build_parser
from gaze import GazeResult


FACE=np.array([0,0,30,40,5,10,25,10,15,22,8,32,22,32,.99],dtype=np.float32)


class IdentitySwitchTests(unittest.TestCase):
    def worker(self,directory):
        app=ChiCamera.__new__(ChiCamera)
        app.args=build_parser().parse_args([])
        app.args.profile=Path(directory)/'profile.json'
        app.args.output_dir=Path(directory)/'screenshots'
        app._scope=(0,None)
        app.events=queue.Queue()
        app.frames=queue.Queue(maxsize=1)
        app.stop_event=threading.Event()
        app.refresh_request=threading.Event()
        app.calibration_request=threading.Event()
        app.active_camera=None
        return app

    def test_switch_reuses_camera_gaze_and_identity_model(self):
        with tempfile.TemporaryDirectory() as directory:
            app=self.worker(directory)
            picture=Image.new('RGB',(80,60))
            modes=iter(enumerate([None,'chi','chi','me','me',None,None]))
            def next_frame(**kwargs):
                try:index,target=next(modes)
                except StopIteration:return None
                if target!=app.args.target:
                    revision=app.current_scope()[0]+1
                    app._scope=(revision,app.args.target)
                    app._pending_switch=(revision,target)
                    app.switching=True
                return picture,time.monotonic(),index+1
            capture=Mock(read=Mock(side_effect=next_frame))
            gaze=Mock(measure=Mock(return_value=GazeResult(False,'test')))
            verifier=Mock()
            verifier.gallery={'chi':[], 'me':[]}
            verifier.identify.side_effect=lambda *args:(app.args.target,.9)
            with patch('latest_camera.LatestCamera',return_value=capture) as camera_ctor,\
                 patch('chi_camera.FaceFinder') as detector_ctor,\
                 patch('chi_camera.GazeEstimator',return_value=gaze) as gaze_ctor,\
                 patch('identity_training.load_profile',return_value=(verifier,{'classes':['chi','me']})) as loader:
                detector_ctor.return_value.detect.return_value=[FACE]
                app.run_session()
            camera_ctor.assert_called_once()
            capture.start.assert_called_once()
            capture.close.assert_called_once()
            gaze_ctor.assert_called_once()
            gaze.close.assert_called_once()
            loader.assert_called_once()
            self.assertEqual(verifier.identify.call_count,2)
            self.assertEqual(gaze.reset_calibration.call_count,3)
            self.assertFalse([value for kind,value in app.events.queue if kind=='error'])

    def test_scope_change_during_gaze_discards_old_hit_even_when_target_value_returns_to_same(self):
        for target in ['chi',None]:
            with self.subTest(target=target),tempfile.TemporaryDirectory() as directory:
                app=self.worker(directory)
                picture=Image.new('RGB',(80,60))
                capture=Mock(read=Mock(side_effect=[(picture,time.monotonic(),1),None]))
                def switch(*args):
                    app.args.target=target
                    app._scope=(2,target)
                    app.events=queue.Queue()
                    app.frames=queue.Queue(maxsize=1)
                    return GazeResult(True,'旧视线结果')
                with patch('latest_camera.LatestCamera',return_value=capture),\
                     patch('chi_camera.FaceFinder') as finder,\
                     patch('chi_camera.GazeEstimator',return_value=Mock(measure=Mock(side_effect=switch))):
                    finder.return_value.detect.return_value=[FACE]
                    app.run_session()
                self.assertTrue(app.events.empty())
                self.assertTrue(app.frames.empty())
                self.assertFalse(app.args.output_dir.exists())

    def test_selection_handler_keeps_video_and_never_requests_full_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            app=self.worker(directory)
            app.args.target='chi';app._scope=(0,'chi')
            app.target_var=Mock(get=Mock(return_value='me'))
            app.available_targets=(ALL_PEOPLE,'chi','me')
            app.refreshing=False;app.want_running=True
            app.menu_indicator=Mock()
            app.last_frame=(Image.new('RGB',(80,60)),[])
            app.preview=Mock()
            app.draw_frame=Mock();app.status=Mock();app.gaze_status=Mock()
            app.target_choice=Mock();app.refresh_button=Mock();app.start_button=Mock();app.switch_status=Mock()
            app.request_refresh=Mock()
            app.events.put(('hit',('old.jpg',1)))
            app.change_target()
            app.request_refresh.assert_not_called()
            self.assertEqual(app.args.target,'chi')
            self.assertEqual(app.current_scope(),(1,'chi'))
            self.assertEqual(app._pending_switch,(1,'me'))
            self.assertTrue(app.switching)
            self.assertTrue(app.events.empty())
            self.assertFalse(app.refresh_request.is_set())
            app.draw_frame.assert_not_called()
            app.preview.delete.assert_called_once_with('face_overlay')

    def test_switch_stops_old_detection_before_applying_target_and_restarts_on_fresh_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            app=self.worker(directory)
            app.target_var=Mock(get=Mock(return_value='chi'))
            app.available_targets=(ALL_PEOPLE,'chi')
            app.refreshing=False;app.want_running=True
            app.menu_indicator=Mock();app.last_frame=None
            app.target_choice=Mock();app.refresh_button=Mock();app.start_button=Mock()
            app.switch_status=Mock();app.status=Mock();app.gaze_status=Mock()
            order=[]
            picture=Image.new('RGB',(80,60))
            capture=Mock(read=Mock(side_effect=[(picture,time.monotonic(),n) for n in (1,2,3)]+[None]))
            def measure(*args):
                if not order:
                    order.append('old detection')
                    app.change_target()
                    self.assertIsNone(app.args.target)
                    self.assertTrue(app.switching)
                    order.append('old detection finished')
                else:
                    self.assertEqual(app.args.target,'chi')
                    self.assertFalse(app.switching)
                    order.append('new detection')
                return GazeResult(True,'test')
            verifier=Mock();verifier.gallery={'chi':[]};verifier.identify.return_value=('chi',.9)
            def load(*args):
                self.assertEqual(order[-1],'old detection finished')
                order.append('apply target')
                return verifier,{'classes':['chi']}
            with patch('latest_camera.LatestCamera',return_value=capture),\
                 patch('chi_camera.FaceFinder') as finder,\
                 patch('chi_camera.GazeEstimator',return_value=Mock(measure=Mock(side_effect=measure))),\
                 patch('identity_training.load_profile',side_effect=load):
                finder.return_value.detect.return_value=[FACE]
                app.run_session()
            self.assertEqual(order,['old detection','old detection finished','apply target','new detection'])
            events=list(app.events.queue)
            kinds=[kind for kind,value in events]
            self.assertLess(kinds.index('switch_progress'),kinds.index('switch_complete'))
            self.assertEqual(kinds.count('hit'),1)
            self.assertEqual(len(list(app.args.output_dir.glob('*.jpg'))),1)
            self.assertFalse([value for kind,value in events if kind=='error'])

    def test_cancel_pending_switch_keeps_old_target(self):
        with tempfile.TemporaryDirectory() as directory:
            app=self.worker(directory)
            app._pending_switch=(1,'chi');app.switching=True
            app.cancel_target_switch()
            self.assertIsNone(app._pending_switch)
            self.assertFalse(app.switching)
            self.assertIsNone(app.args.target)

    def test_all_mode_with_exclusions_loads_identity_and_never_saves_excluded_box(self):
        with tempfile.TemporaryDirectory() as directory:
            app=self.worker(directory)
            app._excluded=frozenset({'chi'})
            picture=Image.new('RGB',(80,60))
            capture=Mock(read=Mock(side_effect=[(picture,time.monotonic(),1),None]))
            verifier=Mock(gallery={'chi':[], 'me':[]})
            verifier.identify.side_effect=[('chi',.9),('me',.9)]
            other=FACE.copy();other[0]+=40;other[4:14:2]+=40
            gaze=Mock(measure=Mock(return_value=GazeResult(True,'test')))
            with patch('latest_camera.LatestCamera',return_value=capture),\
                 patch('chi_camera.FaceFinder') as finder,\
                 patch('chi_camera.GazeEstimator',return_value=gaze),\
                 patch('identity_training.load_profile',return_value=(verifier,{'classes':['chi','me']})) as load,\
                 patch('chi_camera.save_hit',return_value=Path(directory)/'hit.jpg') as save:
                finder.return_value.detect.return_value=[FACE,other]
                app.run_session()
            load.assert_called_once()
            gaze.measure.assert_called_once()
            self.assertEqual(save.call_args.kwargs['face_boxes'],[(40,0,30,40)])
            self.assertEqual([value[1] for kind,value in app.events.queue if kind=='hit'],[1])


if __name__=='__main__':
    unittest.main()
