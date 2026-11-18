"""校准中心、扩展边角与暂停清理的 GUI 联调，不打开真实摄像头。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import json
import time
import tkinter as tk
import traceback
from unittest.mock import Mock,patch
import numpy as np

from chi_camera import ChiCamera,build_parser
from gaze import GazeEstimator,GazeResult


def main():
    root=tk.Tk();errors=[];state={'opened':0,'released':0,'x':.65};stage=[0]
    root.report_callback_exception=lambda *args:errors.append(str(args[1]))
    estimator=GazeEstimator.__new__(GazeEstimator)
    estimator.horizontal,estimator.vertical=.12,.25;estimator.reset_calibration()
    def measure(*args):
        return GazeResult(False,'待校准',((state['x'],.5),(state['x'],.5)),(0,.5,0),(.1,.1,.2,.3))
    gaze=Mock(measure=Mock(side_effect=measure),calibrate=Mock(side_effect=estimator.calibrate))
    def camera(index):
        state['opened']+=1
        stream=Mock();stream.isOpened.return_value=True
        def read():
            time.sleep(.03)
            return True,np.zeros((240,320,3),dtype=np.uint8)
        stream.read.side_effect=read
        stream.release.side_effect=lambda:state.update(released=state['released']+1)
        return stream
    with patch('cv2.VideoCapture',side_effect=camera),patch('chi_camera.FaceFinder') as finder,\
         patch('chi_camera.GazeEstimator',return_value=gaze),patch('menu_bar.MenuBarIndicator'),\
         patch('tray_window.attach_tray',return_value=None),patch('chi_camera.save_hit') as save:
        finder.return_value.detect.return_value=[np.array([0,0,30,40,5,10,25,10,15,22,8,32,22,32,.99])]
        app=ChiCamera(root,build_parser().parse_args([]),autostart=True)
        started=time.monotonic()
        def tick():
            try:
                assert time.monotonic()-started<12,(stage,app.gaze_status.get())
                if stage[0]==0 and app.last_frame is not None:
                    app.request_calibration('screen_extend')
                    assert not app.calibration_request.is_set()
                    assert '请先校准屏幕中心' in app.gaze_status.get()
                    app.request_calibration('screen');stage[0]=1
                elif stage[0]==1 and app._screen_calibrated:
                    assert len(estimator.screen_samples)==1
                    state['x']=.8
                    app.request_calibration('screen_extend');stage[0]=2
                elif stage[0]==2 and gaze.calibrate.call_count==2 and not app.calibration_request.is_set():
                    assert len(estimator.screen_samples)==2
                    assert state['opened']==1
                    save.assert_not_called()
                    app.pause_monitor()
                    assert not app._screen_calibrated
                    app.close();return
                root.after(30,tick)
            except Exception:
                errors.append(traceback.format_exc());app.close()
        root.after(100,tick);root.mainloop()
    assert not errors,errors
    assert state['opened']==state['released']==1,state
    print(json.dumps({'ok':True,'real_camera_accessed':False,'screen_center_and_extension':True,
                      'pause_cleared_calibration_state':True}))


if __name__=='__main__':
    main()
