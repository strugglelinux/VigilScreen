"""本地原生窗口与模拟摄像头联调；不会打开真实摄像头。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import threading
import time
import tkinter as tk
from unittest.mock import Mock,patch
import numpy as np
from main import VigilScreen


def main():
    state = {'active':0,'maximum':0,'opened':0,'released':0}
    lock = threading.Lock()
    def camera(index):
        with lock:
            state['active']+=1
            state['opened']+=1
            state['maximum']=max(state['maximum'],state['active'])
        capture = Mock()
        capture.isOpened.return_value = True
        def read():
            time.sleep(.03)
            return True,np.zeros((240,320,3),dtype=np.uint8)
        def release():
            with lock:
                state['active']-=1
                state['released']+=1
        capture.read.side_effect = read
        capture.release.side_effect = release
        return capture
    root = tk.Tk()
    errors = []
    with patch('cv2.VideoCapture',side_effect=camera),patch('chi_camera.FaceFinder') as finder,patch('chi_camera.GazeEstimator'):
        finder.return_value.detect.return_value = []
        app = VigilScreen(root)
        started = time.monotonic()
        stage = [0]
        def tick():
            try:
                if time.monotonic()-started>20:
                    raise AssertionError((stage[0],state,app.monitor.status.get(),app.studio.camera_status.get()))
                if stage[0]==0:
                    assert state['opened']==0
                    assert not app.monitor.worker.is_alive()
                    assert not app.monitor.want_running
                    assert str(app.monitor.target_choice.cget('state'))=='readonly'
                    app.notebook.select(1)
                    stage[0]=1
                elif stage[0]==1:
                    assert state['opened']==0
                    app.notebook.select(0)
                    stage[0]=2
                elif stage[0]==2:
                    assert state['opened']==0
                    app.monitor.start_button.invoke()
                    stage[0]=3
                elif stage[0]==3 and state['opened']==1 and app.monitor.last_frame is not None:
                    assert app.monitor.worker.is_alive()
                    app.notebook.select(1)
                    stage[0]=4
                elif stage[0]==4 and not app.monitor.worker.is_alive():
                    assert state['active']==0
                    app.studio.open_camera()
                    stage[0]=5
                elif stage[0]==5 and state['opened']==2:
                    app.notebook.select(0)
                    stage[0]=6
                elif stage[0]==6 and state['active']==0:
                    assert state['opened']==2
                    assert not app.monitor.want_running
                    app.monitor.start_button.invoke()
                    stage[0]=7
                elif stage[0]==7 and state['opened']==3 and app.monitor.last_frame is not None:
                    assert state['maximum']==1
                    assert app.monitor.worker.is_alive()
                    app.close()
                    return
                root.after(100,tick)
            except Exception as error:
                errors.append(str(error))
                app.close()
        root.after(100,tick)
        root.mainloop()
    assert not errors,errors
    assert state['active']==0,state
    assert stage[0]==7,state
    print(json.dumps({'ok':True,'camera_accessed':False,**state},ensure_ascii=False))


if __name__=='__main__':
    main()
