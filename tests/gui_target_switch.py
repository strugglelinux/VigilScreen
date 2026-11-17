"""原生窗口目标切换验证；摄像头与身份模型均使用模拟对象。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import json
import time
import traceback
import tkinter as tk
from tkinter import ttk
from unittest.mock import Mock,patch
import numpy as np
from chi_camera import ALL_PEOPLE,ChiCamera,build_parser


def main():
    state={'opened':0,'released':0,'stopped_prompt':False,'loading_prompt':False}
    def camera(index):
        state['opened']+=1
        capture=Mock()
        capture.isOpened.return_value=True
        def read():
            time.sleep(.03)
            return True,np.zeros((240,320,3),dtype=np.uint8)
        def release():state['released']+=1
        capture.read.side_effect=read
        capture.release.side_effect=release
        return capture
    verifier=Mock(gallery={'chi':[], 'me':[]})
    def load(*args):
        time.sleep(.4)
        return verifier,{'classes':['chi','me']}
    root=tk.Tk()
    errors=[]
    with patch('cv2.VideoCapture',side_effect=camera),\
         patch('chi_camera.FaceFinder') as finder,\
         patch('chi_camera.GazeEstimator') as gaze,\
         patch('menu_bar.MenuBarIndicator') as indicator,\
         patch('chi_camera.save_hit') as save,\
         patch('identity_training.available_identities',return_value=['chi','me']),\
         patch('identity_training.load_profile',side_effect=load) as loader:
        finder.return_value.detect.return_value=[]
        app=ChiCamera(root,build_parser().parse_args([]),autostart=True)
        started=time.monotonic()
        stage=[0]
        def select(target):
            app.target_var.set(target)
            app.change_target()
            assert app.switching
            assert str(app.target_choice.cget('state'))=='disabled'
            assert app.switch_status.get()=='① 正在停止检测…'
            state['stopped_prompt']=True
        def tick():
            try:
                assert time.monotonic()-started<15,(stage,state,app.status.get())
                if stage[0]==0 and app.last_frame is not None:
                    select('chi')
                    assert app.args.target is None
                    stage[0]=1
                elif stage[0]==1:
                    if '正在切换目标' in app.switch_status.get():
                        state['loading_prompt']=True
                        assert app.args.target is None
                    if '③ 检测中' in app.switch_status.get():
                        assert app.args.target=='chi'
                        assert str(app.target_choice.cget('state'))=='readonly'
                        select(ALL_PEOPLE)
                        assert app.args.target=='chi'
                        stage[0]=2
                elif stage[0]==2 and '③ 检测中' in app.switch_status.get():
                    assert app.args.target is None
                    app.filter_button.invoke()
                    dialog=next(widget for widget in root.winfo_children() if isinstance(widget,tk.Toplevel))
                    controls=dialog.winfo_children()[0].winfo_children()
                    next(widget for widget in controls if isinstance(widget,ttk.Checkbutton) and widget.cget('text')=='chi').invoke()
                    next(widget for widget in controls if isinstance(widget,ttk.Button) and widget.cget('text')=='应用过滤').invoke()
                    assert app.switching or app._excluded==frozenset({'chi'})
                    stage[0]=3
                elif stage[0]==3 and '③ 检测中' in app.switch_status.get():
                    assert app._excluded==frozenset({'chi'})
                    assert 'chi' in app.filter_summary.get()
                    assert app.args.target is None
                    assert app.want_running
                    assert state['opened']==1,state
                    assert loader.call_count==1
                    gaze.assert_called_once()
                    save.assert_not_called()
                    indicator.return_value.show_hit.assert_not_called()
                    app.pause_monitor()
                    app.target_var.set('me')
                    app.change_target()
                    assert app.args.target=='me' and not app.want_running
                    assert '待开始' in app.switch_status.get()
                    stage[0]=4
                    app.close()
                    return
                root.after(25,tick)
            except Exception as error:
                errors.append(traceback.format_exc())
                app.close()
        root.after(50,tick)
        root.mainloop()
    assert not errors,errors
    assert stage[0]==4,state
    assert state['opened']==state['released']==1,state
    assert state['loading_prompt'],state
    print(json.dumps({'ok':True,'real_camera_accessed':False,**state},ensure_ascii=False))


if __name__=='__main__':
    main()
