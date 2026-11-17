"""初始化联调：只清理临时目录，绝不清空项目中的实际数据。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import time
import tkinter as tk
import traceback
from unittest.mock import Mock,patch
import numpy as np
from initialization import reset_project_data
from main import VigilScreen


def main():
    with tempfile.TemporaryDirectory() as directory:
        sandbox=Path(directory)
        for relative in ('data/labeled/me/face.png','checkpoints/identities/model.pt','screenshots/hit.jpg'):
            path=sandbox/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('test')
        camera_state={'opened':0,'released':0}
        def camera(index):
            camera_state['opened']+=1
            capture=Mock();capture.isOpened.return_value=True
            def read():
                time.sleep(.03)
                return True,np.zeros((240,320,3),dtype=np.uint8)
            capture.read.side_effect=read
            def release():camera_state['released']+=1
            capture.release.side_effect=release
            return capture
        root=tk.Tk()
        errors=[]
        root.report_callback_exception=lambda *args:errors.append('Tk callback failed: '+str(args[1]))
        def clear(project_root):
            assert not old_monitor.worker.is_alive()
            assert old_studio.process.poll() is not None
            assert old_monitor.closed and old_studio.closed
            reset_project_data(sandbox)
        with patch('cv2.VideoCapture',side_effect=camera),\
             patch('chi_camera.FaceFinder') as finder,\
             patch('chi_camera.GazeEstimator'),\
             patch('menu_bar.MenuBarIndicator'),\
             patch('identity_training.available_identities',return_value=[]),\
             patch('main.messagebox.askyesno',side_effect=[False,True]) as confirm,\
             patch('main.messagebox.showinfo') as info,\
             patch('main.messagebox.showerror') as failure,\
             patch('main.reset_project_data',side_effect=clear) as cleanup:
            finder.return_value.detect.return_value=[]
            app=VigilScreen(root)
            old_monitor=app.monitor;old_studio=app.studio
            started=time.monotonic();stage=[0]
            def tick():
                try:
                    assert time.monotonic()-started<20,(stage,app.app_status.get())
                    if stage[0]==0:
                        app.monitor.start_button.invoke()
                        stage[0]=1
                    elif stage[0]==1 and app.monitor.last_frame is not None:
                        app.initialize_button.invoke()  # 取消确认应当保持监控和全部数据。
                        assert not app.resetting and app.monitor.want_running
                        cleanup.assert_not_called()
                        assert (sandbox/'data/labeled/me/face.png').exists()
                        app.monitor._excluded=frozenset({'me'})
                        app.studio.category.set('me')
                        training=Mock()
                        training.poll.return_value=None
                        training.terminate.side_effect=lambda:setattr(training.poll,'return_value',0)
                        app.studio.process=training
                        app.initialize_button.invoke()
                        assert app.resetting
                        stage[0]=2
                    elif stage[0]==2 and not app.resetting:
                        failure.assert_not_called();info.assert_called_once()
                        assert confirm.call_count==2
                        cleanup.assert_called_once()
                        assert old_monitor.closed and old_studio.closed
                        old_studio.process.terminate.assert_called_once()
                        assert app.monitor is not old_monitor and app.studio is not old_studio
                        assert app.monitor.args.target is None and not app.monitor._excluded
                        assert not app.monitor.want_running and not app.monitor.worker.is_alive()
                        assert app.studio.category.get()=='' and not app.studio.history
                        assert app.studio.last_validation is None
                        assert not [p for p in sandbox.rglob('*') if p.is_file()]
                        assert camera_state['opened']==camera_state['released']==1
                        stage.append(time.monotonic())
                        stage[0]=3
                    elif stage[0]==3 and time.monotonic()-stage[1]>.4:
                        assert camera_state['opened']==1
                        app.monitor.start_button.invoke()
                        stage[0]=4
                    elif stage[0]==4 and camera_state['opened']==2 and app.monitor.last_frame is not None:
                        app.close()
                        return
                    root.after(50,tick)
                except Exception:
                    errors.append(traceback.format_exc())
                    app.resetting=False
                    if hasattr(app,'reset_dialog') and app.reset_dialog.winfo_exists():
                        app.reset_dialog.destroy()
                    app.close()
            root.after(100,tick)
            root.mainloop()
        assert not errors,errors
        assert stage[0]==4
        assert camera_state['opened']==camera_state['released']==2
        print(json.dumps({'ok':True,'actual_project_data_deleted':False,
                          'cancel_preserved_data_and_monitoring':True,'reset_cleared_temporary_data':True,
                          'restart_after_reset_verified':True,**camera_state}))


if __name__=='__main__':
    main()
