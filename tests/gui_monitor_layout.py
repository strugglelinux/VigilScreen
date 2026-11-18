"""监控侧栏在小窗口、长文本和滚动后的可见性；不打开真实摄像头。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import json
import tkinter as tk
import traceback
from unittest.mock import patch

from main import VigilScreen


def main():
    root=tk.Tk();errors=[];stage=[0];result={}
    root.report_callback_exception=lambda *args:errors.append(str(args[1]))
    with patch('cv2.VideoCapture') as camera,patch('menu_bar.MenuBarIndicator'),\
         patch('main.attach_tray',return_value=None):
        app=VigilScreen(root);monitor=app.monitor
        root.geometry('1040x680')
        monitor.saved_summary.set('新命中 1 人\n'+'非常长的截图文件名_'*15+'.jpg')
        monitor.gaze_status.set('这是较长的视线校准状态说明。'*10)
        def footer_visible():
            for button in (monitor.start_button,monitor.exit_button):
                assert button.winfo_viewable()
                assert button.winfo_height()>=button.winfo_reqheight()
                assert button.winfo_rooty()+button.winfo_height()<=root.winfo_rooty()+root.winfo_height()
                assert button.winfo_rootx()+button.winfo_width()<=root.winfo_rootx()+root.winfo_width()
        def tick():
            try:
                footer_visible()
                if stage[0]==0:
                    assert monitor.sidebar_canvas.yview()[1]<1
                    result['small_window_exit_height']=monitor.exit_button.winfo_height()
                    target=monitor.target_var.get()
                    monitor.target_choice.event_generate('<MouseWheel>',delta=-120)
                    assert monitor.target_var.get()==target
                    assert monitor.sidebar_canvas.yview()[0]>0
                    monitor.sidebar_canvas.yview_moveto(1)
                    stage[0]=1
                elif stage[0]==1:
                    assert monitor.refresh_button.winfo_rooty()>=monitor.sidebar_canvas.winfo_rooty()
                    assert monitor.refresh_button.winfo_rooty()+monitor.refresh_button.winfo_height()<=\
                        monitor.sidebar_canvas.winfo_rooty()+monitor.sidebar_canvas.winfo_height()
                    root.geometry('1280x850')
                    monitor.saved_summary.set('尚无命中截图')
                    monitor.gaze_status.set('镜头采用默认参考。')
                    stage[0]=2
                elif stage[0]==2:
                    assert monitor.sidebar_canvas.yview()==(0.0,1.0)
                    root.geometry('1040x680')
                    stage[0]=3
                else:
                    camera.assert_not_called()
                    app.close()
                    return
                root.after(250,tick)
            except Exception:
                errors.append(traceback.format_exc());app.close()
        root.after(500,tick);root.mainloop()
    assert not errors,errors
    print(json.dumps({'ok':True,'real_camera_accessed':False,'fixed_footer':True,
                      'scroll_and_resize_verified':True,**result}))


if __name__=='__main__':
    main()
