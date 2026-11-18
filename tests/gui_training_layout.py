"""原生训练页布局检查，不训练、不访问摄像头、不改动用户数据。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import json
import tkinter as tk
import traceback
from unittest.mock import patch
from main import VigilScreen


def main():
    root=tk.Tk();errors=[];result={};stage=[0]
    root.report_callback_exception=lambda *args:errors.append(str(args[1]))
    with patch('cv2.VideoCapture') as camera:
        app=VigilScreen(root)
        studio=app.studio
        app.notebook.select(1)
        studio.notebook.select(2)
        root.geometry('1200x850')
        studio.history=[{'epoch':i,'train':{'loss':1/i,'accuracy':.95},
                                      'val':{'loss':1.2/i,'accuracy':.9}} for i in range(1,31)]
        studio.append_log(''.join(f'Epoch {i}: '+('/long/example/path/'*20)+'\n' for i in range(100)))
        def tick():
            try:
                if stage[0]==0:
                    assert studio.log.winfo_viewable()
                    result['initial_log_height']=studio.log.winfo_height()
                    studio.set_training_split(.25)
                    stage[0]=1
                elif stage[0]==1:
                    result['expanded_log_height']=studio.log.winfo_height()
                    assert result['expanded_log_height']>result['initial_log_height']
                    assert studio.log.xview()[1]<1
                    assert studio.log.yview()[0]>0
                    studio.set_training_split(.5)
                    root.geometry('1040x680')
                    stage[0]=2
                elif stage[0]==2:
                    studio.set_training_split(.25)
                    stage[0]=3
                elif stage[0]==3:
                    result['small_window_log_height']=studio.log.winfo_height()
                    assert studio.log.winfo_height()>=60,result
                    assert studio.chart.winfo_height()>=80,result
                    # 分隔条实际坐标可移动，恢复大窗口后保留可调整性。
                    root.geometry('1200x850')
                    stage[0]=4
                elif stage[0]==4:
                    panes=studio.training_panes
                    position=panes.sash_coord(0)[1]
                    panes.sash_place(0,0,position-50)
                    result['before_drag']=studio.log.winfo_height()
                    stage[0]=5
                else:
                    assert studio.log.winfo_height()>result['before_drag']
                    camera.assert_not_called()
                    app.close()
                    return
                root.after(200,tick)
            except Exception:
                errors.append(traceback.format_exc())
                app.close()
        root.after(350,tick)
        root.mainloop()
    assert not errors,errors
    print(json.dumps({'ok':True,'camera_accessed':False,**result}))


if __name__=='__main__':
    main()
