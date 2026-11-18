"""检查图片与摄像头标记的操作栏、分隔条和裁剪；只使用临时生成的图片。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import tkinter as tk
from tkinter import ttk
import traceback
from unittest.mock import patch
from PIL import Image
from main import VigilScreen
from labeling import LabelStore


def assert_controls_visible(controls,panes):
    geometry={'controls_height':controls.winfo_height(),'controls_request':controls.winfo_reqheight(),
              'minimum':str(panes.panecget(controls,'minsize')),'panes_height':panes.winfo_height()}
    widgets=list(controls.winfo_children())
    checked=0
    while widgets:
        widget=widgets.pop();widgets.extend(widget.winfo_children())
        if isinstance(widget,(ttk.Button,ttk.Combobox,ttk.Spinbox)):
            assert widget.winfo_viewable(),(str(widget),geometry)
            assert widget.winfo_rooty()+widget.winfo_height()<=panes.winfo_rooty()+panes.winfo_height()+1,str(widget)
            assert widget.winfo_rootx()+widget.winfo_width()<=panes.winfo_rootx()+panes.winfo_width()+1,str(widget)
            assert widget.winfo_height()>=widget.winfo_reqheight(),(str(widget),geometry)
            checked+=1
    assert checked>=5
    return checked


def drag_divider(panes,x):
    start_x,_=panes.sash_coord(0)
    panes.event_generate('<ButtonPress-1>',x=start_x+2,y=30)
    panes.event_generate('<B1-Motion>',x=x+2,y=30)
    panes.event_generate('<ButtonRelease-1>',x=x+2,y=30)
    panes.update_idletasks()
    position=panes.sash_coord(0)[0]
    assert abs(position-x)<=2,(x,position)
    return position


def main():
    with tempfile.TemporaryDirectory() as directory:
        temporary=Path(directory).resolve();source=temporary/'sample.png'
        picture=Image.new('RGB',(400,300),'#456789');picture.save(source)
        root=tk.Tk();errors=[];result={};stage=[-1]
        root.report_callback_exception=lambda *args:errors.append(str(args[1]))
        with patch('cv2.VideoCapture') as camera,patch('gui.messagebox.showinfo'),\
             patch('gui.messagebox.showerror') as error_dialog:
            app=VigilScreen(root);studio=app.studio
            for name in ('chi','me'):
                (temporary/'labeled'/name).mkdir(parents=True)
            studio.identity_provider=lambda:('registered',)
            studio.checkpoints.set(str(temporary/'models'))
            studio.output.set(str(temporary/'labeled'))
            studio.category.set('正在输入的新身份')
            studio.store=LabelStore(temporary/'labeled');studio.images=[source]
            app.notebook.select(1);studio.notebook.select(0)
            root.geometry('1040x680')
            def tick():
                try:
                    if stage[0]==-1:
                        assert tuple(studio.categories.cget('values'))==('chi','me','registered')
                        assert tuple(studio.camera_categories.cget('values'))==('chi','me','registered')
                        assert studio.category.get()=='正在输入的新身份'
                        assert not app.monitor_hint.winfo_viewable()
                        assert app.actions_menu.entrycget(app.initialize_menu_index,'label')=='初始化…'
                        assert studio.preview.find_withtag('empty')
                        studio.show_image()
                        studio.label_roi=(40,30,240,180);studio.draw_label_image()
                        stage[0]=0
                    elif stage[0]==0:
                        result['image_controls']=assert_controls_visible(studio.label_controls,studio.label_panes)
                        result['small_image_preview_height']=studio.preview.winfo_height()
                        assert studio.preview.winfo_height()>=100
                        result['manual_divider']=drag_divider(studio.label_panes,540)
                        stage[0]=1
                    elif stage[0]==1:
                        assert studio.label_panes.sash_coord(0)[0]==result['manual_divider']
                        assert studio.label_roi==(40,30,240,180)
                        assert_controls_visible(studio.label_controls,studio.label_panes)
                        studio.category.set('alice');studio.save_label()
                        assert 'alice' in studio.camera_categories.cget('values')
                        saved=list((temporary/'labeled/alice').glob('*.png'))
                        assert saved
                        with Image.open(saved[0]) as crop:assert crop.size==(200,150)
                        studio.notebook.select(1)
                        studio.redraw_camera()
                        assert studio.camera_preview.find_withtag('empty')
                        studio.camera_frame=picture.copy();studio.freeze_camera()
                        assert studio.camera_snapshot is not None
                        studio.camera_roi=(50,50,250,200)
                        studio.camera_saved.set('上次保存：'+('/example/很长的目录/'*12)+'sample.png')
                        studio.redraw_camera()
                        stage[0]=2
                    elif stage[0]==2:
                        result['camera_controls']=assert_controls_visible(studio.camera_controls,studio.camera_panes)
                        assert studio.camera_preview.winfo_height()>=100
                        drag_divider(studio.camera_panes,540)
                        root.geometry('1200x850')
                        stage[0]=3
                    elif stage[0]==3:
                        drag_divider(studio.camera_panes,540)
                        stage[0]=4
                    elif stage[0]==4:
                        root.geometry('1040x680')
                        stage[0]=5
                    else:
                        assert_controls_visible(studio.camera_controls,studio.camera_panes)
                        assert studio.camera_roi==(50,50,250,200)
                        studio.camera_category.set('bob');studio.save_camera_label()
                        assert 'bob' in studio.categories.cget('values')
                        saved=list((temporary/'labeled/bob').glob('*.png'))
                        assert saved
                        with Image.open(saved[0]) as crop:assert crop.size==(200,150)
                        studio.use_annotation_data()
                        assert studio.data_dir.get()==studio.output.get()
                        assert studio.notebook.index(studio.notebook.select())==2
                        studio.output.set(str(temporary/'another_output'))
                        studio.category.set('保留输入')
                        studio.refresh_annotation_categories()
                        assert tuple(studio.categories.cget('values'))==('registered',)
                        assert tuple(studio.camera_categories.cget('values'))==('registered',)
                        assert studio.category.get()=='保留输入'
                        camera.assert_not_called();error_dialog.assert_not_called()
                        app.close()
                        return
                    root.after(250,tick)
                except Exception:
                    errors.append(traceback.format_exc());app.close()
            root.after(400,tick);root.mainloop()
        assert not errors,errors
        print(json.dumps({'ok':True,'real_camera_accessed':False,'user_data_modified':False,**result}))


if __name__=='__main__':
    main()
