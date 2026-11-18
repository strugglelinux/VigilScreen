"""独立应用入口，统一窗口、训练子进程和离线打包自检。"""
import json
import os
import platform
from pathlib import Path
import sys
import traceback

from runtime_paths import RESOURCE_ROOT,data_root,initialize_app_data
from runtime_support import configure_process_environment,restore_worker_stream,worker_process_options


def require(condition,message='独立应用自检失败'):
    if not condition:
        raise RuntimeError(message)


def check_tray_window(root,app):
    """通过原生最小化事件和托盘菜单验证打包后的恢复路径。"""
    import time
    if app.tray is None:
        app.close()
        root.mainloop()
        return False
    from Foundation import NSObject,NSTimer,NSRunLoop,NSDefaultRunLoopMode
    from AppKit import NSEventTrackingRunLoopMode

    class VigilScreenBundleMenuProbe(NSObject):
        def selectWindow_(self,timer):
            self.tracking=NSRunLoop.currentRunLoop().currentMode()==NSEventTrackingRunLoopMode
            for _ in range(25):
                self.menu.performActionForItemAtIndex_(0)
            self.menu.cancelTracking()

        def openMenu_(self,timer):
            self.item.popUpStatusItemMenu_(self.menu)
            self.returned=True

    probe=VigilScreenBundleMenuProbe.alloc().init()
    probe.menu=app.monitor.menu_indicator.menu
    probe.item=app.monitor.menu_indicator.item
    errors = []
    stage = [0]
    completed = [0]
    started = time.monotonic()
    def tick():
        try:
            require(time.monotonic()-started<35,'托盘最小化/恢复等待超时')
            if stage[0]==0:
                root.iconify()
                stage[0]=1
            elif stage[0]==1 and app.tray.hidden and app.tray.settle_timer is None:
                require(app.tray.application.activationPolicy()==app.tray.accessory_policy,'隐藏后 Dock 策略不正确')
                require(app.monitor.menu_indicator.item.button().image() is not None,'托盘图标未显示')
                windows=app.tray.hidden_windows
                require(bool(windows),'未找到主窗口')
                require(app.tray.application.isHidden(),'应用尚未隐藏')
                probe.tracking=False;probe.returned=False
                probe.selection=NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(.2,probe,'selectWindow:',None,False)
                probe.opening=NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(.05,probe,'openMenu:',None,False)
                NSRunLoop.currentRunLoop().addTimer_forMode_(probe.selection,NSEventTrackingRunLoopMode)
                NSRunLoop.currentRunLoop().addTimer_forMode_(probe.opening,NSDefaultRunLoopMode)
                stage[0]=2
            elif stage[0]==2 and not app.tray.hidden and not app.tray.restoring:
                require(probe.tracking and probe.returned,'原生菜单跟踪循环未正常完成')
                probe.selection.invalidate();probe.opening.invalidate()
                require(root.state()=='normal','托盘未恢复窗口')
                require(root.winfo_ismapped() and app.notebook.winfo_viewable(),'主体界面未映射')
                windows=[window for window in app.tray.application.windows() if str(window.title())==root.title()]
                require(bool(windows) and all(window.isVisible() and not window.isMiniaturized() for window in windows),
                        '原生主窗口未显示')
                require(not app.tray.application.isHidden(),'应用仍处于隐藏状态')
                require(app.tray.application.activationPolicy()==app.tray.regular_policy,'恢复后 Dock 策略不正确')
                completed[0]+=1
                if completed[0]<4:
                    root.iconify()
                    stage[0]=1
                    root.after(75,tick)
                    return
                stage[0]=3
                app.close()
                return
            root.after(75,tick)
        except Exception:
            errors.append(traceback.format_exc())
            app.close()
    root.after(150,tick)
    root.mainloop()
    require(not errors,'\n'.join(errors))
    require(stage[0]==3,'托盘验证未完成')
    return True


def configure_runtime():
    root = data_root()
    (root/'logs').mkdir(parents=True,exist_ok=True)
    configure_process_environment()
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.cache/matplotlib'))
    os.environ.setdefault('MPLBACKEND','Agg')
    if '--identity-worker' in sys.argv:
        restore_worker_stream('stdout',1,root/'logs')
        restore_worker_stream('stderr',2,root/'logs')
    elif getattr(sys,'frozen',False) or sys.stdout is None or sys.stderr is None:
        sys.stdout = sys.stderr = (root/'logs/application.log').open('a',encoding='utf-8',buffering=1)


def self_test(report_path):
    import subprocess
    import tkinter as tk
    from unittest.mock import patch
    import numpy as np
    from PIL import Image
    import torch
    from chi_camera import FaceFinder,recognize_faces,save_hit
    from gaze import GazeEstimator
    from identity_training import available_identities,load_profile,read_profile
    from main import VigilScreen
    from model import ImageClassifier,build_transform
    from runtime_paths import worker_command
    root_path = data_root()
    finder = FaceFinder(RESOURCE_ROOT/'face_models/yunet.onnx')
    gaze = GazeEstimator(RESOURCE_ROOT/'face_models/face_landmarker.task')
    try:
        blank = Image.new('RGB',(320,240))
        picture = blank
        require(not finder.detect(blank),'空白图出现误检')
        # 验证训练时才触发的优化器、自动求导、图像变换等动态依赖。
        network = ImageClassifier(2)
        optimizer = torch.optim.Adam(network.parameters(),lr=.001)
        batch = torch.stack([build_transform(True)(blank) for _ in range(2)])
        loss = torch.nn.functional.cross_entropy(network(batch),torch.tensor([0,1]))
        loss.backward();optimizer.step()
        profile = root_path/'checkpoints/identities/active_profile.json'
        classes = available_identities(profile)
        recognized = []
        if classes:
            verifier,metadata = load_profile(profile,RESOURCE_ROOT/'face_models/sface_int8.onnx')
            require(set(verifier.gallery)==set(classes),'身份样本不完整')
            from labeling import image_files
            for image_path in image_files(root_path/'data/labeled'):
                with Image.open(image_path) as image:
                    picture = image.convert('RGB')
                faces = finder.detect(picture)
                if faces:
                    recognized = recognize_faces(picture,faces,verifier=verifier,target=classes[0],gaze=gaze)
                    break
        else:
            picture = blank
        screenshot = save_hit(picture,root_path/'diagnostics',face_boxes=[row[:4] for row in recognized if row[6]])
        root = tk.Tk()
        with patch('cv2.VideoCapture') as camera:
            app = VigilScreen(root)
            root.update_idletasks()
            app.monitor.draw_frame(picture,recognized)
            require(app.monitor.photo.width()>1,'视频显示失败')
            require(not app.monitor.want_running and not app.monitor.worker.is_alive(),'启动时意外开始监控')
            camera.assert_not_called()
            if app.monitor.menu_indicator is not None:
                app.monitor.menu_indicator.show_hit(None,str(screenshot))
                require('Peek 1' in str(app.monitor.menu_indicator.item.button().title()),'菜单栏提示失败')
            tray_verified = check_tray_window(root,app)
        training_report = None
        if classes:
            path,metadata = read_profile(profile)
            materials = path.parent/'materials'
            output = root_path/'diagnostics/training'
            command = worker_command('train','--data-dir',materials,'--output-dir',output,
                                     '--epochs','1','--batch-size','4','--device','cpu')
            completed = subprocess.run(command,cwd=root_path,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                       text=True,timeout=180,**worker_process_options())
            (root_path/'diagnostics/worker.log').write_text(completed.stdout,encoding='utf-8')
            require(completed.returncode==0,completed.stdout[-4000:])
            require((output/'active_profile.json').is_file(),'训练未生成身份模型')
            completed = subprocess.run(worker_command('validate','--profile',output/'active_profile.json',
                                        '--output',output/'revalidation.json'),cwd=root_path,
                                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120,**worker_process_options())
            require(completed.returncode==0,completed.stdout[-4000:])
            training_report = json.loads((output/'revalidation.json').read_text())
        report = {'ok':True,'frozen':bool(getattr(sys,'frozen',False)),
                  'architecture':platform.machine(),'platform':sys.platform,
                  'resources':str(RESOURCE_ROOT),'data_root':str(root_path),
                  'identity_classes':classes,'detected_sample_faces':len(recognized),
                  'native_face_and_gaze_models':True,'torch_gradient_and_optimizer':True,
                  'tk_window_and_menu_bar':True,'startup_camera_closed':True,'camera_accessed':False,
                  'minimize_to_tray_and_restore':tray_verified,
                  'native_menu_tracking_verified':tray_verified,
                  'tray_restore_cycles':4 if tray_verified else 0,
                  'show_requests_per_cycle':25 if tray_verified else 0,
                  'packaged_training_and_validation':training_report is not None,
                  'validation_images':training_report['images'] if training_report else 0,
                  'screenshot':str(screenshot)}
        Path(report_path).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    finally:
        gaze.close()


def main():
    import multiprocessing
    multiprocessing.freeze_support()
    configure_runtime()
    initialize_app_data()
    if len(sys.argv)>1 and sys.argv[1]=='--identity-worker':
        from identity_training import main as identity_main
        identity_main(sys.argv[2:])
        return
    if len(sys.argv)>1 and sys.argv[1]=='--self-test':
        self_test(sys.argv[2])
        return
    from main import main as gui_main
    gui_main()


if __name__=='__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        if '--identity-worker' not in sys.argv and '--self-test' not in sys.argv:
            from tkinter import messagebox
            messagebox.showerror('VigilScreen 启动失败',f'请查看日志：{data_root()/"logs/application.log"}\n\n{traceback.format_exc()[-1000:]}')
        raise SystemExit(1)
