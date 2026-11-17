"""独立应用入口，统一窗口、训练子进程和离线打包自检。"""
import json
import os
from pathlib import Path
import sys
import traceback

from runtime_paths import RESOURCE_ROOT,data_root,initialize_app_data


def require(condition,message='独立应用自检失败'):
    if not condition:
        raise RuntimeError(message)


def configure_runtime():
    root = data_root()
    (root/'logs').mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('MKL_NUM_THREADS','1')
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.cache/matplotlib'))
    os.environ.setdefault('MPLBACKEND','Agg')
    if '--identity-worker' in sys.argv:
        if sys.stdout is None:
            sys.stdout = os.fdopen(os.dup(1),'w',buffering=1)
        if sys.stderr is None:
            sys.stderr = os.fdopen(os.dup(2),'w',buffering=1)
    elif getattr(sys,'frozen',False):
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
            app.close()
        training_report = None
        if classes:
            path,metadata = read_profile(profile)
            materials = path.parent/'materials'
            output = root_path/'diagnostics/training'
            command = worker_command('train','--data-dir',materials,'--output-dir',output,
                                     '--epochs','1','--batch-size','4','--device','cpu')
            completed = subprocess.run(command,cwd=root_path,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
            (root_path/'diagnostics/worker.log').write_text(completed.stdout,encoding='utf-8')
            require(completed.returncode==0,completed.stdout[-4000:])
            require((output/'active_profile.json').is_file(),'训练未生成身份模型')
            completed = subprocess.run(worker_command('validate','--profile',output/'active_profile.json',
                                        '--output',output/'revalidation.json'),cwd=root_path,
                                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
            require(completed.returncode==0,completed.stdout[-4000:])
            training_report = json.loads((output/'revalidation.json').read_text())
        report = {'ok':True,'frozen':bool(getattr(sys,'frozen',False)),
                  'architecture':os.uname().machine,'resources':str(RESOURCE_ROOT),'data_root':str(root_path),
                  'identity_classes':classes,'detected_sample_faces':len(recognized),
                  'native_face_and_gaze_models':True,'torch_gradient_and_optimizer':True,
                  'tk_window_and_menu_bar':True,'startup_camera_closed':True,'camera_accessed':False,
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
