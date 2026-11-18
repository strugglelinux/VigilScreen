"""检查本机依赖和通用视觉模型，不打开摄像头或修改训练数据。"""
import argparse
import importlib.metadata
import json
import os
import platform
import struct
import sys

from runtime_support import configure_process_environment,native_architecture


def check(models=False):
    configure_process_environment()
    from runtime_paths import data_root
    cache=data_root()/'.cache/matplotlib'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    os.environ.setdefault('MPLBACKEND','Agg')
    if struct.calcsize('P')!=8:
        raise RuntimeError('VigilScreen 需要 64 位 Python。')
    if sys.platform=='darwin' and native_architecture()=='x86_64' and sys.version_info[:2]!=(3,11):
        raise RuntimeError('Intel Mac 请使用 Python 3.11 x86_64 并安装 requirements-macos-intel.txt。')
    import torch
    import torchvision
    import cv2
    import mediapipe
    from test import select_device
    report={'platform':sys.platform,'architecture':native_architecture(),
            'python':platform.python_version(),'device':str(select_device('auto')),
            'torch':torch.__version__,'torchvision':torchvision.__version__,
            'opencv':cv2.__version__,'mediapipe':importlib.metadata.version('mediapipe'),
            'cpu_threads':torch.get_num_threads(),'camera_accessed':False}
    if models:
        import numpy as np
        from PIL import Image
        from download_models import prepare_models
        from runtime_paths import RESOURCE_ROOT
        from chi_camera import FaceFinder
        from gaze import GazeEstimator
        prepare_models(RESOURCE_ROOT/'face_models',check_only=True)
        detector=FaceFinder(RESOURCE_ROOT/'face_models/yunet.onnx')
        if detector.detect(Image.new('RGB',(320,240))):
            raise RuntimeError('空白图片出现人脸误检。')
        recognizer=cv2.FaceRecognizerSF.create(str(RESOURCE_ROOT/'face_models/sface_int8.onnx'),'')
        feature=recognizer.feature(np.zeros((112,112,3),dtype=np.uint8))
        if feature.size!=128 or not np.isfinite(feature).all():
            raise RuntimeError('身份模型特征输出无效。')
        gaze=GazeEstimator(RESOURCE_ROOT/'face_models/face_landmarker.task')
        try:
            face=np.array([40,30,160,160,80,80,160,80,120,110,90,150,150,150,.99])
            if gaze.measure(Image.new('RGB',(320,240)),face).looking:
                raise RuntimeError('空白图片出现视线误命中。')
        finally:
            gaze.close()
        report['native_models_verified']=True
    report['ok']=True
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',action='store_true',help='加载并验证真实检测、身份和视线模型')
    args=parser.parse_args()
    print(json.dumps(check(args.models),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
