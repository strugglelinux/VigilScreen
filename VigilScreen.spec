# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files,collect_dynamic_libs,copy_metadata

project = Path(SPECPATH)
datas = [(str(project/'face_models'),'face_models'),(str(project/'build/seed'),'seed'),
         (str(project/'build/licenses'),'licenses')]
datas += collect_data_files('mediapipe',excludes=['**/test*','**/__pycache__/*'])
for package in ('mediapipe','torch','torchvision','numpy','pillow','pyobjc-core','pyobjc-framework-Cocoa'):
    datas += copy_metadata(package)
binaries = collect_dynamic_libs('mediapipe')+collect_dynamic_libs('torchvision',search_patterns=['*.dylib','*.so'])
a = Analysis([str(project/'vigil_app.py')],pathex=[str(project)],binaries=binaries,datas=datas,
    hiddenimports=['AppKit','Foundation','objc','mediapipe.tasks.c','unittest.mock'],
    excludes=['tensorflow','keras','jax','jaxlib','IPython','notebook','pandas','pytest','scipy',
              'sklearn','PyQt5','PyQt6','PySide2','PySide6',
              'h5py','rich','pygments','torch._dynamo.test_case',
              'tensorboard','torch.utils.tensorboard','onnx','onnxruntime'],
    hooksconfig={'matplotlib':{'backends':['Agg']}},noarchive=False,optimize=1)
# 移除未使用的编译开发文件及测试数据，保留运行时需要的 Torch Python 源码。
a.datas = [item for item in a.datas if not any(part in {'tests','test','include','cmake','__pycache__'}
                                            for part in Path(item[0]).parts)]
a.binaries = [item for item in a.binaries if not Path(item[0]).name.startswith('protoc')]
pyz = PYZ(a.pure)
exe = EXE(pyz,a.scripts,[],exclude_binaries=True,name='VigilScreen',debug=False,
          strip=True,upx=False,console=False,argv_emulation=False,target_arch='arm64',
          codesign_identity=None,entitlements_file=str(project/'macos_entitlements.plist'))
coll = COLLECT(exe,a.binaries,a.datas,strip=True,upx=False,name='VigilScreen')
app = BUNDLE(coll,name='VigilScreen.app',bundle_identifier='local.vigilscreen.desktop',
             info_plist={'CFBundleDisplayName':'VigilScreen','CFBundleShortVersionString':'1.1.1',
                         'CFBundleVersion':'1.1.1',
                         'NSCameraUsageDescription':'点击开始监控或摄像头标记时，采集图像以检查注视方向、识别身份和保存训练图片。',
                         'NSHighResolutionCapable':True,'LSMinimumSystemVersion':'15.0',
                         'NSPrincipalClass':'NSApplication'})
