"""在 Windows x64 构建 Intel / AMD 通用 CPU 版独立应用。"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import struct
import subprocess
import sys

from build_macos import prepare_licenses,prepare_seed
from runtime_support import native_architecture

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-training-data',action='store_true')
    options=parser.parse_args()
    if sys.platform!='win32' or native_architecture()!='x86_64' or struct.calcsize('P')!=8:
        raise SystemExit('请在 Windows 使用 64 位 x86_64 Python 构建，支持 Intel 与 AMD 处理器。')
    if not importlib.metadata.version('torch').endswith('+cpu'):
        raise SystemExit('体积优化构建需要 CPU 版 Torch，请先安装 requirements-windows.txt。')
    prepare_seed(include_training=options.include_training_data)
    prepare_licenses()
    environment=os.environ.copy()
    environment['PYINSTALLER_CONFIG_DIR']=str(ROOT/'build/pyinstaller-cache')
    environment['MPLCONFIGDIR']=str(ROOT/'build/matplotlib-cache')
    subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm',
                    '--distpath',str(ROOT/'dist'),'--workpath',str(ROOT/'build/pyinstaller'),
                    str(ROOT/'VigilScreen.spec')],cwd=ROOT,env=environment,check=True)
    print('构建完成：'+str(ROOT/'dist/VigilScreen/VigilScreen.exe'))
    print('分发整个 dist/VigilScreen 目录；不能只复制 .exe。')


if __name__=='__main__':
    main()
