"""使用目标架构的 Python 构建 macOS 应用，不修改源训练数据。"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
from runtime_support import native_architecture

ROOT = Path(__file__).resolve().parent


def prepare_seed(include_training=True):
    seed = ROOT/'build/seed'
    if seed.exists():
        shutil.rmtree(seed)
    seed.mkdir(parents=True)
    if not include_training:
        return
    for name in ('data','checkpoints'):
        source = ROOT/name
        if source.is_dir():
            shutil.copytree(source,seed/name,ignore=shutil.ignore_patterns('.DS_Store','__pycache__','*.tmp'))
    for path in seed.rglob('*.json'):
        encoded_root=json.dumps(str(ROOT),ensure_ascii=False)[1:-1]
        path.write_text(path.read_text(encoding='utf-8').replace(encoded_root,'__VIGILSCREEN_DATA__'),encoding='utf-8')


def prepare_licenses():
    destination = ROOT/'build/licenses'
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    versions = {}
    packages = ['torch','torchvision','numpy','pillow','mediapipe','opencv-contrib-python','opencv-python','pyobjc-core',
                'pyobjc-framework-Cocoa','sympy','mpmath','networkx','jinja2','markupsafe',
                'protobuf','flatbuffers','cffi','sounddevice','certifi','setuptools','filelock',
                'fsspec','typing_extensions','pycparser','packaging','matplotlib',
                'python-dateutil','six','kiwisolver','cycler','contourpy','fonttools','pyparsing',
                'jax','jaxlib','scipy','ml-dtypes','sentencepiece']
    for package in packages:
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        versions[package] = distribution.version
        for relative in distribution.files or []:
            if relative.suffix.lower() in {'.so','.dylib','.pyd','.py','.pyc','.h'}:
                continue
            if relative.name.lower().startswith(('license','copying','notice')):
                source = distribution.locate_file(relative)
                if source.is_file():
                    relative_parts = [part for part in relative.parts if part not in {'.','..'}]
                    target = destination/package/Path(*relative_parts)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,target)
    python_license = Path(sysconfig.get_path('stdlib'))/'LICENSE.txt'
    if python_license.is_file():
        shutil.copy2(python_license,destination/'Python-LICENSE.txt')
    (destination/'versions.json').write_text(json.dumps(versions,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch',choices=['arm64','x86_64'],default=native_architecture())
    parser.add_argument('--include-training-data',action='store_true',help='主动包含本地训练材料；默认构建干净应用')
    options=parser.parse_args()
    if sys.platform!='darwin' or native_architecture()!=options.arch:
        raise SystemExit('请在 macOS 上使用与 --arch 相同架构的 Python 和依赖构建。')
    if options.arch=='x86_64' and sys.version_info[:2]!=(3,11):
        raise SystemExit('Intel Mac 构建请使用 Python 3.11 x86_64 和 requirements-macos-intel.txt。')
    prepare_seed(include_training=options.include_training_data)
    prepare_licenses()
    environment = os.environ.copy()
    environment['PYINSTALLER_CONFIG_DIR'] = str(ROOT/'build/pyinstaller-cache')
    environment['MPLCONFIGDIR'] = str(ROOT/'build/matplotlib-cache')
    environment['VIGILSCREEN_BUILD_ARCH']=options.arch
    subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm',
                    '--distpath',str(ROOT/'dist'),'--workpath',str(ROOT/'build/pyinstaller'),
                    str(ROOT/'VigilScreen.spec')],cwd=ROOT,env=environment,check=True)
    # 只交付 .app，删除 PyInstaller 同时产生的重复命令行目录。
    shutil.rmtree(ROOT/'dist/VigilScreen')
    from optimize_macos import optimize_bundle
    optimize_bundle(ROOT/'dist/VigilScreen.app')
    print('构建完成：'+str(ROOT/'dist/VigilScreen.app'))


if __name__=='__main__':
    main()
