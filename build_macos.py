"""构建当前 Apple Silicon 机器可独立运行的应用，不修改源训练数据。"""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent


def prepare_seed():
    seed = ROOT/'build/seed'
    if seed.exists():
        shutil.rmtree(seed)
    seed.mkdir(parents=True)
    for name in ('data','checkpoints'):
        source = ROOT/name
        if source.is_dir():
            shutil.copytree(source,seed/name,ignore=shutil.ignore_patterns('.DS_Store','__pycache__','*.tmp'))
    for path in seed.rglob('*.json'):
        path.write_text(path.read_text(encoding='utf-8').replace(str(ROOT),'__VIGILSCREEN_DATA__'),encoding='utf-8')


def prepare_licenses():
    destination = ROOT/'build/licenses'
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    versions = {}
    packages = ['torch','torchvision','numpy','pillow','mediapipe','opencv-python','pyobjc-core',
                'pyobjc-framework-Cocoa','sympy','mpmath','networkx','jinja2','markupsafe',
                'protobuf','flatbuffers','cffi','sounddevice','certifi','setuptools','filelock',
                'fsspec','typing_extensions','pycparser','packaging','matplotlib',
                'python-dateutil','six','kiwisolver','cycler','contourpy','fonttools','pyparsing']
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
    if sys.platform!='darwin' or platform.machine()!='arm64':
        raise SystemExit('请使用 Apple Silicon macOS 与 arm64 Python 构建。')
    prepare_seed()
    prepare_licenses()
    environment = os.environ.copy()
    environment['PYINSTALLER_CONFIG_DIR'] = str(ROOT/'build/pyinstaller-cache')
    environment['MPLCONFIGDIR'] = str(ROOT/'build/matplotlib-cache')
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
