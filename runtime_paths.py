"""源码和独立应用共享路径规则；资源只读，用户数据保存在应用包之外。"""
import json
import os
from pathlib import Path
import shutil
import sys

RESOURCE_ROOT = Path(__file__).resolve().parent


def data_root():
    override = os.environ.get('VIGILSCREEN_DATA_DIR')
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys,'frozen',False):
        return Path.home()/'Library/Application Support/VigilScreen'
    return RESOURCE_ROOT


def worker_command(*arguments):
    if getattr(sys,'frozen',False):
        return [sys.executable,'--identity-worker',*map(str,arguments)]
    return [sys.executable,'-u',str(RESOURCE_ROOT/'identity_training.py'),*map(str,arguments)]


def initialize_app_data():
    root = data_root()
    root.mkdir(parents=True,exist_ok=True)
    if not getattr(sys,'frozen',False):
        return root
    marker = root/'.seed_initialized'
    if marker.exists():
        return root
    seed = RESOURCE_ROOT/'seed'
    if seed.is_dir():
        for source in seed.rglob('*'):
            if source.is_file():
                target = root/source.relative_to(seed)
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    if source.suffix=='.json':
                        target.write_text(source.read_text(encoding='utf-8').replace('__VIGILSCREEN_DATA__',str(root)),encoding='utf-8')
                    else:
                        shutil.copy2(source,target)
    # 保留标记，用户点击初始化后不能再次自动恢复内置身份和训练图片。
    marker.write_text(json.dumps({'initialized':True})+'\n',encoding='utf-8')
    return root
