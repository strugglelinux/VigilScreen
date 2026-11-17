"""仅清除项目生成的数据，不删除运行所需的通用视觉模型。"""
from pathlib import Path
import shutil


DATA_DIRECTORIES = ('data','checkpoints','screenshots')


def reset_project_data(root):
    root = Path(root)
    for name in DATA_DIRECTORIES:
        path = root/name
        # 顶层软链接只删除链接本身，不能清空链接指向的外部目录。
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    for relative in ('data/labeled','checkpoints/identities','screenshots'):
        (root/relative).mkdir(parents=True,exist_ok=True)
