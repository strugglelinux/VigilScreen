"""手动分类的持久化记录与 ImageFolder 数据导出。"""

import math
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff'}


def image_files(directory: Path) -> list[Path]:
    return sorted(p.resolve() for p in directory.rglob('*')
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)


def existing_categories(directory: Path) -> list[str]:
    """发现已有类别，不要求图片曾经由本程序标记，也不创建或修改目录。"""
    def valid(name):
        return (isinstance(name,str) and bool(name.strip()) and not name.startswith('.')
                and name not in {'train','val','test'} and not any(c in name for c in '/\\:'))
    names=set()
    directory=directory.expanduser()
    try:
        names.update(child.name for child in directory.iterdir() if child.is_dir() and valid(child.name))
    except OSError:
        pass
    try:
        labels=json.loads((directory/'.labels.json').read_text(encoding='utf-8'))
        if isinstance(labels,dict):
            names.update(name for name in labels.values() if valid(name))
    except (OSError,ValueError):
        pass
    return sorted(names)

def selection_box(start, end, display_box, image_size):
    """把预览画布上的拖拽坐标映射为原图像素，支持反向拖拽与越界截断。"""
    x, y, width, height = display_box
    image_width, image_height = image_size
    if width <= 0 or height <= 0:
        raise ValueError('预览尺寸无效。')
    left, right = sorted((start[0], end[0]))
    top, bottom = sorted((start[1], end[1]))
    left = max(0, min(image_width, math.floor((left - x) * image_width / width)))
    right = max(0, min(image_width, math.ceil((right - x) * image_width / width)))
    top = max(0, min(image_height, math.floor((top - y) * image_height / height)))
    bottom = max(0, min(image_height, math.ceil((bottom - y) * image_height / height)))
    if right - left < 2 or bottom - top < 2:
        raise ValueError('框选区域过小，请重新拖动画框。')
    return left, top, right, bottom


def prepare_capture(picture: Image.Image, box=None, max_side=0) -> Image.Image:
    if max_side < 0:
        raise ValueError('最长边必须大于等于 0；0 表示不缩放。')
    if box is not None:
        left, top, right, bottom = box
        if not (0 <= left < right <= picture.width and 0 <= top < bottom <= picture.height):
            raise ValueError('框选区域超出图片范围。')
        picture = picture.crop(box)
    else:
        picture = picture.copy()
    picture = picture.convert('RGB')
    if max_side:
        picture.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return picture



class LabelStore:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.manifest = self.root / '.labels.json'
        self.labels = json.loads(self.manifest.read_text(encoding='utf-8')) if self.manifest.exists() else {}
        self.regions_path = self.root / '.regions.json'
        self.regions = json.loads(self.regions_path.read_text(encoding='utf-8')) if self.regions_path.exists() else {}

    def save(self, source: Path, category: str, box=None, max_side=0) -> Path:
        category = category.strip()
        if (not category or category in {'.', '..', 'train', 'val', 'test'}
                or any(c in category for c in '/\\:') or category.startswith('.')):
            raise ValueError('请输入有效类别名称，不能含路径分隔符，或使用 train、val、test。')
        source = source.resolve()
        # 文件名由原始路径生成，避免同名图片互相覆盖；原图始终保留。
        filename = hashlib.sha256(str(source).encode()).hexdigest() + '.png'
        target = self.root / category / filename
        if target.resolve() == source:
            raise ValueError('请从原始图片目录导入，标记输出目录应与原图目录分开。')
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as picture:
            picture = prepare_capture(ImageOps.exif_transpose(picture), box, max_side)
            temporary = target.with_suffix('.tmp')
            picture.save(temporary, format='PNG')
            temporary.replace(target)
        previous = self.labels.get(str(source))
        updated = dict(self.labels)
        updated[str(source)] = category
        temporary_manifest = self.manifest.with_suffix('.tmp')
        temporary_manifest.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary_manifest.replace(self.manifest)
        self.labels = updated
        self.regions[str(source)] = {'box': list(box) if box is not None else None, 'max_side': max_side}
        temporary_regions = self.regions_path.with_suffix('.tmp')
        temporary_regions.write_text(json.dumps(self.regions, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary_regions.replace(self.regions_path)
        if previous and previous != category:
            old = self.root / previous / filename
            if old.resolve().is_relative_to(self.root) and old.is_file():
                old.unlink()
                if not any(old.parent.iterdir()):
                    old.parent.rmdir()
        return target
