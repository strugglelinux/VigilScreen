"""按原始训练/验证划分导出与摄像头相同的人脸裁剪，原始图片不变。"""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Subset
from chi_camera import FaceFinder, ROOT
from runtime_paths import RESOURCE_ROOT
from data import build_datasets


def samples(dataset):
    if isinstance(dataset, Subset):
        return [dataset.dataset.samples[i] for i in dataset.indices]
    return dataset.samples


def prepare(source, output, face_model, seed=42, val_ratio=0.2, include_all=False):
    if output.exists():
        raise ValueError('输出目录已存在，请使用新的目录，避免旧图片混入。')
    bundle = build_datasets('folder', source, seed=seed, val_ratio=val_ratio)
    finder = FaceFinder(face_model)
    report = {'classes': bundle.classes, 'seed': seed, 'val_ratio': val_ratio,
              'samples': [], 'skipped': [], 'counts': {}, 'include_all': include_all, 'fallbacks': []}
    for split, dataset in [('train', bundle.train), ('val', bundle.val), ('test', bundle.test)]:
        if dataset is None:
            continue
        counts = {name: 0 for name in bundle.classes}
        for path, label in samples(dataset):
            with Image.open(path) as picture:
                picture = ImageOps.exif_transpose(picture).convert('RGB')
            if include_all:
                faces = finder.detect(picture)
                faces = [tuple(int(value) for value in face[:4]) for face in faces]
            else:
                faces = finder.find(picture)
            method = 'face_crop'
            if len(faces) != 1:
                record = {'source': path, 'split': split, 'faces': len(faces)}
                if not include_all:
                    report['skipped'].append(record)
                    continue
                # 来源已由用户分类/框选：检测不确定时保留该人工标记图，不丢弃。
                report['fallbacks'].append(record)
                method = 'manual_crop'
                x, y, width, height = 0, 0, picture.width, picture.height
            else:
                x, y, width, height = faces[0]
            category = bundle.classes[label]
            filename = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest() + '.png'
            target = output / split / category / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            picture.crop((x, y, x+width, y+height)).save(target)
            counts[category] += 1
            report['samples'].append({'source': path, 'split': split, 'category': category,
                                      'box': [x, y, x+width, y+height], 'output': str(target), 'method': method})
        report['counts'][split] = counts
        if not all(counts.values()):
            raise ValueError(f'{split} 某类别没有可用正脸图片：{counts}；请补充样本后使用新目录重试。')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'preparation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/labeled')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--face-model', type=Path, default=RESOURCE_ROOT / 'face_models/yunet.onnx')
    parser.add_argument('--include-all', action='store_true', help='检测不确定时使用人工标记图，保留全部有效标记图片')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    args = parser.parse_args()
    report = prepare(args.source.resolve(), args.output.resolve(), args.face_model.resolve(), args.seed, args.val_ratio, args.include_all)
    print(json.dumps({'counts': report['counts'], 'skipped': len(report['skipped'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
