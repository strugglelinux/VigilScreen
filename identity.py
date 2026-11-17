"""SFace 身份比对：不把所有人强行分入已知类别。"""
import hashlib
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from PIL import Image, ImageOps
from labeling import EXTENSIONS


def match_identity(feature, gallery, threshold=0.5, margin=0.06):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    if not np.isfinite(norm) or norm <= 0:
        return 'unknown', 0.0
    feature = feature / norm
    scores = []
    for category, templates in gallery.items():
        similarities = np.asarray(templates) @ feature
        # 至少用多张登记图片确认，减少偶然相似的单张图片引发的误认。
        count = min(3, len(similarities))
        score = float(np.sort(similarities)[-count:].mean())
        scores.append((score, category))
    scores.sort(reverse=True)
    if not scores:
        return 'unknown', 0.0
    best, category = scores[0]
    second = scores[1][0] if len(scores) > 1 else -1.0
    if best < threshold or best-second < margin:
        return 'unknown', best
    return category, best


def enrollment_paths(samples_dir, category):
    samples_dir = Path(samples_dir)
    manifest = samples_dir.parent / 'preparation.json'
    if samples_dir.name == 'train' and manifest.is_file():
        records = json.loads(manifest.read_text(encoding='utf-8'))['samples']
        # 直接用训练原图取得关键点，避免对紧裁剪的人脸再次检测造成丢失。
        paths = []
        for row in records:
            if row['split']!='train' or row['category']!=category:
                continue
            source,output = Path(row['source']),Path(row['output'])
            if not source.is_absolute():
                source = manifest.parent/source
            if not output.is_absolute():
                output = manifest.parent/output
            paths.append(source if source.is_file() else output)
        return paths
    return sorted(path for path in (samples_dir/category).rglob('*')
                  if path.is_file() and path.suffix.lower() in EXTENSIONS)


class IdentityVerifier:
    def __init__(self, model_path, samples_dir, finder, classes, threshold=0.5, margin=0.06):
        import cv2
        self.recognizer = cv2.FaceRecognizerSF.create(str(model_path), '')
        self.threshold, self.margin = threshold, margin
        self.gallery = {}
        self.enrollment_report = {'snapshot_id':uuid4().hex[:12], 'samples_dir':str(Path(samples_dir).resolve()), 'classes':{}}
        fingerprint = hashlib.sha256(Path(model_path).read_bytes())
        for category in classes:
            templates = []
            paths = enrollment_paths(samples_dir, category)
            audit = {'candidates':len(paths),'accepted':0,'skipped':[]}
            self.enrollment_report['classes'][category] = audit
            for path in paths:
                fingerprint.update(category.encode())
                fingerprint.update(str(path.resolve()).encode())
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    audit['skipped'].append({'source':str(path),'reason':'文件不存在或格式不支持'})
                    continue
                try:
                    fingerprint.update(path.read_bytes())
                    with Image.open(path) as picture:
                        picture = ImageOps.exif_transpose(picture).convert('RGB')
                    faces = finder.detect(picture)
                    if len(faces) != 1:
                        audit['skipped'].append({'source':str(path),'reason':f'唯一人脸未确认（{len(faces)} 张）'})
                        continue
                    templates.append(self.feature(picture, faces[0]))
                    audit['accepted'] += 1
                except (OSError,ValueError,RuntimeError,cv2.error) as error:
                    audit['skipped'].append({'source':str(path),'reason':str(error)})
            if len(templates) < 3:
                raise ValueError(f'类别 {category} 可用身份样本不足 3 张，请检查 --identity-samples 目录。')
            self.gallery[category] = np.stack(templates)
        self.enrollment_report['fingerprint'] = fingerprint.hexdigest()

    def save_report(self, path):
        path = Path(path)
        temporary = path.with_name(path.name+'.'+uuid4().hex[:8]+'.tmp')
        temporary.write_text(json.dumps(self.enrollment_report,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(path)

    def feature(self, picture, face):
        import cv2
        frame = cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)
        aligned = self.recognizer.alignCrop(frame, np.asarray(face, dtype=np.float32))
        feature = self.recognizer.feature(aligned).reshape(-1).copy()
        norm = np.linalg.norm(feature)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('无法提取有效身份特征，请更换清晰正脸样本。')
        return feature / norm

    def identify(self, picture, face):
        return match_identity(self.feature(picture, face), self.gallery, self.threshold, self.margin)
