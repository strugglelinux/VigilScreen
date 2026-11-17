"""可独立使用的身份训练、特征登记与留出验证流程。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import shutil
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps

from identity import IdentityVerifier
from labeling import EXTENSIONS,image_files

from runtime_paths import RESOURCE_ROOT
ROOT = RESOURCE_ROOT


def read_profile(path):
    path = Path(path).expanduser().resolve()
    metadata = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(metadata,dict):
        raise ValueError('身份模型入口必须为 JSON 对象。')
    if 'profile' in metadata:
        if not isinstance(metadata['profile'],str) or not metadata['profile']:
            raise ValueError('身份模型入口路径无效。')
        path = (path.parent/metadata['profile']).resolve()
        metadata = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(metadata,dict):
        raise ValueError('身份模型内容无效。')
    classes = metadata.get('classes')
    if (metadata.get('format_version')!=1 or not isinstance(classes,list) or not classes
            or not all(isinstance(name,str) and name for name in classes) or len(set(classes))!=len(classes)
            or any(name in {'unknown','人脸','全部人脸（不指定身份）'} for name in classes)):
        raise ValueError('不是有效的 VigilScreen 身份模型。')
    return path,metadata


def available_identities(path):
    try:
        return read_profile(path)[1]['classes']
    except (OSError,ValueError,KeyError,TypeError):
        return []


def heldout_records(profile_path,metadata):
    dataset = Path(profile_path).parent/metadata['dataset']
    prepared = json.loads((dataset/'preparation.json').read_text(encoding='utf-8'))
    records = []
    for row in prepared['samples']:
        if row['split']=='val':
            record = dict(row)
            source = Path(row['source'])
            record['source'] = str(source if source.is_absolute() else (dataset/source).resolve())
            records.append(record)
    return records


def load_profile(path, model_path, threshold=None, margin=None):
    import cv2
    path,metadata = read_profile(path)
    if metadata['embedding_model_sha256'] != hashlib.sha256(Path(model_path).read_bytes()).hexdigest():
        raise ValueError('身份特征模型与登记版本不一致，请重新训练身份模型。')
    verifier = IdentityVerifier.__new__(IdentityVerifier)
    verifier.recognizer = cv2.FaceRecognizerSF.create(str(model_path),'')
    verifier.threshold = metadata['threshold'] if threshold is None else threshold
    verifier.margin = metadata['margin'] if margin is None else margin
    verifier.gallery = {}
    with np.load(path.parent/metadata['gallery'],allow_pickle=False) as arrays:
        for index,category in enumerate(metadata['classes']):
            templates = arrays[f'class_{index}'].copy()
            if templates.ndim!=2 or len(templates)<3 or not np.isfinite(templates).all():
                raise ValueError(f'{category} 的身份特征无效。')
            verifier.gallery[category] = templates
    verifier.enrollment_report = metadata['enrollment']
    return verifier,metadata


def evaluate(verifier, records, finder, training_hashes=(),progress=None):
    """身份验证不推断注视真值；验证失败与无法检测均计入分母。"""
    import cv2
    training_hashes = set(training_hashes)
    details = []
    counts = {}
    for index,record in enumerate(records,1):
        source,category = Path(record['source']),record['category']
        expected = category if category in verifier.gallery else 'unknown'
        row = {'source':str(source),'expected':expected,'provided_label':category,
               'predicted':'unknown','similarity':0.0,'faces':0,'training_content_overlap':False}
        try:
            row['training_content_overlap'] = hashlib.sha256(source.read_bytes()).hexdigest() in training_hashes
            with Image.open(source) as picture:
                picture = ImageOps.exif_transpose(picture).convert('RGB')
            faces = finder.detect(picture,scan_grid=2)
            row['faces'] = len(faces)
            if len(faces)==1:
                row['predicted'],row['similarity'] = verifier.identify(picture,faces[0])
            else:
                row['error'] = f'需要唯一人脸，实际检测到 {len(faces)} 张'
        except (OSError,ValueError,RuntimeError,cv2.error) as error:
            row['error'] = str(error)
        row['correct'] = row['faces']==1 and 'error' not in row and row['predicted']==expected
        stat = counts.setdefault(category,{'images':0,'correct':0,'unknown':0,'wrong_identity':0,'unusable':0})
        stat['images'] += 1
        stat['correct'] += int(row['correct'])
        stat['unknown'] += int(row['predicted']=='unknown')
        stat['wrong_identity'] += int(row['predicted'] not in ('unknown',expected))
        stat['unusable'] += int('error' in row)
        details.append(row)
        if progress is not None:
            progress(index,len(records),source)
    usable_unknown = [row for row in details if row['expected']=='unknown' and 'error' not in row]
    known = [row for row in details if row['expected']!='unknown']
    return {'kind':'identity_only','images':len(details),'per_class':counts,
            'unusable_images':sum('error' in row for row in details),
            'usable_unknown_images':len(usable_unknown),
            'known_identity_match_rate':sum(row['correct'] for row in known)/len(known) if known else None,
            'unknown_false_accept_rate':sum(row['predicted']!='unknown' for row in usable_unknown)/len(usable_unknown) if usable_unknown else None,
            'training_content_overlaps':sum(row['training_content_overlap'] for row in details),
            'limitations':'仅验证身份匹配，不包含视线准确率。单类别 CNN 的准确率不能评价身份排除能力；无陌生人验证图时不报告误接受率。内容重复时不属于独立验证。',
            'details':details}


def describe_validation(report,profile_path,metadata,verifier):
    report.update(profile_id=metadata['profile_id'],profile_path=str(Path(profile_path).resolve()),
                  classes=list(metadata['classes']),threshold=verifier.threshold,margin=verifier.margin)
    return report


def train_profile(args):
    from chi_camera import FaceFinder
    from prepare_face_training import prepare
    from test import train
    source = args.data_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output==source or output.is_relative_to(source):
        raise ValueError('模型输出目录不能放在训练图片目录内部。')
    if args.dataset!='folder':
        raise ValueError('身份训练请使用按身份分文件夹的 folder 数据集。')
    run = output/'runs'/uuid4().hex[:12]
    dataset = run/'dataset'
    direct_images = [path for path in image_files(source) if path.parent==source]
    materials = run/'materials'
    if direct_images:
        name = args.person_name.strip()
        if not name or name in {'.','..','train','val','test','unknown','人脸'} or any(char in name for char in '/\\:') or name.startswith('.'):
            raise ValueError('平铺图片目录请填写有效的单人材料名称，例如 alice。')
        if any(path.parent!=source for path in image_files(source)):
            raise ValueError('训练目录不能混放平铺图片和按身份分组的图片。')
        destination = materials/name
        destination.mkdir(parents=True)
        for path in direct_images:
            shutil.copy2(path,destination/path.name)
    else:
        # 固定本次训练材料，避免外部目录在训练中被修改后污染登记或验证。
        materials.mkdir(parents=True,exist_ok=True)
        for path in image_files(source):
            target = materials/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    print('准备人脸数据：先划分原图，再裁剪；验证图不参与身份登记。',flush=True)
    prepared = prepare(materials,dataset,args.face_model,seed=args.seed,val_ratio=args.val_ratio,include_all=True)
    if any(name in {'unknown','人脸','全部人脸（不指定身份）'} for name in prepared['classes']):
        raise ValueError('unknown、人脸及全员选项是保留名称，请使用实际身份名称作为训练类别。')
    finder = FaceFinder(args.face_model)
    verifier = IdentityVerifier(args.identity_model,dataset/'train',finder,prepared['classes'])
    original_data = args.data_dir
    args.data_dir = dataset
    train(args)
    args.data_dir = original_data
    training_hashes = sorted({hashlib.sha256(Path(row['source']).read_bytes()).hexdigest()
                             for row in prepared['samples'] if row['split']=='train'})
    validation = evaluate(verifier,[row for row in prepared['samples'] if row['split']=='val'],finder,training_hashes)
    describe_validation(validation,run/'profile.json',{'profile_id':run.name,'classes':prepared['classes']},verifier)
    (run/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(run/'gallery.npz',**{f'class_{index}':verifier.gallery[category]
                                         for index,category in enumerate(prepared['classes'])})
    profile = {'format_version':1,'profile_id':run.name,'classes':prepared['classes'],
               'source_directory':str(source),'dataset':'dataset','gallery':'gallery.npz',
               'embedding_model_sha256':hashlib.sha256(args.identity_model.read_bytes()).hexdigest(),
               'threshold':.5,'margin':.06,'enrollment':verifier.enrollment_report,
               'training_hashes':training_hashes,'validation':'validation.json'}
    (run/'profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8')
    # 当前批次的材料和裁剪均在 run 内，移动整个项目后仍可重新验证。
    for group in ('samples','fallbacks','skipped'):
        for row in prepared.get(group,[]):
            for key in ('source','output'):
                if key in row and Path(row[key]).is_absolute():
                    row[key] = os.path.relpath(row[key],dataset)
    (dataset/'preparation.json').write_text(json.dumps(prepared,ensure_ascii=False,indent=2),encoding='utf-8')
    # 只有完整训练、登记与验证成功后才替换入口，失败或停止保留原身份模型。
    temporary = output/f'.active-{run.name}.tmp'
    temporary.write_text(json.dumps({'profile':str((run/'profile.json').relative_to(output))}),encoding='utf-8')
    temporary.replace(output/'active_profile.json')
    print('身份登记完成：'+str({name:len(rows) for name,rows in verifier.gallery.items()}),flush=True)
    print('身份留出验证：'+json.dumps(validation['per_class'],ensure_ascii=False),flush=True)
    print(f'当前身份模型：{output/"active_profile.json"}',flush=True)
    print(f'验证报告：{run/"validation.json"}',flush=True)


def validation_records(directory, expected=None):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f'验证目录不存在：{directory}')
    direct = [path.resolve() for path in sorted(directory.iterdir())
              if path.is_file() and path.suffix.lower() in EXTENSIONS]
    grouped = [{'source':str(path),'category':child.name} for child in sorted(directory.iterdir())
               if child.is_dir() and not child.name.startswith('.') for path in image_files(child)]
    if direct and grouped:
        raise ValueError('验证目录混有平铺图片和身份子目录，请选择其中一种格式，避免漏掉图片。')
    if direct:
        expected = expected.strip() if expected else ''
        if not expected:
            raise ValueError('直接放图片的验证目录需要指定预期身份；也可按身份分子目录。')
        return [{'source':str(path),'category':expected} for path in direct]
    return grouped


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0]=='train':
        from test import build_parser
        parser = build_parser()
        training_parser = next(action for action in parser._actions if isinstance(action,argparse._SubParsersAction)).choices['train']
        training_parser.set_defaults(dataset='folder')
        training_parser.add_argument('--face-model',type=Path,default=ROOT/'face_models/yunet.onnx')
        training_parser.add_argument('--identity-model',type=Path,default=ROOT/'face_models/sface_int8.onnx')
        training_parser.add_argument('--person-name',default='')
        args = parser.parse_args(argv)
        train_profile(args)
        return
    parser = argparse.ArgumentParser(description='验证身份模型（不验证注视准确率）')
    parser.add_argument('command',choices=['validate'])
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--images-dir',type=Path,default=None)
    parser.add_argument('--expected',default=None)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    from chi_camera import FaceFinder
    profile_path,metadata = read_profile(args.profile)
    verifier,metadata = load_profile(profile_path,ROOT/'face_models/sface_int8.onnx')
    if args.output.resolve() in {args.profile.resolve(),profile_path}:
        raise ValueError('验证报告不能覆盖身份模型入口或模型配置。')
    if args.expected is not None:
        args.expected = args.expected.strip()
        if args.expected not in [*metadata['classes'],'unknown']:
            raise ValueError(f'预期身份无效：{args.expected!r}，请选择 {metadata["classes"]} 或 unknown。')
    if args.images_dir is not None:
        records = validation_records(args.images_dir,args.expected)
    else:
        records = heldout_records(profile_path,metadata)
    if not records:
        raise ValueError('验证目录中没有图片。')
    report = evaluate(verifier,records,FaceFinder(ROOT/'face_models/yunet.onnx'),metadata['training_hashes'],
                      progress=lambda index,total,path:print(f'Validation {index}/{total} · {path.name}',flush=True))
    describe_validation(report,profile_path,metadata,verifier)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.output.with_name(args.output.name+f'.{uuid4().hex[:8]}.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(args.output)
    print(json.dumps({key:value for key,value in report.items() if key!='details'},ensure_ascii=False,indent=2))
    print(f'验证报告：{args.output}',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print(f'失败：{error}',file=sys.stderr,flush=True)
        raise SystemExit(1)
