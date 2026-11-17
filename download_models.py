"""下载通用视觉模型并核对 SHA-256，不涉及个人训练数据。"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import Request,urlopen

ROOT = Path(__file__).resolve().parent


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_models(directory,check_only=False):
    directory = Path(directory)
    manifest = json.loads((directory/'models.json').read_text(encoding='utf-8'))
    if manifest.get('format_version')!=1:
        raise ValueError('不支持的模型清单版本。')
    for model in manifest['models']:
        filename = model['filename']
        if Path(filename).name!=filename or filename in {'.','..'} or '/' in filename or '\\' in filename:
            raise ValueError('模型文件名无效。')
        target = directory/filename
        if target.is_file() and checksum(target)==model['sha256']:
            print(f'已校验：{filename}')
            continue
        if check_only:
            raise ValueError(f'模型缺失或校验失败：{filename}，请运行 python download_models.py')
        if not model['url'].startswith('https://'):
            raise ValueError('模型下载地址必须使用 HTTPS。')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory,prefix='.'+filename+'.',suffix='.tmp',delete=False) as stream:
                temporary = Path(stream.name)
                with urlopen(Request(model['url'],headers={'User-Agent':'VigilScreen/1.0'}),timeout=60) as response:
                    for block in iter(lambda:response.read(1024*1024),b''):
                        stream.write(block)
            if checksum(temporary)!=model['sha256']:
                raise ValueError(f'模型 SHA-256 不匹配：{filename}，未替换已有文件。')
            temporary.replace(target)
            print(f'已下载并校验：{filename}')
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='仅校验本地文件，不下载')
    args = parser.parse_args()
    try:
        prepare_models(ROOT/'face_models',check_only=args.check)
    except (OSError,ValueError,KeyError) as error:
        parser.exit(1,f'模型准备失败：{error}\n')


if __name__=='__main__':
    main()
