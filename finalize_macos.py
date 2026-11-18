"""验证签名并生成压缩安装映像及构建报告。先执行已打包应用的自检。"""
import hashlib
import json
import plistlib
from pathlib import Path
import shutil
import subprocess

from audit_macos import audit_bundle
from optimize_macos import regular_files
from build_macos import prepare_licenses

ROOT = Path(__file__).resolve().parent


def main():
    app = ROOT/'dist/VigilScreen.app'
    self_test = json.loads((ROOT/'build/standalone-self-test.json').read_text())
    if not self_test.get('ok') or not self_test.get('frozen') or not self_test.get('minimize_to_tray_and_restore'):
        raise RuntimeError('请先完成独立应用自检（包括托盘最小化与恢复）。')
    audit=audit_bundle(app)
    names={16777223:'x86_64',16777228:'arm64'}
    if len(audit['architectures'])!=1 or audit['architectures'][0] not in names:
        raise RuntimeError('发布包必须只包含同一受支持架构的本地库。')
    architecture=names[audit['architectures'][0]]
    if self_test.get('architecture')!=architecture:
        raise RuntimeError('独立自检报告与当前应用架构不一致，请重新自检。')
    plist_path=app/'Contents/Info.plist'
    plist=plistlib.loads(plist_path.read_bytes())
    def version(value):
        return tuple(map(int,value.split('.')))
    minimum=max((plist['LSMinimumSystemVersion'],audit['maximum_macos_minimum']),key=version)
    plist['LSMinimumSystemVersion']=minimum
    plist_path.write_bytes(plistlib.dumps(plist))
    prepare_licenses()
    shutil.copytree(ROOT/'build/licenses',app/'Contents/Resources/licenses',dirs_exist_ok=True)
    subprocess.run(['/usr/bin/codesign','--force','--sign','-','--timestamp=none',
                    '--entitlements',str(ROOT/'macos_entitlements.plist'),str(app)],check=True)
    subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',str(app)],check=True)
    audit = audit_bundle(app)
    stage = ROOT/'build/dmg-stage'
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    subprocess.run(['/bin/cp','-cR',str(app),str(stage/'VigilScreen.app')],check=True)
    (stage/'Applications').symlink_to('/Applications',target_is_directory=True)
    shutil.copy2(ROOT/'DISTRIBUTION.md',stage/'使用说明.md')
    archive = ROOT/f'dist/VigilScreen-macOS-{architecture}.dmg'
    subprocess.run(['/usr/bin/hdiutil','create','-volname','VigilScreen','-srcfolder',str(stage),
                    '-format','ULMO','-ov',str(archive)],check=True)
    subprocess.run(['/usr/bin/hdiutil','verify',str(archive)],check=True)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    size = sum(path.stat().st_size for path in regular_files(app))
    revision = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    report = {'application':app.name,'archive':archive.name,'version':plist['CFBundleShortVersionString'],
              'source_commit':revision.stdout.strip() if revision.returncode==0 else None,
              'architecture':architecture,'minimum_macos':minimum,
              'application_bytes':size,'archive_bytes':archive.stat().st_size,
              'archive_sha256':digest.hexdigest(),'compression':'ULMO (LZMA)',
              'bundle_signature_verified':True,
              'signing':'ad-hoc; not Apple-notarized','archive_integrity_verified':True,
              'size_optimization':json.loads((ROOT/'build/size-optimization.json').read_text()),
              'standalone_relocated_self_test':self_test,'native_dependency_audit':audit}
    (ROOT/'dist/build-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'application_bytes':size,'archive_bytes':archive.stat().st_size,'sha256':digest.hexdigest()}))


if __name__=='__main__':
    main()
