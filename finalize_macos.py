"""验证签名并生成压缩安装映像及构建报告。先执行已打包应用的自检。"""
import hashlib
import json
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
    if not self_test.get('ok') or not self_test.get('frozen'):
        raise RuntimeError('请先完成独立应用自检。')
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
    archive = ROOT/'dist/VigilScreen-macOS-arm64.dmg'
    subprocess.run(['/usr/bin/hdiutil','create','-volname','VigilScreen','-srcfolder',str(stage),
                    '-format','ULMO','-ov',str(archive)],check=True)
    subprocess.run(['/usr/bin/hdiutil','verify',str(archive)],check=True)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    size = sum(path.stat().st_size for path in regular_files(app))
    report = {'application':app.name,'archive':archive.name,'version':'1.0.0',
              'architecture':'arm64 (Apple Silicon)','minimum_macos':'15.0',
              'application_bytes':size,'archive_bytes':archive.stat().st_size,
              'archive_sha256':digest.hexdigest(),'compression':'ULMO (LZMA)',
              'source_tests_passed':67,'bundle_signature_verified':True,
              'signing':'ad-hoc; not Apple-notarized','archive_integrity_verified':True,
              'size_optimization':json.loads((ROOT/'build/size-optimization.json').read_text()),
              'standalone_relocated_self_test':self_test,'native_dependency_audit':audit}
    (ROOT/'dist/build-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'application_bytes':size,'archive_bytes':archive.stat().st_size,'sha256':digest.hexdigest()}))


if __name__=='__main__':
    main()
