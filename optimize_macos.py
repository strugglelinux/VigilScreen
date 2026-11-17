"""精简应用包里的本地库，再重新进行本机签名；不改动开发环境。"""
import json
from pathlib import Path
import subprocess
from macholib.MachO import MachO

ROOT = Path(__file__).resolve().parent
MAGIC = {b'\xcf\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xca\xfe\xba\xbe',b'\xbe\xba\xfe\xca'}


def regular_files(directory):
    return [path for path in Path(directory).rglob('*') if path.is_file() and not path.is_symlink()]


def native_files(directory):
    result = []
    for path in regular_files(directory):
        with path.open('rb') as stream:
            if stream.read(4) in MAGIC:
                result.append(path)
    return result


def optimize_bundle(bundle):
    bundle = Path(bundle)
    before = sum(path.stat().st_size for path in regular_files(bundle))
    optimized = 0
    for path in native_files(bundle/'Contents/Frameworks'):
        # 只精简共享库及扩展。主可执行文件含 PyInstaller 压缩存档，不能 strip。
        if all(header.header.filetype in {6,8} for header in MachO(str(path)).headers):
            subprocess.run(['/usr/bin/strip','-S','-x',str(path)],check=True,capture_output=True)
            subprocess.run(['/usr/bin/codesign','--force','--sign','-','--timestamp=none',str(path)],
                           check=True,capture_output=True)
            optimized += 1
    frameworks = sorted((bundle/'Contents/Frameworks').rglob('*.framework'),key=lambda p:len(p.parts),reverse=True)
    for framework in frameworks:
        if not framework.is_symlink():
            subprocess.run(['/usr/bin/codesign','--force','--sign','-','--timestamp=none',str(framework)],
                           check=True,capture_output=True)
    subprocess.run(['/usr/bin/codesign','--force','--sign','-','--timestamp=none',
                    '--entitlements',str(ROOT/'macos_entitlements.plist'),str(bundle)],check=True,capture_output=True)
    subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',str(bundle)],check=True,capture_output=True)
    after = sum(path.stat().st_size for path in regular_files(bundle))
    report = {'before_bytes':before,'after_bytes':after,'saved_bytes':before-after,'libraries_optimized':optimized}
    (ROOT/'build/size-optimization.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return report


if __name__=='__main__':
    optimize_bundle(ROOT/'dist/VigilScreen.app')
