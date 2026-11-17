"""检查应用是否仍引用开发机上的非系统动态库。"""
import json
from pathlib import Path
from macholib.MachO import MachO
from macholib import mach_o
from optimize_macos import native_files


def audit_bundle(bundle):
    files = native_files(bundle)
    external = []
    external_rpaths = []
    versions = []
    architectures = set()
    dependency_commands = {mach_o.LC_LOAD_DYLIB,mach_o.LC_LOAD_WEAK_DYLIB,mach_o.LC_REEXPORT_DYLIB}
    for path in files:
        for header in MachO(str(path)).headers:
            architectures.add(header.header.cputype)
            for command,body,data in header.commands:
                if command.cmd in dependency_commands:
                    dependency = data.split(b'\0',1)[0].decode('utf-8')
                    if dependency.startswith('/') and not dependency.startswith(('/System/Library/','/usr/lib/')):
                        external.append({'file':str(path),'dependency':dependency})
                elif command.cmd==mach_o.LC_RPATH:
                    search_path = data.split(b'\0',1)[0].decode('utf-8')
                    if search_path.startswith('/') and not search_path.startswith(('/System/Library/','/usr/lib/')):
                        external_rpaths.append({'file':str(path),'search_path':search_path})
                if command.cmd==mach_o.LC_BUILD_VERSION:
                    versions.append(int(body.minos))
                elif command.cmd==mach_o.LC_VERSION_MIN_MACOSX:
                    versions.append(int(body.version))
    maximum = max(versions,default=0)
    result = {'native_binaries':len(files),'external_non_system_dependencies':external,
              'external_non_system_rpaths':external_rpaths,
              'architectures':sorted(architectures),
              'maximum_macos_minimum':f'{maximum>>16}.{(maximum>>8)&255}.{maximum&255}'}
    if external or external_rpaths:
        raise RuntimeError(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    result=audit_bundle(root/'dist/VigilScreen.app')
    (root/'build/native-dependency-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
