"""Intel、AMD 与 Apple Silicon 共用的 CPU 预算和子进程设置。"""
import os
import platform
import subprocess
import sys


def native_architecture(machine=None):
    name=(machine or platform.machine()).lower()
    return {'amd64':'x86_64','x64':'x86_64','aarch64':'arm64'}.get(name,name)


def cpu_thread_budget():
    value=os.environ.get('VIGILSCREEN_CPU_THREADS') or os.environ.get('OMP_NUM_THREADS')
    if value:
        try:
            count=int(value)
        except ValueError:
            raise ValueError('VIGILSCREEN_CPU_THREADS / OMP_NUM_THREADS 必须是正整数。') from None
        if count<1:
            raise ValueError('CPU 线程数必须大于 0。')
        return min(count,32)
    count=getattr(os,'process_cpu_count',os.cpu_count)() or 1
    return min(4,max(1,count-1))


def configure_process_environment():
    threads=str(cpu_thread_budget())
    os.environ.setdefault('OMP_NUM_THREADS',threads)
    os.environ.setdefault('MKL_NUM_THREADS',threads)


def configure_opencv(cv2):
    # 实时检测给主界面和采集线程留出 CPU，不依赖 CUDA、Metal 或处理器品牌。
    cv2.setNumThreads(min(2,cpu_thread_budget()))
    cv2.setUseOptimized(True)


def available_training_devices(torch):
    devices=['auto','cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    mps=getattr(torch.backends,'mps',None)
    if mps is not None and mps.is_available():
        devices.append('mps')
    return devices


def worker_process_options():
    environment=os.environ.copy()
    environment['PYTHONIOENCODING']='utf-8'
    environment['PYTHONUTF8']='1'
    options={'encoding':'utf-8','errors':'replace','env':environment}
    if sys.platform=='win32':
        options['creationflags']=subprocess.CREATE_NO_WINDOW
    return options


def windows_worker_stream(standard_handle):
    """恢复 PyInstaller windowed EXE 从父进程继承的日志管道。"""
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetStdHandle.argtypes=[wintypes.DWORD]
    kernel.GetStdHandle.restype=wintypes.HANDLE
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.DuplicateHandle.argtypes=[wintypes.HANDLE,wintypes.HANDLE,wintypes.HANDLE,
                                    ctypes.POINTER(wintypes.HANDLE),wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.GetStdHandle(standard_handle)
    if handle in (None,0,ctypes.c_void_p(-1).value):
        raise OSError('训练子进程没有继承日志管道。')
    process=kernel.GetCurrentProcess()
    duplicate=wintypes.HANDLE()
    if not kernel.DuplicateHandle(process,handle,process,ctypes.byref(duplicate),0,False,2):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd=msvcrt.open_osfhandle(duplicate.value,os.O_WRONLY|os.O_BINARY)
    except OSError:
        kernel.CloseHandle(duplicate)
        raise
    return os.fdopen(fd,'w',encoding='utf-8',errors='replace',buffering=1)


def restore_worker_stream(name,descriptor,logs):
    stream=getattr(sys,name)
    if stream is not None:
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='replace')
        return
    try:
        stream=(windows_worker_stream(-11 if name=='stdout' else -12) if sys.platform=='win32'
                else os.fdopen(os.dup(descriptor),'w',encoding='utf-8',errors='replace',buffering=1))
    except OSError:
        stream=(logs/'identity-worker.log').open('a',encoding='utf-8',buffering=1)
    setattr(sys,name,stream)
