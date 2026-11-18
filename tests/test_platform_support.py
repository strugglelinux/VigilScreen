"""不依赖目标机，检查架构、CPU 回退、Windows 管道和跨平台配置。"""
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

import runtime_support as runtime
from test import select_device


class PlatformSupportTests(unittest.TestCase):
    def test_amd_and_intel_use_same_x64_architecture(self):
        for name in ('AMD64','x86_64','x64'):
            self.assertEqual(runtime.native_architecture(name),'x86_64')
        self.assertEqual(runtime.native_architecture('aarch64'),'arm64')

    def test_cpu_thread_budget_leaves_capacity_for_ui_and_respects_override(self):
        with patch.dict(os.environ,{},clear=True),patch.object(runtime.os,'process_cpu_count',return_value=8,create=True):
            self.assertEqual(runtime.cpu_thread_budget(),4)
            os.environ['VIGILSCREEN_CPU_THREADS']='2'
            runtime.configure_process_environment()
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'2')
            self.assertEqual(os.environ['MKL_NUM_THREADS'],'2')
            os.environ['VIGILSCREEN_CPU_THREADS']='0'
            with self.assertRaises(ValueError):
                runtime.cpu_thread_budget()

    def test_auto_device_on_intel_mac_and_windows_is_cpu_without_cuda(self):
        with patch('test.torch.cuda.is_available',return_value=False),\
             patch('test.native_architecture',return_value='x86_64'),\
             patch('test.torch.backends.mps.is_available',return_value=True):
            for system in ('win32','darwin'):
                with patch('test.sys.platform',system):
                    self.assertEqual(str(select_device('auto')),'cpu')

    def test_unavailable_accelerators_not_offered_in_training(self):
        torch=Mock();torch.cuda.is_available.return_value=False
        torch.backends.mps.is_available.return_value=False
        self.assertEqual(runtime.available_training_devices(torch),['auto','cpu'])

    def test_windows_worker_hidden_console_and_utf8_options(self):
        with patch.object(runtime.sys,'platform','win32'),\
             patch.object(subprocess,'CREATE_NO_WINDOW',0x08000000,create=True):
            options=runtime.worker_process_options()
        self.assertEqual(options['creationflags'],0x08000000)
        self.assertEqual(options['encoding'],'utf-8')
        self.assertEqual(options['env']['PYTHONIOENCODING'],'utf-8')

    def test_real_child_process_chinese_logs(self):
        child=subprocess.run([sys.executable,'-c',"print('训练完成：身份 chi')"],
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             **runtime.worker_process_options(),check=True)
        self.assertEqual(child.stdout.strip(),'训练完成：身份 chi')

    def test_windowed_worker_recovers_inherited_windows_stream(self):
        stream=io.StringIO()
        with tempfile.TemporaryDirectory() as directory,\
             patch.object(sys,'platform','win32'),patch.object(sys,'stdout',None),\
             patch.object(runtime,'windows_worker_stream',return_value=stream) as restore:
            runtime.restore_worker_stream('stdout',1,Path(directory))
            restore.assert_called_once_with(-11)
            self.assertIs(sys.stdout,stream)


if __name__=='__main__':
    unittest.main()
