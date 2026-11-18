"""VigilScreen：实时注视监测、训练材料标记与身份训练验证的一体窗口。"""
import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox,ttk

from runtime_paths import data_root
ROOT = data_root()
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

from chi_camera import ChiCamera,build_parser
from gui import TrainingWindow
from initialization import reset_project_data
from tray_window import attach_tray


class VigilScreen:
    def __init__(self,root,autostart=False):
        self.root = root
        self.closing = False
        self.resetting = False
        self.tab_generation = 0
        self.autostart = autostart
        self.tray = None
        root.title('VigilScreen · 视线监测与身份训练')
        root.geometry(f'{min(1280,root.winfo_screenwidth()-70)}x{min(850,root.winfo_screenheight()-90)}')
        root.minsize(1040,680)
        toolbar = ttk.Frame(root,padding=(12,8))
        toolbar.pack(fill='x')
        self.app_status = tk.StringVar(value='点击开始监控后才打开摄像头')
        ttk.Label(toolbar,textvariable=self.app_status).pack(side='left')
        self.initialize_button = ttk.Button(toolbar,text='初始化',command=self.initialize)
        self.initialize_button.pack(side='right')
        self.build_views()
        root.protocol('WM_DELETE_WINDOW',self.close)

    def build_views(self):
        root = self.root
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both',expand=True)
        monitor_frame = ttk.Frame(self.notebook)
        studio_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame,text='实时监测')
        self.notebook.add(studio_frame,text='标记 · 训练 · 验证')
        self.monitor = ChiCamera(monitor_frame,build_parser().parse_args([]),embedded=True,
                                 autostart=False,on_close=self.close)
        self.studio = TrainingWindow(studio_frame,embedded=True,on_trained=self.model_trained,
                                     camera_guard=lambda:not self.monitor.worker.is_alive())
        self.notebook.bind('<<NotebookTabChanged>>',self.change_tab)
        self.tray = attach_tray(root,self.monitor.menu_indicator,self.close,self.tray)

    def change_tab(self,event=None):
        self.tab_generation += 1
        generation = self.tab_generation
        if self.closing or self.resetting:
            return
        if self.notebook.index('current')==1:
            self.monitor.pause_monitor()
            return
        self.studio.stop_camera()
        def resume():
            if self.closing or generation!=self.tab_generation:
                return
            camera = self.studio.camera
            if camera is not None and camera.thread.is_alive():
                self.monitor.status.set('正在释放工作室摄像头，请稍候…')
                self.root.after(100,resume)
            elif self.autostart:
                self.monitor.start_monitor()
        resume()

    def model_trained(self,profile):
        if self.resetting or self.closing:
            return
        self.monitor.set_profile(profile)
        self.studio.training_status.set('训练、身份登记与验证完成 · 返回实时监测选择身份即可使用')

    def initialize(self):
        if self.resetting or self.closing:
            return
        confirmed = messagebox.askyesno('初始化确认',
            '初始化会清空所有模型训练、身份识别相关信息，返回初始数据状态。\n\n'
            '将清空本项目的训练图片、标记及裁剪记录、已训练模型、身份样本、训练历史、验证报告和命中截图；'
            '同时清除目标选择、过滤名单和视线校准。\n\n'
            '正在进行的监控、采集、训练或验证会先停止。\n'
            '运行所需的通用视觉模型会保留；项目外的原图、模型和备份不删除。\n\n'
            '此操作不可撤销。是否继续初始化？',parent=self.root,icon='warning',default='no')
        if not confirmed:
            return
        self.resetting = True
        self.tab_generation += 1
        self.autostart = False
        self.initialize_button.configure(state='disabled')
        self.app_status.set('正在初始化：停止监控、摄像头与训练任务…')
        dialog = tk.Toplevel(self.root)
        dialog.title('正在初始化')
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.protocol('WM_DELETE_WINDOW',lambda:None)
        ttk.Label(dialog,textvariable=self.app_status,padding=24).pack()
        self.reset_dialog = dialog
        self.monitor.pause_monitor()
        self.studio.stop_camera()
        self.studio.stopping = True
        self.studio.stop()
        self.reset_deadline = time.monotonic()+15
        self.reset_kill_at = time.monotonic()+5
        self.wait_for_reset_idle()

    def wait_for_reset_idle(self):
        process = self.studio.process
        process_alive = process is not None and process.poll() is None
        camera = self.studio.camera
        camera_alive = camera is not None and camera.thread.is_alive()
        if process_alive and time.monotonic()>=self.reset_kill_at:
            process.kill()
        if self.monitor.worker.is_alive() or camera_alive or process_alive:
            if time.monotonic()>=self.reset_deadline:
                self.end_initialization('后台任务尚未退出，未清空数据。请等待任务停止后重试。')
                return
            self.root.after(100,self.wait_for_reset_idle)
            return
        # 所有数据写入者退出之后，停止旧界面轮询，避免重新载入待清理的模型。
        self.monitor.shutdown()
        self.studio.shutdown()
        self.app_status.set('正在初始化：清空训练、身份数据与截图…')
        self.reset_results = queue.Queue()
        def clear():
            try:
                reset_project_data(ROOT)
            except Exception as error:
                self.reset_results.put(str(error))
            else:
                self.reset_results.put(None)
        threading.Thread(target=clear,daemon=True).start()
        self.poll_initialization()

    def poll_initialization(self):
        try:
            error = self.reset_results.get_nowait()
        except queue.Empty:
            self.root.after(100,self.poll_initialization)
            return
        self.notebook.destroy()
        self.build_views()
        self.end_initialization(error)

    def end_initialization(self,error=None):
        self.reset_dialog.destroy()
        self.resetting = False
        self.initialize_button.configure(state='normal')
        if error is not None:
            self.app_status.set('初始化未完成，请检查提示后重试')
            messagebox.showerror('初始化未完成',error,parent=self.root)
        else:
            self.app_status.set('初始化完成 · 当前为全员模式 · 点击开始监控后才打开摄像头')
            self.monitor.status.set('初始化完成：已清空训练与身份信息，请重新导入材料或开始全员监控。')
            messagebox.showinfo('初始化完成','已恢复初始数据状态。摄像头保持关闭，可重新训练身份或点击开始监控。',parent=self.root)

    def close(self):
        if self.closing or self.resetting:
            return
        self.closing = True
        if self.tray is not None:
            self.tray.close()
        self.root.withdraw()
        self.monitor.shutdown()
        self.studio.stop_camera()
        self.studio.stop()
        deadline = time.monotonic()+5
        def finish():
            process = self.studio.process
            process_alive = process is not None and process.poll() is None
            camera_alive = self.studio.camera is not None and self.studio.camera.thread.is_alive()
            if time.monotonic()<deadline and (self.monitor.worker.is_alive() or process_alive or camera_alive):
                self.root.after(100,finish)
                return
            if process_alive:
                process.kill()
                process.wait(timeout=2)
            self.root.destroy()
        finish()


def main():
    root = tk.Tk()
    VigilScreen(root)
    root.mainloop()


if __name__=='__main__':
    main()
