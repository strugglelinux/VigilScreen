"""本地图像标记与训练窗口：python gui.py。"""

import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from uuid import uuid4
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from labeling import LabelStore, image_files, selection_box
from camera import CameraStream, save_capture

from runtime_paths import data_root,worker_command
ROOT = data_root()


class TrainingWindow:
    def __init__(self,root,embedded=False,on_trained=None,camera_guard=None):
        self.root = root
        self.closed = False
        self.on_trained = on_trained
        self.camera_guard = camera_guard
        self.task_kind = 'train'
        self.last_validation = None
        self.last_validation_profile = None
        if not embedded:
            root.title('VigilScreen · 标记与训练工作室')
            root.geometry('1200x820')
        self.process = None
        self.messages = queue.Queue()
        self.images = []
        self.index = 0
        self.store = None
        self.label_picture = None
        self.label_roi = None
        self.label_drag = None
        self.history = []
        self.running_output = None
        self.stopping = False
        self.camera = None
        self.camera_frame = None
        self.camera_snapshot = None
        self.camera_roi = None
        self.camera_drag = None
        self.output = tk.StringVar(value=str(ROOT / 'data' / 'labeled'))
        self.category = tk.StringVar()
        self.image_status = tk.StringVar(value='导入图片目录开始标记')
        self.training_status = tk.StringVar(value='就绪')
        notebook = ttk.Notebook(root)
        notebook.pack(fill='both',expand=True,padx=8,pady=8)
        self.notebook = notebook
        labels = ttk.Frame(notebook, padding=12)
        training = ttk.Frame(notebook, padding=12)
        notebook.add(labels, text='图片手动标记')
        camera = ttk.Frame(notebook, padding=12)
        notebook.add(camera, text='摄像头标记')
        notebook.add(training, text='可视化训练')
        self.build_labels(labels)
        self.build_training(training)
        self.build_camera(camera)
        validation = ttk.Frame(notebook,padding=12)
        notebook.add(validation,text='身份效果验证')
        self.build_validation(validation)
        if not embedded:
            root.protocol('WM_DELETE_WINDOW',self.close)
        self.poll_timer = root.after(150, self.poll)

    def path_row(self, parent, title, variable):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=4)
        ttk.Label(frame, text=title, width=14).pack(side='left')
        ttk.Entry(frame, textvariable=variable).pack(side='left', fill='x', expand=True)
        ttk.Button(frame, text='选择目录', command=lambda: self.choose_directory(variable)).pack(side='left', padx=5)

    def choose_directory(self, variable):
        directory = filedialog.askdirectory()
        if directory:
            variable.set(directory)

    def build_labels(self, frame):
        self.path_row(frame, '标记输出目录', self.output)
        bar = ttk.Frame(frame)
        bar.pack(fill='x', pady=8)
        ttk.Button(bar, text='导入图片目录（含子目录）', command=self.import_images).pack(side='left')
        ttk.Label(bar, text='原图保留；标记副本按类别保存，可直接用于训练。').pack(side='left', padx=12)
        self.preview = tk.Canvas(frame, bg='#20252b', height=420, highlightthickness=0)
        self.preview.bind('<ButtonPress-1>', self.begin_label_selection)
        self.preview.bind('<B1-Motion>', self.drag_label_selection)
        self.preview.bind('<ButtonRelease-1>', self.end_label_selection)
        self.preview.bind('<Configure>', lambda event: self.draw_label_image())
        self.preview.pack(fill='both', expand=True)
        ttk.Label(frame, textvariable=self.image_status, wraplength=1000).pack(pady=8)
        bar = ttk.Frame(frame)
        bar.pack(pady=8)
        ttk.Button(bar, text='上一张', command=lambda: self.move(-1)).pack(side='left', padx=5)
        ttk.Label(bar, text='类别（可输入新类别）：').pack(side='left')
        self.categories = ttk.Combobox(bar, textvariable=self.category, width=24)
        self.categories.pack(side='left', padx=5)
        ttk.Button(bar, text='保存标记并下一张', command=self.save_label).pack(side='left', padx=5)
        ttk.Button(bar, text='跳过 / 下一张', command=lambda: self.move(1)).pack(side='left', padx=5)
        sizing = ttk.Frame(frame)
        sizing.pack(pady=4)
        ttk.Button(sizing, text='清除选框 / 使用整图', command=self.clear_label_selection).pack(side='left', padx=6)
        ttk.Label(sizing, text='保存最长边（像素，0 不缩放）：').pack(side='left')
        self.label_max_side = tk.StringVar(value='256')
        ttk.Spinbox(sizing, from_=0, to=4096, increment=32, textvariable=self.label_max_side, width=7).pack(side='left')
        ttk.Label(frame, text='拖动画框选择训练区域，再输入类别保存。原图保留，不画框时保存整图。').pack(pady=3)
        ttk.Button(frame, text='将标记输出目录设为训练数据', command=lambda: self.data_dir.set(self.output.get())).pack(pady=5)

    def import_images(self):
        directory = filedialog.askdirectory(title='选择原始图片目录')
        if not directory:
            return
        try:
            output = Path(self.output.get()).expanduser().resolve()
            source = Path(directory).resolve()
            if source == output or source.is_relative_to(output):
                raise ValueError('原图目录不能位于标记输出目录中。')
            self.store = LabelStore(output)
            self.images = [p for p in image_files(source) if not p.is_relative_to(output)]
            self.index = 0
            self.show_image()
        except (OSError, ValueError) as error:
            messagebox.showerror('导入失败', str(error))

    def show_image(self):
        self.label_picture = None
        self.label_roi = None
        self.label_drag = None
        self.preview.delete('all')
        if not self.images:
            self.image_status.set('目录中没有支持的图片。')
            return
        path = self.images[self.index]
        label = self.store.labels.get(str(path), '')
        self.category.set(label)
        self.categories.configure(values=sorted(set(self.store.labels.values())))
        self.image_status.set(f'{self.index + 1} / {len(self.images)} · {path}\n当前标记：{label or "未标记"}')
        try:
            with Image.open(path) as picture:
                self.label_picture = ImageOps.exif_transpose(picture).convert('RGB')
            region = self.store.regions.get(str(path))
            if region:
                self.label_roi = tuple(region['box']) if region['box'] is not None else None
                self.label_max_side.set(str(region['max_side']))
            self.draw_label_image()
        except (OSError, ValueError) as error:
            self.label_picture = None
            self.preview.create_text(20, 20, text=f'图片无法打开，可以跳过：{error}', fill='white', anchor='nw')

    def draw_label_image(self):
        if self.label_picture is None:
            return
        picture = self.label_picture
        preview = picture.copy()
        width, height = max(1, self.preview.winfo_width()), max(1, self.preview.winfo_height())
        preview.thumbnail((width, height))
        x, y = (width-preview.width)/2, (height-preview.height)/2
        self.label_display_box = (x, y, preview.width, preview.height)
        self.photo = ImageTk.PhotoImage(preview)
        self.preview.delete('all')
        self.preview.create_image(x, y, image=self.photo, anchor='nw')
        if self.label_roi is not None:
            left, top, right, bottom = self.label_roi
            sx, sy = preview.width/picture.width, preview.height/picture.height
            self.preview.create_rectangle(x+left*sx, y+top*sy, x+right*sx, y+bottom*sy,
                                          outline='#22ff88', width=3, tags='selection')

    def begin_label_selection(self, event):
        if self.label_picture is None:
            return
        x, y, width, height = self.label_display_box
        if x <= event.x <= x+width and y <= event.y <= y+height:
            self.label_drag = (event.x, event.y)

    def drag_label_selection(self, event):
        if self.label_drag is None:
            return
        x, y, width, height = self.label_display_box
        self.preview.delete('selection')
        self.preview.create_rectangle(*self.label_drag, max(x, min(x+width, event.x)),
                                      max(y, min(y+height, event.y)),
                                      outline='#22ff88', width=3, tags='selection')

    def end_label_selection(self, event):
        if self.label_drag is None:
            return
        try:
            self.label_roi = selection_box(self.label_drag, (event.x, event.y),
                                           self.label_display_box, self.label_picture.size)
            left, top, right, bottom = self.label_roi
            self.image_status.set(f'{self.images[self.index]}\n选区：{right-left} × {bottom-top} 像素 · 输入类别后保存')
        except ValueError as error:
            self.image_status.set(f'{error} 当前选区保持不变。')
        self.label_drag = None
        self.draw_label_image()

    def clear_label_selection(self):
        self.label_roi = None
        self.label_drag = None
        self.draw_label_image()
        self.image_status.set('已清除选框，保存时使用整张图片。')

    def move(self, delta):
        if self.images:
            self.index = max(0, min(len(self.images) - 1, self.index + delta))
            self.show_image()

    def save_label(self):
        if not self.images:
            return
        if self.process is not None:
            messagebox.showinfo('训练进行中', '请在训练结束或停止后修改标记。')
            return
        try:
            if self.store.root != Path(self.output.get()).expanduser().resolve():
                raise ValueError('输出目录已更改，请重新导入原始图片目录。')
            category = self.category.get()
            self.store.save(self.images[self.index], category, box=self.label_roi,
                            max_side=int(self.label_max_side.get()))
            if self.index == len(self.images) - 1:
                self.show_image()
                messagebox.showinfo('保存成功', '最后一张图片已保存标记。')
            else:
                self.move(1)
                self.category.set(category)
        except (OSError, ValueError) as error:
            messagebox.showerror('标记失败', str(error))

    def build_camera(self, frame):
        self.path_row(frame, '标记输出目录', self.output)
        bar = ttk.Frame(frame)
        bar.pack(fill='x', pady=8)
        ttk.Label(bar, text='摄像头编号：').pack(side='left')
        self.camera_index = tk.StringVar(value='0')
        ttk.Spinbox(bar, from_=0, to=20, textvariable=self.camera_index, width=5).pack(side='left')
        ttk.Button(bar, text='打开摄像头', command=self.open_camera).pack(side='left', padx=8)
        ttk.Button(bar, text='关闭摄像头', command=self.stop_camera).pack(side='left')
        self.camera_preview = tk.Canvas(frame, bg='#20252b', height=420, highlightthickness=0)
        self.camera_preview.bind('<ButtonPress-1>', self.begin_camera_selection)
        self.camera_preview.bind('<B1-Motion>', self.drag_camera_selection)
        self.camera_preview.bind('<ButtonRelease-1>', self.end_camera_selection)
        self.camera_preview.bind('<Configure>', self.redraw_camera)
        self.camera_preview.pack(fill='both', expand=True)
        self.camera_status = tk.StringVar(value='摄像头未打开')
        ttk.Label(frame, textvariable=self.camera_status, wraplength=980).pack(pady=10)
        self.camera_saved = tk.StringVar()
        ttk.Label(frame, textvariable=self.camera_saved, wraplength=980).pack(pady=3)
        bar = ttk.Frame(frame)
        bar.pack(pady=8)
        ttk.Button(bar, text='拍照 / 冻结画面', command=self.freeze_camera).pack(side='left', padx=5)
        ttk.Label(bar, text='类别：').pack(side='left')
        self.camera_category = tk.StringVar()
        self.camera_categories = ttk.Combobox(bar, textvariable=self.camera_category, width=22)
        self.camera_categories.pack(side='left', padx=5)
        ttk.Button(bar, text='保存标记', command=self.save_camera_label).pack(side='left', padx=5)
        ttk.Button(bar, text='继续预览', command=self.resume_camera).pack(side='left', padx=5)
        sizing = ttk.Frame(frame)
        sizing.pack(pady=4)
        ttk.Button(sizing, text='清除选框 / 使用整图', command=self.clear_camera_selection).pack(side='left', padx=6)
        ttk.Label(sizing, text='保存最长边（像素，0 不缩放）：').pack(side='left')
        self.camera_max_side = tk.StringVar(value='256')
        ttk.Spinbox(sizing, from_=0, to=4096, increment=32, textvariable=self.camera_max_side, width=7).pack(side='left')
        ttk.Button(frame, text='将标记输出目录设为训练数据',
                   command=lambda: self.data_dir.set(self.output.get())).pack(pady=5)
        ttk.Label(frame, text='在实时画面上按下鼠标即可冻结，继续拖动画框；输入类别后点击保存标记。').pack(pady=5)
        self.camera_poll_timer = self.root.after(40, self.poll_camera)

    def open_camera(self):
        if self.camera_guard is not None and not self.camera_guard():
            self.camera_status.set('监测摄像头正在释放，请稍后再次点击打开。')
            return
        if self.camera is not None:
            self.camera_status.set('请先关闭当前摄像头，再切换设备。')
            return
        try:
            index = int(self.camera_index.get())
            if index < 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror('摄像头编号无效', '请输入大于等于 0 的整数。')
            return
        self.camera_frame = None
        self.camera_snapshot = None
        self.camera = CameraStream(index)
        self.camera.start()
        self.camera_status.set('正在打开摄像头，请允许系统摄像头权限…')

    def stop_camera(self):
        if self.camera is not None:
            self.camera.stop()
            # 等待后台释放完成后才允许再次打开。
            self.camera_status.set('正在关闭摄像头…')
        self.camera_frame = None
        if self.camera_snapshot is None:
            self.camera_preview.delete('all')

    def poll_camera(self):
        if self.closed:
            return
        if self.camera is not None:
            if self.camera.stopped.is_set():
                if not self.camera.thread.is_alive():
                    self.camera = None
                    self.camera_status.set('摄像头已关闭；已拍摄的画面仍可标记')
            else:
                try:
                    kind, value = self.camera.events.get_nowait()
                    if kind == 'error':
                        self.camera.stop()
                        self.camera_frame = None
                        self.camera_status.set(value)
                        messagebox.showerror('摄像头不可用', value)
                    else:
                        self.camera_frame = value
                        if self.camera_snapshot is None:
                            self.display_camera(value)
                            self.camera_status.set('实时预览 · 在画面上按下鼠标冻结并拖动画框')
                except queue.Empty:
                    pass
        self.camera_poll_timer = self.root.after(40, self.poll_camera)

    def display_camera(self, picture):
        preview = picture.copy()
        width = max(1, self.camera_preview.winfo_width())
        height = max(1, self.camera_preview.winfo_height())
        preview.thumbnail((width, height))
        x, y = (width - preview.width) / 2, (height - preview.height) / 2
        self.camera_display_box = (x, y, preview.width, preview.height)
        self.camera_photo = ImageTk.PhotoImage(preview)
        self.camera_preview.delete('all')
        self.camera_preview.create_image(x, y, image=self.camera_photo, anchor='nw')
        if self.camera_snapshot is not None and self.camera_roi is not None:
            left, top, right, bottom = self.camera_roi
            sx, sy = preview.width / picture.width, preview.height / picture.height
            self.camera_preview.create_rectangle(x + left*sx, y + top*sy, x + right*sx, y + bottom*sy,
                                                 outline='#22ff88', width=3, tags='selection')

    def redraw_camera(self, event=None):
        picture = self.camera_snapshot if self.camera_snapshot is not None else self.camera_frame
        if picture is not None:
            self.display_camera(picture)

    def begin_camera_selection(self, event):
        if self.camera_snapshot is None and self.camera_frame is None:
            self.camera_status.set('请先打开摄像头并等待画面。')
            return
        x, y, width, height = self.camera_display_box
        if not (x <= event.x <= x + width and y <= event.y <= y + height):
            return
        if self.camera_snapshot is None:
            self.freeze_camera()
        self.camera_drag = (event.x, event.y)

    def drag_camera_selection(self, event):
        if self.camera_drag is None or self.camera_snapshot is None:
            return
        x, y, width, height = self.camera_display_box
        end_x = max(x, min(x + width, event.x))
        end_y = max(y, min(y + height, event.y))
        self.camera_preview.delete('selection')
        self.camera_preview.create_rectangle(*self.camera_drag, end_x, end_y,
                                             outline='#22ff88', width=3, tags='selection')

    def end_camera_selection(self, event):
        if self.camera_drag is None or self.camera_snapshot is None:
            return
        if self.camera_drag == (event.x, event.y):
            self.camera_drag = None
            self.redraw_camera()
            return
        try:
            self.camera_roi = selection_box(self.camera_drag, (event.x, event.y),
                                            self.camera_display_box, self.camera_snapshot.size)
            left, top, right, bottom = self.camera_roi
            self.camera_status.set(f'选区：{right-left} × {bottom-top} 像素 · 输入类别后保存')
        except ValueError as error:
            self.camera_status.set(f'{error} 当前选区保持不变。')
        self.camera_drag = None
        self.redraw_camera()

    def clear_camera_selection(self):
        self.camera_roi = None
        self.camera_drag = None
        self.redraw_camera()
        self.camera_status.set('已清除选框，保存时使用整张图片。')

    def freeze_camera(self):
        if self.camera_frame is None:
            self.camera_status.set('请先打开摄像头并等待画面。')
            return
        self.camera_roi = None
        self.camera_drag = None
        self.camera_snapshot = self.camera_frame.copy()
        self.display_camera(self.camera_snapshot)
        self.camera_status.set('画面已冻结 · 拖动画框选择训练区域，再输入类别保存')

    def resume_camera(self):
        self.camera_roi = None
        self.camera_drag = None
        self.camera_snapshot = None
        if self.camera_frame is not None:
            self.display_camera(self.camera_frame)
        else:
            self.camera_preview.delete('all')
        self.camera_status.set('等待摄像头画面…')

    def save_camera_label(self):
        if self.camera_snapshot is None:
            self.camera_status.set('请先点击摄像头画面冻结并框选，再保存标记。')
            return
        if self.process is not None:
            messagebox.showinfo('训练进行中', '请在训练结束或停止后保存标记。')
            return
        try:
            if not self.output.get().strip():
                raise ValueError('请选择标记输出目录。')
            max_side = int(self.camera_max_side.get())
            target = save_capture(self.camera_snapshot, Path(self.output.get()), self.camera_category.get(),
                                  box=self.camera_roi, max_side=max_side)
            with Image.open(target) as saved:
                dimensions = f'{saved.width} × {saved.height}'
            self.camera_categories.configure(values=sorted(set(LabelStore(Path(self.output.get())).labels.values())))
            self.resume_camera()
            self.camera_saved.set(f'上次保存：{dimensions} 像素 · {target}')
        except (OSError, ValueError) as error:
            messagebox.showerror('保存失败', str(error))

    def build_training(self, frame):
        self.data_dir = tk.StringVar(value=self.output.get())
        self.checkpoints = tk.StringVar(value=str(ROOT / 'checkpoints' / 'identities'))
        self.path_row(frame, '训练数据目录', self.data_dir)
        self.path_row(frame, '模型输出目录', self.checkpoints)
        person_row = ttk.Frame(frame)
        person_row.pack(fill='x',pady=4)
        ttk.Label(person_row,text='单人材料名称（平铺图片目录时填写）：').pack(side='left')
        self.person_name = tk.StringVar()
        ttk.Entry(person_row,textvariable=self.person_name,width=24).pack(side='left',padx=8)
        options = ttk.Frame(frame)
        options.pack(fill='x', pady=8)
        self.fields = {}
        for title, name, default, choices in [
            ('数据集', 'dataset', 'folder', ['folder']),
            ('设备', 'device', 'auto', ['auto', 'cpu', 'mps', 'cuda']),
            ('轮数', 'epochs', '20', None), ('批大小', 'batch-size', '64', None),
            ('学习率', 'learning-rate', '0.001', None), ('验证比例', 'val-ratio', '0.2', None),
        ]:
            ttk.Label(options, text=title).pack(side='left', padx=4)
            variable = tk.StringVar(value=default)
            self.fields[name] = variable
            if choices:
                ttk.Combobox(options, textvariable=variable, values=choices, state='readonly', width=8).pack(side='left')
            else:
                ttk.Entry(options, textvariable=variable, width=7).pack(side='left')
        self.download = tk.BooleanVar()
        ttk.Label(frame,text='每个身份一个子目录；每类至少 4 张清晰人脸，训练分层划分后至少 3 张可登记。单类别可登记身份，CNN 准确率不代表陌生人排除能力。',wraplength=1100).pack(anchor='w')
        ttk.Label(frame,text='曲线显示 CNN 分类指标；实时身份使用 SFace 登记特征，请在“身份效果验证”查看实际匹配结果。',wraplength=1100).pack(anchor='w')
        bar = ttk.Frame(frame)
        bar.pack(fill='x', pady=8)
        self.start_button = ttk.Button(bar, text='开始训练并登记身份', command=self.start)
        self.start_button.pack(side='left')
        self.stop_button = ttk.Button(bar, text='停止训练', command=self.stop, state='disabled')
        self.stop_button.pack(side='left', padx=8)
        ttk.Label(bar, textvariable=self.training_status).pack(side='left', padx=8)
        self.progress = ttk.Progressbar(frame, maximum=100)
        self.progress.pack(fill='x')
        self.chart = tk.Canvas(frame, height=280, bg='white', highlightthickness=0)
        self.chart.pack(fill='both', expand=True, pady=10)
        self.chart.bind('<Configure>', lambda event: self.draw_chart())
        self.log = tk.Text(frame, height=12, state='disabled', wrap='word')
        self.log.pack(fill='both', expand=True)
        self.draw_chart()

    def start(self):
        if self.process is not None:
            return
        # 使用原命令行参数解析器校验参数，训练仍运行在独立进程。
        command = worker_command('train',
                   '--data-dir', self.data_dir.get(), '--output-dir', self.checkpoints.get(),
                   '--person-name',self.person_name.get().strip())
        for name, variable in self.fields.items():
            command.extend(['--' + name, variable.get()])
        if self.download.get():
            command.append('--download')
        try:
            if not self.data_dir.get().strip() or not self.checkpoints.get().strip():
                raise ValueError('请选择训练数据目录和模型输出目录。')
            if not image_files(Path(self.data_dir.get()).expanduser().resolve()):
                raise ValueError('训练目录中没有图片，请先导入或标记训练材料。')
            self.running_output = Path(self.checkpoints.get()).expanduser().resolve()
            command[command.index('--data-dir') + 1] = str(Path(self.data_dir.get()).expanduser().resolve())
            command[command.index('--output-dir') + 1] = str(self.running_output)
            self.history = []
            self.history_started = False
            self.total_epochs = max(1, int(self.fields['epochs'].get()))
            self.stopping = False
            self.progress['value'] = 0
            self.draw_chart()
            self.process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1)
        except (OSError, ValueError) as error:
            messagebox.showerror('启动失败', str(error))
            return
        self.task_kind = 'train'
        self.start_button.configure(state='disabled')
        self.validation_run_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.training_status.set('训练启动中…')
        self.append_log('\n—— 开始新的训练 ——\n')
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()

    def read_output(self, process):
        for line in process.stdout:
            self.messages.put(('log', line))
        process.stdout.close()
        self.messages.put(('exit', process.wait()))

    def append_log(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text)
        self.log.see('end')
        self.log.configure(state='disabled')

    def poll(self):
        if self.closed:
            return
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == 'log':
                    self.append_log(value)
                    if self.task_kind=='validate':
                        self.append_validation_log(value)
                        if value.startswith('Validation '):
                            self.validation_status.set(value.strip())
                    if value.startswith('Epoch '):
                        self.history_started = True
                        self.refresh_history()
                else:
                    self.training_status.set('已停止' if self.stopping else ('任务完成' if value==0 else '任务失败，请查看日志'))
                    self.process = None
                    self.start_button.configure(state='normal')
                    self.validation_run_button.configure(state='normal')
                    self.stop_button.configure(state='disabled')
                    if value==0 and not self.stopping:
                        if self.task_kind=='train':
                            self.checkpoints.set(str(self.running_output))
                            self.last_validation = None
                            self.last_validation_profile = None
                            if self.on_trained is not None:
                                self.on_trained(self.running_output/'active_profile.json')
                        else:
                            self.last_validation = self.validation_report_path
                            self.last_validation_profile = self.validation_job_profile
                        self.display_validation()
                    elif self.task_kind=='validate':
                        self.clear_validation_table()
                        self.validation_status.set('验证已停止。' if self.stopping else '验证失败，请查看可视化训练页中的任务日志。')
        except queue.Empty:
            pass
        if self.process is not None and self.task_kind=='train' and getattr(self,'history_started',False):
            self.refresh_history()
        self.poll_timer = self.root.after(200, self.poll)

    def refresh_history(self):
        try:
            history = json.loads((self.running_output / 'history.json').read_text(encoding='utf-8'))
            if history and history != self.history:
                self.history = history
                self.progress['value'] = 100 * history[-1]['epoch'] / self.total_epochs
                # 总轮数由当前进程的日志提供，不依赖训练中修改的输入框。
                self.draw_chart()
                self.training_status.set(f'已完成 {history[-1]["epoch"]} 轮 · 验证准确率 {history[-1]["val"]["accuracy"]:.2%}')
        except (OSError, ValueError, KeyError):
            pass

    def draw_chart(self):
        self.chart.delete('all')
        width = max(self.chart.winfo_width(), 700)
        height = max(self.chart.winfo_height(), 240)
        for panel, metric, title in [(0, 'loss', '损失'), (1, 'accuracy', '准确率')]:
            left, right = panel * width / 2 + 55, (panel + 1) * width / 2 - 25
            top, bottom = 40, height - 35
            self.chart.create_text((left + right) / 2, 15, text='CNN '+title+'    蓝：训练 / 橙：验证')
            self.chart.create_line(left, top, left, bottom, right, bottom, fill='#888888')
            maximum = 1 if metric == 'accuracy' else max([row[split][metric] for row in self.history for split in ('train', 'val')] + [0.1]) * 1.1
            for tick in range(5):
                value = maximum * tick / 4
                y = bottom - (bottom - top) * tick / 4
                self.chart.create_text(left - 6, y, text=f'{value:.2f}', anchor='e')
            self.chart.create_text((left + right) / 2, height - 12, text=f'轮次 1 — {len(self.history)}' if self.history else '等待训练数据')
            for split, color in [('train', '#2563eb'), ('val', '#ea580c')]:
                points = []
                for i, row in enumerate(self.history):
                    x = left + i / max(1, len(self.history) - 1) * (right - left)
                    y = bottom - row[split][metric] / maximum * (bottom - top)
                    points.extend((x, y))
                    self.chart.create_oval(x-3, y-3, x+3, y+3, fill=color, outline=color)
                if len(points) >= 4:
                    self.chart.create_line(*points, fill=color, width=2)

    def build_validation(self,frame):
        self.path_row(frame,'模型输出目录',self.checkpoints)
        self.validation_dir = tk.StringVar()
        self.path_row(frame,'外部验证目录',self.validation_dir)
        bar = ttk.Frame(frame)
        bar.pack(fill='x',pady=8)
        ttk.Label(bar,text='平铺图片的预期身份：').pack(side='left')
        self.validation_expected = tk.StringVar(value='按子目录类别')
        from identity_training import available_identities
        self.validation_expected_choice = ttk.Combobox(bar,textvariable=self.validation_expected,
            values=['按子目录类别',*available_identities(Path(self.checkpoints.get())/'active_profile.json'),'unknown'],width=22)
        self.validation_expected_choice.pack(side='left',padx=8)
        self.validation_run_button = ttk.Button(bar,text='运行身份验证',command=self.validate_identity)
        self.validation_run_button.pack(side='left',padx=8)
        ttk.Button(bar,text='查看已有验证结果',command=self.display_validation).pack(side='left')
        self.validation_status = tk.StringVar(value='外部目录留空时使用训练时保留的验证原图。此处验证身份匹配，不验证看向屏幕的准确率。')
        ttk.Label(frame,textvariable=self.validation_status,wraplength=1080).pack(fill='x',pady=8)
        self.validation_log = tk.Text(frame,height=4,state='disabled',wrap='word')
        self.validation_log.pack(fill='x',pady=(0,8))
        columns = ('expected','predicted','score','overlap','file','reason')
        self.validation_table = ttk.Treeview(frame,columns=columns,show='headings')
        for key,title,width in [('expected','预期身份',100),('predicted','识别结果',100),('score','相似度',80),
                                ('overlap','与训练重复',100),('file','文件',260),('reason','检测情况',220)]:
            self.validation_table.heading(key,text=title)
            self.validation_table.column(key,width=width,minwidth=60)
        scroll = ttk.Scrollbar(frame,orient='vertical',command=self.validation_table.yview)
        scroll.pack(side='right',fill='y')
        self.validation_table.configure(yscrollcommand=scroll.set)
        self.validation_table.pack(fill='both',expand=True)
        ttk.Label(frame,text='外部验证建议包含各身份子目录和 unknown/ 陌生人目录。训练过的同内容图片会标记重复，不能当作独立验证。',wraplength=1080).pack(fill='x',pady=8)
        self.checkpoints.trace_add('write',self.validation_model_changed)

    def clear_validation_table(self):
        for item in self.validation_table.get_children():
            self.validation_table.delete(item)

    def validation_model_changed(self,*args):
        self.clear_validation_table()
        self.last_validation = None
        self.last_validation_profile = None
        self.validation_status.set('模型目录已更改，请重新查看已有结果或运行验证。')

    def append_validation_log(self,text,clear=False):
        self.validation_log.configure(state='normal')
        if clear:
            self.validation_log.delete('1.0','end')
        self.validation_log.insert('end',text)
        self.validation_log.see('end')
        self.validation_log.configure(state='disabled')

    def validate_identity(self):
        if self.process is not None:
            messagebox.showinfo('任务进行中','请先等待当前训练或验证结束。')
            return
        try:
            from identity_training import heldout_records,read_profile,validation_records
            output = Path(self.checkpoints.get()).expanduser().resolve()
            profile = output/'active_profile.json'
            if not profile.is_file():
                raise ValueError('尚无身份模型，请先训练身份，或选择已有身份模型的输出目录。')
            profile,metadata = read_profile(profile)
            expected = None if self.validation_expected.get()=='按子目录类别' else self.validation_expected.get().strip()
            if expected is not None and expected not in [*metadata['classes'],'unknown']:
                raise ValueError(f'预期身份无效，请选择 {metadata["classes"]} 或 unknown。')
            records = validation_records(self.validation_dir.get(),expected) if self.validation_dir.get().strip() else heldout_records(profile,metadata)
            if not records:
                raise ValueError('验证目录中没有可用图片。')
            self.validation_job_profile = profile
            self.validation_report_path = output/'validations'/f'{uuid4().hex[:12]}.json'
            command = worker_command('validate',
                       '--profile',str(profile),'--output',str(self.validation_report_path))
            if self.validation_dir.get().strip():
                command += ['--images-dir',str(Path(self.validation_dir.get()).expanduser().resolve())]
                if self.validation_expected.get()!='按子目录类别':
                    command += ['--expected',self.validation_expected.get().strip()]
            self.process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            self.task_kind = 'validate'
            self.stopping = False
            self.start_button.configure(state='disabled')
            self.validation_run_button.configure(state='disabled')
            self.stop_button.configure(state='normal')
            self.training_status.set('身份验证中…')
            self.clear_validation_table()
            self.append_validation_log(f'模型：{metadata["profile_id"]} · 图片 {len(records)} 张\n',clear=True)
            self.validation_status.set(f'正在验证 {len(records)} 张图片，请稍候…')
            threading.Thread(target=self.read_output,args=(self.process,),daemon=True).start()
        except (OSError,ValueError) as error:
            self.validation_status.set(f'验证无法启动：{error}')
            self.clear_validation_table()
            messagebox.showerror('验证失败',str(error))

    def display_validation(self):
        self.clear_validation_table()
        try:
            from identity_training import read_profile
            entry = Path(self.checkpoints.get()).expanduser()/'active_profile.json'
            if not entry.is_file():
                raise ValueError('尚无身份模型，请先训练身份，或选择已有身份模型的输出目录。')
            path,profile = read_profile(entry)
            self.validation_expected_choice.configure(values=['按子目录类别',*profile['classes'],'unknown'])
            report_path = self.last_validation if self.last_validation_profile==path else None
            report_path = report_path or path.parent/profile['validation']
            report = json.loads(Path(report_path).read_text(encoding='utf-8'))
            if report.get('profile_id',profile['profile_id'])!=profile['profile_id']:
                raise ValueError('验证报告不属于当前身份模型，请重新运行验证。')
            for row in report['details']:
                reason = row.get('error','已检测到唯一人脸')
                if row.get('provided_label',row['expected'])!=row['expected']:
                    reason += f" · {row['provided_label']} 未登记，按陌生人验证"
                self.validation_table.insert('', 'end',values=(row['expected'],row['predicted'],f"{row['similarity']:.3f}",
                    '是' if row['training_content_overlap'] else '否',Path(row['source']).name,reason))
            known = report['known_identity_match_rate']
            unknown = report['unknown_false_accept_rate']
            self.validation_status.set(f"图片 {report['images']} · 已知身份匹配：{known:.1%}" if known is not None else f"图片 {report['images']} · 无已知身份验证图")
            self.validation_status.set(self.validation_status.get()+
                (f' · 陌生人误接受：{unknown:.1%}' if unknown is not None else ' · 未提供可用陌生人验证图')+
                f" · 与训练重复 {report['training_content_overlaps']} · 不可用 {sum(r.get('unusable',0) for r in report['per_class'].values())}\n模型：{profile['profile_id']} · 报告：{report_path}\n已知身份匹配包含所有已知身份验证图；陌生人误接受仅按可用的唯一人脸图计算。仅衡量身份，不代表注视命中率。")
        except (OSError,ValueError,KeyError) as error:
            self.clear_validation_table()
            self.validation_status.set(f'尚无有效验证报告：{error}')

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.stopping = True
            self.process.terminate()
            self.stop_button.configure(state='disabled')
            self.training_status.set('正在停止… 已完成轮次的模型与记录保留')

    def shutdown(self):
        self.closed = True
        for attribute in ('poll_timer','camera_poll_timer'):
            timer = getattr(self,attribute,None)
            if timer is not None:
                self.root.after_cancel(timer)
        self.stop_camera()
        self.stop()

    def close(self):
        if self.camera is not None:
            self.camera.stop()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    TrainingWindow(root)
    root.mainloop()


if __name__ == '__main__':
    main()
