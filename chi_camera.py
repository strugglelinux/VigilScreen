"""识别所有人脸，目标注视镜头或已校准屏幕时在菜单栏提示命中并截图。"""
import argparse
import math
from datetime import datetime
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from gaze import GazeEstimator, GazeResult
from test import positive_int,positive_float,nonnegative_int
from runtime_paths import RESOURCE_ROOT, data_root

ROOT = data_root()
ALL_PEOPLE = '全部人脸（不指定身份）'


class HitGate:
    """连续确认，每次出现仅触发一次；离开后重新出现才再次触发。"""
    def __init__(self, confirmations=1, cooldown=0.0, reset_frames=3):
        self.confirmations = confirmations
        self.cooldown = cooldown
        self.reset_frames = reset_frames
        self.matches = self.misses = 0
        self.latched = False
        self.last_hit = float('-inf')

    def update(self, matched, now):
        if not matched:
            self.matches = 0
            self.misses += 1
            if self.misses >= self.reset_frames:
                self.latched = False
            return False
        self.misses = 0
        self.matches += 1
        if self.matches >= self.confirmations and not self.latched and now - self.last_hit >= self.cooldown:
            self.latched = True
            self.last_hit = now
            return True
        return False


def save_hit(picture, directory, face_boxes=()):
    """在原尺寸副本上标注当前命中的人脸，不修改采集和预览原图。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'vigil_hit_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}.jpg'
    screenshot = picture.convert('RGB').copy()
    draw = ImageDraw.Draw(screenshot)
    line_width = max(3, min(8, round(min(screenshot.size)/200)))
    for x, y, width, height in face_boxes:
        left, top = max(0, int(x)), max(0, int(y))
        right = min(screenshot.width-1, int(x+width)-1)
        bottom = min(screenshot.height-1, int(y+height)-1)
        if right > left and bottom > top:
            draw.rectangle((left,top,right,bottom),outline=(255,0,0),width=line_width)
    screenshot.save(path, quality=95)
    return path


def validate_target(classes, target):
    if target not in classes:
        raise ValueError(f'模型不包含类别 {target!r}，实际类别：{classes}')


def is_front_facing(landmarks):
    """用双眼和鼻尖的相对位置近似判断正脸，不判断眼球视线。"""
    right_eye, left_eye, nose = landmarks[:3]
    distance = math.dist(right_eye, left_eye)
    if distance < 1:
        return False
    middle_x = (right_eye[0] + left_eye[0]) / 2
    return (abs(right_eye[1] - left_eye[1]) / distance <= 0.3
            and abs(nose[0] - middle_x) / distance <= 0.25
            and nose[1] > min(right_eye[1], left_eye[1]))


class FaceFinder:
    def __init__(self, model_path):
        import cv2
        self.detector = cv2.FaceDetectorYN.create(str(model_path), '', (320, 320), 0.8, 0.3, 5000)

    def _detect_region(self, picture):
        # 小裁剪图也放到检测器适合的输入尺寸；坐标始终映射回原图。
        if max(picture.size) < 320:
            scale = 320/max(picture.size)
            size = (max(1,round(picture.width*scale)),max(1,round(picture.height*scale)))
            rows = self._detect_scaled(picture.resize(size,Image.Resampling.LANCZOS))
            sx,sy = size[0]/picture.width,size[1]/picture.height
            for face in rows:
                face[[0,2]] /= sx
                face[[1,3]] /= sy
                face[4:14:2] /= sx
                face[5:14:2] /= sy
            return rows
        return self._detect_scaled(picture)

    def _detect_scaled(self, picture):
        import cv2
        preview = picture.copy()
        preview.thumbnail((320, 320))
        canvas = Image.new('RGB', (320, 320))
        offset_x, offset_y = (320-preview.width)//2, (320-preview.height)//2
        canvas.paste(preview, (offset_x, offset_y))
        frame = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        scale_x, scale_y = picture.width/preview.width, picture.height/preview.height
        boxes = []
        for face in faces:
            x, y, width, height = face[:4]
            left = max(0, min(picture.width, int((x-offset_x)*scale_x)))
            top = max(0, min(picture.height, int((y-offset_y)*scale_y)))
            right = max(0, min(picture.width, int((x+width-offset_x)*scale_x)))
            bottom = max(0, min(picture.height, int((y+height-offset_y)*scale_y)))
            if right > left and bottom > top:
                mapped = face.copy()
                mapped[:4] = (left, top, right-left, bottom-top)
                mapped[4:14:2] = (face[4:14:2]-offset_x)*scale_x
                mapped[5:14:2] = (face[5:14:2]-offset_y)*scale_y
                boxes.append(mapped)
        return boxes

    def detect(self, picture, scan_grid=1, front_only=False):
        faces = self._detect_region(picture)
        # 在大画面中补充重叠分区扫描，让远处的小人脸获得更高的检测分辨率。
        if scan_grid > 1 and max(picture.size) > 320:
            margin_x, margin_y = picture.width * 0.1, picture.height * 0.1
            for row in range(scan_grid):
                for column in range(scan_grid):
                    left = max(0, int(column * picture.width / scan_grid - margin_x))
                    top = max(0, int(row * picture.height / scan_grid - margin_y))
                    right = min(picture.width, int((column+1) * picture.width / scan_grid + margin_x))
                    bottom = min(picture.height, int((row+1) * picture.height / scan_grid + margin_y))
                    for face in self._detect_region(picture.crop((left, top, right, bottom))):
                        # 边缘分区可能把半张脸预测为整脸，甚至将关键点放到区外。
                        # 保留全图结果，分区补检只接受关键点在实际采样范围内的脸。
                        points = face[4:14].reshape(5,2)
                        if (not np.isfinite(points).all() or (points < 0).any()
                                or (points[:,0] >= right-left).any()
                                or (points[:,1] >= bottom-top).any()):
                            continue
                        face = face.copy()
                        face[0] += left
                        face[1] += top
                        face[4:14:2] += left
                        face[5:14:2] += top
                        faces.append(face)
        faces = deduplicate_faces(faces)
        return [face for face in faces if is_front_facing(face[4:14].reshape(5, 2))] if front_only else faces

    def find(self, picture):
        return [tuple(int(value) for value in row[:4]) for row in self.detect(picture, front_only=True)]


def deduplicate_faces(faces, overlap=0.35):
    """合并重复框；分区边缘截断框可通过双眼、鼻尖位置确认是同一脸。"""
    kept = []
    for face in sorted(faces, key=lambda row: float(row[14]), reverse=True):
        x, y, width, height = face[:4]
        duplicate = False
        for previous in kept:
            px, py, pw, ph = previous[:4]
            intersection = max(0, min(x+width, px+pw)-max(x, px)) * max(0, min(y+height, py+ph)-max(y, py))
            union = width*height + pw*ph - intersection
            same_landmarks = False
            if intersection > 0:
                anchors = np.asarray(face[4:10]).reshape(3,2)
                previous_anchors = np.asarray(previous[4:10]).reshape(3,2)
                eye_span = min(np.linalg.norm(anchors[0]-anchors[1]),
                               np.linalg.norm(previous_anchors[0]-previous_anchors[1]))
                if eye_span >= 1:
                    same_landmarks = bool(np.max(np.linalg.norm(anchors-previous_anchors,axis=1)) <= .25*eye_span)
            if (union > 0 and intersection/union > overlap) or same_landmarks:
                duplicate = True
                break
        if not duplicate:
            kept.append(face)
    return kept


def recognize_faces(picture,faces,verifier=None,target=None,gaze=None,identity_ready=None,
                    excluded=(),cancelled=None):
    """先确认身份及过滤范围，再对候选人检查视线；切换时及时放弃旧任务。"""
    import cv2
    excluded = frozenset(excluded)
    if (target is not None or excluded) and verifier is None:
        raise ValueError('指定身份或过滤身份需要有效身份模型。')
    results = []
    for face in faces:
        if cancelled is not None and cancelled():
            return []
        x,y,width,height = (int(value) for value in face[:4])
        if target is None and not excluded:
            label,similarity = '人脸',0.0
        else:
            try:
                label,similarity = verifier.identify(picture,face)
            except (ValueError,RuntimeError,cv2.error):
                label,similarity = 'unknown',0.0
            if label not in verifier.gallery:
                label = 'unknown'
        facing = is_front_facing(face[4:14].reshape(5,2))
        candidate = (target is None or label==target) and label not in excluded
        reason = '正在判断视线' if facing else '脸部朝向未确认'
        eye = GazeResult(False,'已过滤 · 不检查视线' if label in excluded else reason if candidate else '非指定身份')
        results.append((x,y,width,height,label,0.0,False,similarity,facing,eye))
    if identity_ready is not None:
        identity_ready(picture,list(results))
    for index,(face,row) in enumerate(zip(faces,results)):
        if cancelled is not None and cancelled():
            return []
        if row[4] in excluded or (target is not None and row[4]!=target) or not row[8]:
            continue
        try:
            eye = gaze.measure(picture,face) if gaze is not None else GazeResult(False,'视线检测未就绪')
        except (ValueError,RuntimeError,cv2.error):
            eye = GazeResult(False,'双眼或虹膜未确认')
        results[index] = row[:6]+(bool(eye.looking),)+row[7:9]+(eye,)
    return results


def face_color(label, matched=False):
    if matched:
        return '#ff5555'
    if label == 'unknown':
        return '#a0a0a0'
    if label == 'me':
        return '#22ff88'
    return '#ff5555' if matched else '#5599ff'


class ChiCamera:
    def __init__(self, root, args, embedded=False, autostart=False, on_close=None):
        self.root, self.args = root, args
        self._scope = (0,args.target)
        self._transition_lock = threading.Lock()
        self.switching = False
        self._pending_switch = None
        self._excluded = frozenset()
        self._pending_excluded = None
        self._awaiting_scope_frame = False
        self._switch_preview_sequence = None
        self.embedded = embedded
        self.on_close = on_close or self.close
        self.closed = False
        self.want_running = False
        self.active_camera = None
        self.stop_event = threading.Event()
        self.calibration_request = threading.Event()
        self.calibration_kind = 'camera'
        self.refresh_request = threading.Event()
        self.refreshing = True
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.menu_indicator = None
        if sys.platform == 'darwin':
            from menu_bar import MenuBarIndicator
            self.menu_indicator = MenuBarIndicator(root,args.target)
        if not embedded:
            root.title('VigilScreen · 实时监测')
            window_width = min(1200,root.winfo_screenwidth()-80)
            window_height = min(800,root.winfo_screenheight()-100)
            root.geometry(f'{window_width}x{window_height}')
            root.minsize(min(940,window_width),min(650,window_height))
        root.columnconfigure(0,weight=1)
        root.rowconfigure(1,weight=1)
        style = ttk.Style(root)
        style.configure('Camera.Title.TLabel',font=('',18,'bold'))
        header = ttk.Frame(root,padding=(16,12))
        header.grid(row=0,column=0,sticky='ew')
        ttk.Label(header,text='VigilScreen · 实时监测',style='Camera.Title.TLabel').pack(side='left')
        ttk.Label(header,text='单次确认 · 菜单栏提示 · 红框截图').pack(side='right')
        body = ttk.Frame(root,padding=(14,0,14,14))
        body.grid(row=1,column=0,sticky='nsew')
        body.columnconfigure(0,weight=1)
        body.rowconfigure(0,weight=1)
        video = ttk.Frame(body)
        video.grid(row=0,column=0,sticky='nsew',padx=(0,14))
        video.columnconfigure(0,weight=1)
        video.rowconfigure(0,weight=1)
        self.preview = tk.Canvas(video,bg='#171c24',highlightthickness=0,width=480,height=320)
        self.preview.grid(row=0,column=0,sticky='nsew')
        self.last_frame = None
        self.preview.bind('<Configure>',self.resize_preview)
        self.status = tk.StringVar(value='尚未开始监控 · 点击“开始监控”后才打开摄像头')
        self.status_label = ttk.Label(video,textvariable=self.status,wraplength=600)
        self.status_label.grid(row=1,column=0,sticky='ew',pady=(10,3))
        self.performance = tk.StringVar(value='正在等待实时处理统计…')
        self.performance_label = ttk.Label(video,textvariable=self.performance,wraplength=600)
        self.performance_label.grid(row=2,column=0,sticky='ew')
        self.legend_label = ttk.Label(video,text='红：满足命中条件   蓝：已识别   绿：me   灰：未确认',wraplength=600)
        self.legend_label.grid(row=3,column=0,sticky='ew',pady=(6,0))
        sidebar = ttk.Frame(body,width=292)
        sidebar.grid(row=0,column=1,sticky='ns')
        sidebar.columnconfigure(0,weight=1)
        sidebar.rowconfigure(6,weight=1)
        target_box = ttk.LabelFrame(sidebar,text='命中目标',padding=12)
        target_box.grid(row=0,column=0,sticky='ew',pady=(0,12))
        self.available_targets = (ALL_PEOPLE,)
        self.target_var = tk.StringVar(value=args.target or ALL_PEOPLE)
        self.target_choice = ttk.Combobox(target_box,textvariable=self.target_var,state='disabled',width=24)
        self.target_choice.pack(fill='x')
        self.target_choice.bind('<<ComboboxSelected>>',self.change_target)
        self.switch_status = tk.StringVar(value='检测未开始')
        ttk.Label(target_box,textvariable=self.switch_status,wraplength=264).pack(fill='x',pady=(6,0))
        ttk.Label(target_box,text='默认对所有人检测注视；选择身份后，仅该身份可触发。' ,wraplength=264).pack(anchor='w',pady=(8,0))
        target_actions = ttk.Frame(target_box)
        target_actions.pack(fill='x',pady=(6,0))
        self.filter_button = ttk.Button(target_actions,text='过滤目标（0）',command=self.choose_filters)
        self.filter_button.pack(side='left',fill='x',expand=True,padx=(0,6))
        self.filter_summary = tk.StringVar(value='未排除任何身份')
        ttk.Label(target_box,textvariable=self.filter_summary,wraplength=264).pack(fill='x',pady=(4,0))
        gaze_box = ttk.LabelFrame(sidebar,text='视线校准',padding=12)
        gaze_box.grid(row=1,column=0,sticky='ew',pady=(0,12))
        self.gaze_status = tk.StringVar(value='镜头采用默认参考。看屏幕请先校准屏幕视线。')
        ttk.Label(gaze_box,textvariable=self.gaze_status,wraplength=264).pack(fill='x',pady=(0,8))
        ttk.Button(gaze_box,text='校准镜头视线 · 5 帧',command=self.request_calibration).pack(fill='x',pady=3)
        ttk.Button(gaze_box,text='校准屏幕视线 · 5 帧',command=lambda:self.request_calibration('screen')).pack(fill='x',pady=3)
        saved_box = ttk.LabelFrame(sidebar,text='最近截图',padding=12)
        saved_box.grid(row=2,column=0,sticky='ew',pady=(0,12))
        self.last_saved = tk.StringVar(value=f'截图目录：{args.output_dir}')
        self.saved_summary = tk.StringVar(value='尚无命中截图')
        saved_text = ttk.Label(saved_box,textvariable=self.saved_summary,wraplength=264)
        saved_text.pack(fill='x')
        ttk.Button(saved_box,text='复制保存信息',command=self.copy_saved_info).pack(anchor='w',pady=(6,0))
        self.model_info = tk.StringVar(value=f'身份模型入口：{args.profile}\n全员模式无需身份模型。')
        self.refresh_button = ttk.Button(sidebar,text='刷新模型 / 身份样本',command=self.request_refresh,state='disabled')
        self.refresh_button.grid(row=4,column=0,sticky='ew',pady=(0,6))
        ttk.Button(sidebar,text='查看模型与样本详情',command=self.show_model_details).grid(row=5,column=0,sticky='ew')
        ttk.Button(sidebar,text='退出并关闭摄像头',command=self.on_close).grid(row=7,column=0,sticky='ew')
        self.start_button = ttk.Button(sidebar,text='开始监控',command=self.toggle_monitor)
        self.start_button.grid(row=3,column=0,sticky='ew',pady=(0,6))
        ttk.Button(target_actions,text='加载身份模型',command=self.choose_profile).pack(side='right')
        if not embedded:
            root.protocol('WM_DELETE_WINDOW',self.close)
        self.worker = threading.Thread(target=self.run,daemon=True)
        self.stop_event.set()
        self.reload_targets()
        self.refreshing = False
        if autostart:
            self.start_monitor()
        else:
            self.target_choice.configure(state='readonly')
            self.refresh_button.configure(state='normal')
            if self.menu_indicator is not None:
                self.menu_indicator.pause(args.target)
        self.poll_timer = root.after(50, self.poll)

    def request_calibration(self, kind='camera'):
        if self.args.target is not None and self.args.target in getattr(self,'_excluded',frozenset()):
            self.gaze_status.set('当前目标已被过滤，请取消过滤后再校准视线。')
            return
        if getattr(self,'switching',False):
            self.gaze_status.set('正在切换目标，请等待检测恢复后再校准。')
            return
        if not self.want_running:
            self.gaze_status.set('请先开始监控，再校准视线。')
            return
        if self.refreshing or self.calibration_request.is_set():
            return
        self.calibration_kind = kind
        location = '屏幕' if kind == 'screen' else '摄像头镜头'
        self.gaze_status.set(f'请让单个 {self.args.target or '待校准人'} 睁眼注视{location}，正在收集 5 帧；校准期间不触发命中。')
        self.calibration_request.set()

    def reload_targets(self):
        from identity_training import available_identities
        self.available_targets = (ALL_PEOPLE,*available_identities(self.args.profile))
        self.target_choice.configure(values=self.available_targets,state='readonly')
        self.refresh_button.configure(state='normal')
        self.target_var.set(self.args.target or ALL_PEOPLE)

    def choose_profile(self):
        path = filedialog.askopenfilename(title='加载 VigilScreen 身份模型',filetypes=[('身份模型','*.json')])
        if path:
            self.set_profile(Path(path))

    def set_profile(self,path):
        from identity_training import read_profile
        try:
            _,metadata = read_profile(path)
        except (OSError,ValueError,KeyError) as error:
            from tkinter import messagebox
            messagebox.showerror('模型加载失败',str(error))
            return
        self.args.profile = Path(path).resolve()
        if self.args.target is not None and self.args.target not in metadata['classes']:
            self.pause_monitor()
            self.reload_targets()
            self.status.set('原指定身份不在新模型中，监测已暂停；请重新选择身份或明确选择全员模式。')
            return
        self.reload_targets()
        if self.want_running:
            self.request_refresh()
        else:
            self.model_info.set(f'已选择身份模型：{path}\n身份：{metadata["classes"]}')

    def choose_filters(self):
        if self.refreshing or self.switching:
            self.status.set('正在切换或加载，请完成后再设置过滤目标。')
            return
        dialog = tk.Toplevel(self.root)
        dialog.title('过滤目标')
        dialog.transient(self.root.winfo_toplevel())
        dialog.grab_set()
        body = ttk.Frame(dialog,padding=16)
        body.pack(fill='both',expand=True)
        ttk.Label(body,text='勾选要排除的身份（可多选）\n排除后不检查视线，不触发 Peek 或截图。').pack(anchor='w',pady=(0,10))
        names = sorted(set(self.available_targets[1:])|set(self._excluded))
        variables = {name:tk.BooleanVar(value=name in self._excluded) for name in names}
        if not names:
            ttk.Label(body,text='暂无已登记身份，请先训练或加载身份模型。').pack(anchor='w')
        for name,variable in variables.items():
            ttk.Checkbutton(body,text=name,variable=variable).pack(anchor='w')
        def apply():
            selected = frozenset(name for name,var in variables.items() if var.get())
            self.change_target(excluded=selected)
            dialog.destroy()
        ttk.Button(body,text='应用过滤',command=apply).pack(fill='x',pady=(12,0))
        ttk.Button(body,text='取消',command=dialog.destroy).pack(fill='x',pady=(6,0))

    def update_filter_summary(self):
        if not hasattr(self,'filter_button'):
            return
        excluded = getattr(self,'_excluded',frozenset())
        self.filter_button.configure(text=f'过滤目标（{len(excluded)}）')
        self.filter_summary.set('已排除：'+ '、'.join(sorted(excluded)) if excluded else '未排除任何身份')

    def change_target(self,event=None,*,excluded=None):
        selection = self.target_var.get()
        if self.refreshing or getattr(self,'switching',False) or selection not in self.available_targets:
            pending = getattr(self,'_pending_switch',None)
            self.target_var.set((pending[1] if pending else self.args.target) or ALL_PEOPLE)
            return
        target = None if selection==ALL_PEOPLE else selection
        new_excluded = getattr(self,'_excluded',frozenset()) if excluded is None else frozenset(excluded)
        if target==self.args.target and new_excluded==getattr(self,'_excluded',frozenset()):
            return
        if self.want_running and hasattr(self,'worker') and not self.worker.is_alive():
            self.want_running = False
            self.start_button.configure(text='开始监控')
        with self.transition_lock():
            revision = self.current_scope()[0]+1
            if self.want_running:
                self.switching = True
                # 立即作废旧帧，但等工作线程停止旧检测后才应用新目标。
                self._scope = (revision,self.args.target)
            else:
                self.args.target = target
                self._excluded = new_excluded
                self._scope = (revision,target)
        self.calibration_request.clear()
        self.events = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self._switch_preview_sequence = None
        self._awaiting_scope_frame = self.want_running
        if self.menu_indicator is not None:
            self.menu_indicator.pause(self.args.target)
        if self.want_running:
            self.target_choice.configure(state='disabled')
            self.refresh_button.configure(state='disabled')
            if hasattr(self,'filter_button'):
                self.filter_button.configure(state='disabled')
            self.start_button.configure(text='取消切换并暂停')
            if self.last_frame is not None:
                self.last_frame = (self.last_frame[0],[])
                self.preview.delete('face_overlay')
            self.switch_status.set('① 正在停止检测…')
            self.status.set(f'正在停止旧目标检测，随后切换为 {target or ALL_PEOPLE}；切换期间不触发 Peek 或截图。')
            self.gaze_status.set('目标切换中；完成后请重新校准屏幕视线。')
            # 队列与界面准备好后再交给工作线程，避免快速切换先完成再被旧提示覆盖。
            with self.transition_lock():
                self._pending_excluded = new_excluded
                self._pending_switch = (revision,target)
        else:
            self.update_filter_summary()
            self.switch_status.set('检测已停止 · 目标已切换 · 待开始')
            self.status.set(f'已选择：{target or ALL_PEOPLE} · 点击开始监控')

    def transition_lock(self):
        if not hasattr(self,'_transition_lock'):
            self._transition_lock = threading.Lock()
        return self._transition_lock

    def cancel_target_switch(self):
        with self.transition_lock():
            self.switching = False
            self._pending_switch = None
            self._pending_excluded = None
            self._awaiting_scope_frame = False

    def finish_target_switch(self,scope,details):
        if scope!=self.current_scope() or self.stop_event.is_set() or self.refresh_request.is_set():
            return
        self.model_info.set(details)
        self.target_var.set(scope[1] or ALL_PEOPLE)
        self.target_choice.configure(state='readonly')
        self.refresh_button.configure(state='normal')
        if hasattr(self,'filter_button'):
            self.filter_button.configure(state='normal')
        self.update_filter_summary()
        self.start_button.configure(text='暂停监控')
        self.switch_status.set('① 已停止 → ② 已切换 → ③ 检测中')
        self.status.set(f'已切换为 {scope[1] or ALL_PEOPLE}，检测已恢复。')
        self.gaze_status.set('新目标已开始检测；屏幕视线请重新校准。')
        if self.menu_indicator is not None:
            self.menu_indicator.reset(scope[1])

    def current_scope(self):
        return getattr(self,'_scope',(0,self.args.target))

    def toggle_monitor(self):
        if self.want_running:
            self.pause_monitor()
        else:
            self.start_monitor()

    def start_monitor(self):
        if self.closed:
            return
        self.want_running = True
        if self.worker.is_alive():
            if self.stop_event.is_set():
                self.root.winfo_toplevel().after(100,self.restart_refresh_if_idle)
            return
        self.stop_event.clear()
        self.cancel_target_switch()
        self.refresh_request.clear()
        self.calibration_request.clear()
        self.events = queue.Queue()
        self.refreshing = True
        self.status.set('正在加载人脸与视线模型…')
        self.switch_status.set('正在启动检测…')
        self.gaze_status.set('镜头采用默认参考；重新开始后请重新校准屏幕视线。')
        if self.menu_indicator is not None:
            self.menu_indicator.reset(self.args.target)
        self.start_button.configure(text='暂停监控')
        self.target_choice.configure(state='disabled')
        if hasattr(self,'filter_button'):
            self.filter_button.configure(state='disabled')
        self.worker = threading.Thread(target=self.run,daemon=True)
        self.worker.start()

    def pause_monitor(self):
        self.want_running = False
        self.stop_event.set()
        self.cancel_target_switch()
        self.refresh_request.clear()
        self.calibration_request.clear()
        self.events = queue.Queue()
        if self.active_camera is not None:
            self.active_camera.stop()
        self.refreshing = False
        self.start_button.configure(text='开始监控')
        self.switch_status.set('检测已停止')
        self.target_choice.configure(state='readonly')
        self.refresh_button.configure(state='normal')
        if hasattr(self,'filter_button'):
            self.filter_button.configure(state='normal')
        self.update_filter_summary()
        self.target_var.set(self.args.target or ALL_PEOPLE)
        self.status.set('监测已暂停 · 摄像头释放后可在工作室采集训练图片')
        self.gaze_status.set('监测已暂停；重新开始后请重新校准屏幕视线。')
        if self.menu_indicator is not None:
            self.menu_indicator.pause(self.args.target)
        self.last_frame = None
        self.preview.delete('all')
        try:
            while True:self.frames.get_nowait()
        except queue.Empty:
            pass

    def request_refresh(self):
        self.cancel_target_switch()
        self.reload_targets()
        if not self.want_running:
            self.status.set('身份列表已更新 · 点击开始监控加载模型')
            return
        self.refreshing = True
        self.refresh_request.set()
        self.calibration_request.clear()
        self.events = queue.Queue()
        self.target_choice.configure(state='disabled')
        self.gaze_status.set('重新加载后请重新校准视线。')
        self.last_frame = None
        self.preview.delete('all')
        if self.menu_indicator is not None:
            self.menu_indicator.reset(self.args.target)
        self.root.winfo_toplevel().after(100,self.restart_refresh_if_idle)

    def restart_refresh_if_idle(self):
        if self.want_running and not self.closed and not self.worker.is_alive():
            self.start_monitor()
        elif self.want_running and self.stop_event.is_set() and not self.closed:
            self.root.winfo_toplevel().after(100,self.restart_refresh_if_idle)

    def run(self):
        while not self.stop_event.is_set():
            self.refresh_request.clear()
            self.run_session()
            if not self.refresh_request.is_set():
                break

    def run_session(self):
        from identity_training import available_identities,load_profile
        from latest_camera import LatestCamera
        from tracking import PersonHits
        events = self.events  # 暂停/刷新更换队列，旧会话不能再向新界面送命中。
        scope = self.current_scope()
        target = scope[1]
        excluded = getattr(self,'_excluded',frozenset())
        capture = gaze = None
        calibration,deadline = [],None
        try:
            detector = FaceFinder(self.args.face_model)
            verifier = None
            cached_identity = None
            classes = available_identities(self.args.profile)
            if target is not None or excluded:
                verifier,metadata = load_profile(self.args.profile,self.args.identity_model,
                                                self.args.identity_threshold,self.args.identity_margin)
                classes = metadata['classes']
                cached_identity = (verifier,metadata)
                if target is not None:
                    validate_target(classes,target)
                for name in excluded:
                    validate_target(classes,name)
                details = f'范围：{target or ALL_PEOPLE}\n排除：{sorted(excluded)}\n模型：{self.args.profile}\n有效模板：'+str({name:len(rows) for name,rows in verifier.gallery.items()})
            else:
                details = '全员模式：无需身份模型，不执行身份匹配。\n每张当前人脸独立检查头部朝向和双眼视线。'
            events.put(('targets',(ALL_PEOPLE,*classes)))
            events.put(('model',details))
            gaze = GazeEstimator(self.args.gaze_model,self.args.gaze_horizontal,self.args.gaze_vertical,self.args.gaze_min_eye_width)
            if self.stop_event.is_set() or self.refresh_request.is_set():
                return
            capture = LatestCamera(self.args.camera)
            self.active_camera = capture
            capture.start()
            events.put(('ready','正在打开摄像头，请允许系统摄像头权限…'))
            sequence = 0
            tracker = PersonHits(lambda:HitGate(self.args.confirmations,self.args.cooldown))
            while not self.stop_event.is_set() and not self.refresh_request.is_set():
                sample = capture.read(after=sequence)
                if sample is None:
                    break
                picture,captured_at,sequence = sample
                pending = getattr(self,'_pending_switch',None)
                if pending is not None:
                    # 已到帧边界：旧检测调用已经结束，现在才允许切换目标。
                    events = self.events
                    events.put(('switch_progress',pending))
                    revision,new_target = pending
                    new_excluded = getattr(self,'_pending_excluded',None)
                    if new_excluded is None:
                        new_excluded = excluded
                    try:
                        new_verifier = None
                        if new_target is not None or new_excluded:
                            if cached_identity is None:
                                cached_identity = load_profile(self.args.profile,self.args.identity_model,
                                                               self.args.identity_threshold,self.args.identity_margin)
                            new_verifier,metadata = cached_identity
                            classes = metadata['classes']
                            if new_target is not None:
                                validate_target(classes,new_target)
                            for name in new_excluded:
                                validate_target(classes,name)
                        gaze.reset_calibration()
                        calibration,deadline = [],None
                        tracker = PersonHits(lambda:HitGate(self.args.confirmations,self.args.cooldown))
                    except Exception:
                        if (getattr(self,'_pending_switch',None)!=pending or self.stop_event.is_set()
                                or self.refresh_request.is_set()):
                            continue
                        raise
                    with self.transition_lock():
                        if (getattr(self,'_pending_switch',None)!=pending or self.stop_event.is_set()
                                or self.refresh_request.is_set()):
                            continue
                        target = new_target
                        verifier = new_verifier
                        scope = (revision,target)
                        self.args.target = target
                        excluded = new_excluded
                        self._excluded = excluded
                        self._scope = scope
                        self._pending_switch = None
                        self._pending_excluded = None
                        self.switching = False
                    details = ('全员模式：检测停止后已切换为全员。' if verifier is None else
                               f'范围：{target or ALL_PEOPLE}\n排除：{sorted(excluded)}\n模型：{self.args.profile}\n有效模板：'+str({name:len(rows) for name,rows in verifier.gallery.items()}))
                    events.put(('switch_complete',(scope,details)))
                    # 不检测切换前取到的画面，下一轮使用最新帧重新开始。
                    continue
                current = self.current_scope()
                if current!=scope:
                    # 只允许通过停止/切换/恢复事务应用目标，未知版本不执行检测。
                    continue
                started = time.monotonic()
                faces = detector.detect(picture,scan_grid=self.args.scan_grid)
                rows = recognize_faces(picture,faces,verifier=verifier,target=target,
                                       excluded=excluded,cancelled=lambda:self.stop_event.is_set() or self.refresh_request.is_set() or self.current_scope()!=scope,
                                       gaze=gaze,identity_ready=lambda picture,rows,frame_scope=scope:self.publish_frame(picture,rows,scope=frame_scope))
                if self.stop_event.is_set() or self.refresh_request.is_set():
                    break
                if self.current_scope()!=scope or getattr(self,'switching',False):
                    continue
                finished = time.monotonic()
                events.put(('performance',f'推理 {1000*(finished-started):.0f} ms · 采集后延迟 {1000*(finished-captured_at):.0f} ms'))
                if self.calibration_request.is_set():
                    deadline = deadline or time.monotonic()+8
                    valid = [row[9] for row in rows if (target is None or row[4]==target) and row[8] and row[9].ratios is not None]
                    if len(valid)==1:
                        calibration.append(valid[0])
                    if len(calibration)>=5:
                        gaze.calibrate(calibration,kind=self.calibration_kind)
                        events.put(('gaze','校准完成，请移开视线后重新注视验证。'))
                        self.calibration_request.clear()
                    elif time.monotonic()>=deadline:
                        events.put(('gaze','校准失败：画面请仅保留一张清晰待校准人脸，睁眼注视对应位置。'))
                        self.calibration_request.clear()
                    rows = [row[:6]+(False,)+row[7:] for row in rows]
                    tracker.update(rows,time.monotonic())
                    if not self.calibration_request.is_set():
                        calibration,deadline = [],None
                    hits = []
                else:
                    hits,_ = tracker.update(rows,time.monotonic())
                with self.transition_lock():
                    if (hits and self.current_scope()==scope and not getattr(self,'switching',False)
                            and not self.stop_event.is_set() and not self.refresh_request.is_set()):
                        path = save_hit(picture,self.args.output_dir,face_boxes=[row[:4] for row in rows if row[6]])
                        events.put(('hit',(str(path),len(hits))))
                self.publish_frame(picture,rows,scope=scope)
        except Exception as error:
            if not self.stop_event.is_set():
                output = self.events if getattr(self,'switching',False) else events
                output.put(('error',str(error)))
        finally:
            if capture is not None:
                capture.close()
            self.active_camera = None
            if gaze is not None:
                gaze.close()

    def publish_frame(self, picture, results, scope=None):
        frames = self.frames
        if (self.stop_event.is_set() or self.refresh_request.is_set()
                or getattr(self,'switching',False)
                or (scope is not None and self.current_scope()!=scope)):
            return
        try:
            frames.get_nowait()
        except queue.Empty:
            pass
        frames.put_nowait((picture,results))

    def show_model_details(self):
        details = tk.Toplevel(self.root)
        details.title('模型与身份样本详情')
        details.geometry('760x360')
        text = tk.Text(details,wrap='char',padx=12,pady=12)
        scrollbar = ttk.Scrollbar(details,command=text.yview)
        scrollbar.pack(side='right',fill='y')
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(fill='both',expand=True)
        text.insert('1.0',self.model_info.get()+'\n\n'+self.last_saved.get())
        text.configure(state='disabled')

    def copy_saved_info(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_saved.get())

    def resize_preview(self, event=None):
        width = max(1,self.preview.winfo_width())
        self.status_label.configure(wraplength=width)
        self.performance_label.configure(wraplength=width)
        self.legend_label.configure(wraplength=width)
        if self.last_frame is not None and not self.refreshing:
            self.draw_frame(*self.last_frame)

    def draw_frame(self, picture, results):
        self.last_frame = (picture,results)
        width,height = picture.size
        canvas_width = max(1,self.preview.winfo_width())
        canvas_height = max(1,self.preview.winfo_height())
        scale = min(canvas_width/width,canvas_height/height)
        display_width,display_height = max(1,round(width*scale)),max(1,round(height*scale))
        preview = picture.resize((display_width,display_height),Image.Resampling.BILINEAR)
        sx,sy = display_width/width,display_height/height
        ox,oy = (canvas_width-display_width)/2,(canvas_height-display_height)/2
        photo = ImageTk.PhotoImage(preview)
        items = self.preview.find_withtag('camera_image')
        if items:
            self.preview.itemconfigure(items[0],image=photo)
            self.preview.coords(items[0],ox,oy)
        else:
            self.preview.create_image(ox,oy,image=photo,anchor='nw',tags='camera_image')
        self.photo = photo
        self.preview.delete('face_overlay')
        for number,(x,y,w,h,label,confidence,matched,similarity,facing,eye) in enumerate(results,1):
            filtered = label in getattr(self,'_excluded',frozenset())
            color = '#a0a0a0' if filtered else face_color(label,matched)
            left,top = ox+x*sx,oy+y*sy
            right,bottom = ox+(x+w)*sx,oy+(y+h)*sy
            self.preview.create_rectangle(left,top,right,bottom,outline=color,width=2,tags='face_overlay')
            title = f'#{number} 人脸' if label=='人脸' else (f'#{number} {label} · {similarity:.2f}' if label!='unknown' else f'#{number} 未确认')
            if self.args.target is None or label == self.args.target or filtered:
                title += f'\n{eye.reason}'
            # 深色标签底板提升明亮场景下的可读性；尽量放在人脸框上方。
            text = self.preview.create_text(0,0,anchor='nw',fill=color,font=('',11,'bold'),
                                            width=max(1,min(270,display_width-12)),text=title,tags='face_overlay')
            box = self.preview.bbox(text)
            text_width,text_height = box[2]-box[0],box[3]-box[1]
            tx = min(max(ox+6,left),max(ox+6,ox+display_width-text_width-6))
            ty = top-text_height-10
            if ty < oy+6:
                ty = min(bottom+6,max(oy+6,oy+display_height-text_height-6))
            self.preview.coords(text,tx,ty)
            box = self.preview.bbox(text)
            background = self.preview.create_rectangle(box[0]-4,box[1]-3,box[2]+4,box[3]+3,
                                                        fill='#171c24',outline='',tags='face_overlay')
            self.preview.tag_lower(background,text)
        if not getattr(self,'switching',False) and not getattr(self,'_awaiting_scope_frame',False):
            self.status.set(f'人脸 {len(results)} · 面向镜头 {sum(row[8] for row in results)} · ' + ' / '.join(
                row[4] if row[4] != 'unknown' else '未确认' for row in results) if results else '摄像头已打开 · 等待人脸进入画面')
            if self.switch_status.get()=='正在启动检测…':
                self.switch_status.set(f'检测中 · {self.args.target or ALL_PEOPLE}')

    def poll(self):
        if self.closed:
            return
        if (getattr(self,'switching',False) or getattr(self,'_awaiting_scope_frame',False)) and self.active_camera is not None:
            sample = self.active_camera.latest
            if isinstance(sample,tuple) and len(sample)==3 and sample[2]!=self._switch_preview_sequence:
                self._switch_preview_sequence = sample[2]
                self.draw_frame(sample[0],[])
        try:
            picture, results = self.frames.get_nowait()
            if not self.refreshing and not getattr(self,'switching',False):
                self._awaiting_scope_frame = False
                self.draw_frame(picture,results)
        except queue.Empty:
            pass
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'hit':
                    if not self.refreshing:
                        self.show_hit(*value)
                elif kind == 'gaze':
                    self.gaze_status.set(value)
                elif kind == 'performance':
                    self.performance.set(value)
                elif kind == 'model':
                    self.model_info.set(value)
                    self.refreshing = False
                    self.refresh_button.configure(state='normal')
                    self.target_choice.configure(state='readonly')
                    self.filter_button.configure(state='normal')
                    self.update_filter_summary()
                elif kind == 'switch_progress':
                    if getattr(self,'_pending_switch',None)==value:
                        self.switch_status.set('① 检测已停止 · ② 正在切换目标…')
                        self.status.set(f'检测已停止，正在加载并切换为 {value[1] or ALL_PEOPLE}；切换期间仅预览。')
                elif kind == 'switch_complete':
                    self.finish_target_switch(*value)
                elif kind == 'targets':
                    self.available_targets = value
                    self.target_choice.configure(values=value)
                    self.target_var.set(self.args.target or ALL_PEOPLE)
                elif kind == 'ready':
                    self.status.set(value)
                else:
                    was_switching = getattr(self,'switching',False)
                    self.cancel_target_switch()
                    self.want_running = False
                    self.refreshing = False
                    self.start_button.configure(text='开始监控')
                    self.target_choice.configure(state='readonly')
                    self.filter_button.configure(state='normal')
                    self.update_filter_summary()
                    self.target_var.set(self.args.target or ALL_PEOPLE)
                    self.switch_status.set('切换失败 · 检测未恢复' if was_switching else '检测已停止')
                    self.status.set(f'识别已停止：{value}')
                    self.last_saved.set('排查提示后可点击刷新重试。')
                    self.refresh_button.configure(state='normal')
        except queue.Empty:
            pass
        if not self.closed:
            self.poll_timer = self.root.after(20,self.poll)

    def show_hit(self,path,count=1):
        scope = self.args.target or '全部人脸'
        self.last_saved.set(f'{scope} · 新命中 {count} 人 · 截图：{path}')
        self.saved_summary.set(f'新命中 {count} 人\n{Path(path).name}')
        if self.menu_indicator is not None:
            self.menu_indicator.show_hit(self.args.target,path,count=count)
        self.root.bell()

    def shutdown(self):
        self.pause_monitor()
        self.closed = True
        timer = getattr(self,'poll_timer',None)
        if timer is not None:
            self.root.after_cancel(timer)
        if self.menu_indicator is not None:
            self.menu_indicator.close()

    def close(self):
        self.shutdown()
        def finish():
            if self.worker.is_alive():
                self.root.after(100,finish)
            else:
                self.root.destroy()
        finish()


def nonnegative_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('秒数必须为大于等于 0 的有限数。')
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,default=ROOT/'checkpoints/identities/active_profile.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'screenshots')
    parser.add_argument('--face-model', type=Path, default=RESOURCE_ROOT / 'face_models/yunet.onnx')
    parser.add_argument('--identity-model', type=Path, default=RESOURCE_ROOT / 'face_models/sface_int8.onnx')
    parser.add_argument('--identity-threshold', type=positive_float, default=0.5)
    parser.add_argument('--identity-margin', type=positive_float, default=0.06)
    parser.add_argument('--scan-grid', type=int, choices=[1, 2], default=2, help='2：全图加重叠分区，1：仅全图')
    parser.add_argument('--gaze-model', type=Path, default=RESOURCE_ROOT / 'face_models/face_landmarker.task')
    parser.add_argument('--gaze-horizontal', type=positive_float, default=.12)
    parser.add_argument('--gaze-vertical', type=positive_float, default=.25)
    parser.add_argument('--gaze-min-eye-width', type=positive_float, default=12)
    parser.add_argument('--target',default=None,help='可选身份；默认检测所有人')
    parser.add_argument('--camera', type=nonnegative_int, default=0)
    parser.add_argument('--confirmations', type=positive_int, default=1,help='连续确认次数，默认单次通过即命中')
    parser.add_argument('--cooldown', type=nonnegative_seconds, default=0.0,help='再次命中冷却秒数，默认 0；持续注视仍只提示一次')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if max(args.identity_threshold,args.identity_margin) > 1:
        parser.error('分类、身份相似度及身份间距阈值均不能大于 1。')
    if args.gaze_horizontal > .5 or args.gaze_vertical > .5:
        parser.error('视线偏移阈值不能大于 0.5。')
    args.profile = args.profile.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    root = tk.Tk()
    ChiCamera(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
