"""使用眼角、眼睑和虹膜相对位置估计是否注视镜头；不是精确眼动仪。"""
from dataclasses import dataclass
import numpy as np
import os
import tempfile
from pathlib import Path


@dataclass
class GazeResult:
    looking: bool
    reason: str
    ratios: tuple | None = None


def evaluate_eyes(points, image_size, reference=None, horizontal=0.12, vertical=0.25, min_eye_width=12):
    points = np.asarray(points, dtype=float)
    if points.shape != (478, 2) or not np.isfinite(points).all():
        return GazeResult(False, '眼部关键点不足')
    points = points * np.asarray(image_size)
    ratios = []
    # 两眼分别判断，避免把一只眼闭合或偏转误当作注视。
    for outer, inner, upper, lower, iris in [(33,133,159,145,468),(362,263,386,374,473)]:
        a, b = points[outer], points[inner]
        if a[0] > b[0]:
            a, b = b, a
        axis = b-a
        width = np.linalg.norm(axis)
        if width < min_eye_width:
            return GazeResult(False, '眼睛过小，请靠近镜头')
        axis = axis/width
        normal = np.array([-axis[1],axis[0]])
        height = float((points[lower]-points[upper]) @ normal)
        if height < 2 or height/width < 0.14:
            return GazeResult(False, '闭眼或眼睑不可见')
        x = float((points[iris]-a) @ axis / width)
        y = float((points[iris]-points[upper]) @ normal / height)
        if not (0 <= x <= 1 and -0.15 <= y <= 1.15):
            return GazeResult(False, '虹膜位置未确认')
        ratios.append((x,y))
    reference = reference if reference is not None else ((.5,.5),(.5,.5))
    looking = all(abs(x-rx) <= horizontal and abs(y-ry) <= vertical
                  for (x,y),(rx,ry) in zip(ratios,reference))
    return GazeResult(looking, '视线接近镜头' if looking else '视线偏离镜头', tuple(ratios))


class GazeEstimator:
    def __init__(self, model_path, horizontal=.12, vertical=.25, min_eye_width=12):
        from runtime_paths import data_root
        cache = data_root() / '.cache/matplotlib'
        try:
            cache.mkdir(parents=True,exist_ok=True)
        except OSError:
            cache = Path(tempfile.gettempdir())/'image-classification-matplotlib'
            cache.mkdir(parents=True,exist_ok=True)
        os.environ.setdefault('MPLCONFIGDIR',str(cache))
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.reference = None
        self.screen_reference = None
        self.horizontal, self.vertical, self.min_eye_width = horizontal, vertical, min_eye_width
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path),delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE, num_faces=2,
            min_face_detection_confidence=.6,min_face_presence_confidence=.6,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def measure(self, picture, face):
        x,y,w,h = [int(value) for value in face[:4]]
        left,top = max(0,x-int(w*.3)),max(0,y-int(h*.3))
        right,bottom = min(picture.width,x+int(w*1.3)),min(picture.height,y+int(h*1.3))
        if right <= left or bottom <= top:
            return GazeResult(False,'人脸区域无效')
        crop = picture.crop((left,top,right,bottom)).convert('RGB')
        original_size = crop.size
        # 虹膜网络有固定输入尺寸，大 ROI 先缩小以减少预处理成本。
        # 几何判断仍用原始尺寸，不能放大小眼睛来绕过清晰度检查。
        crop.thumbnail((512,512))
        result = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.asarray(crop)))
        candidates = []
        for landmarks in result.face_landmarks:
            points = np.array([(point.x,point.y) for point in landmarks])
            # 每张脸单独裁剪，并核对双眼位置，不能借用相邻人的视线。
            if len(points) != 478:
                continue
            eye_points = points[[33,133,362,263]] * np.array(original_size) + [left,top]
            center = eye_points.mean(axis=0)
            if not (x <= center[0] <= x+w and y <= center[1] <= y+h):
                continue
            expected = face[4:8].reshape(2,2).mean(axis=0)
            distance = np.linalg.norm(center-expected)
            if distance <= w*.25:
                candidates.append((distance,points))
        if not candidates:
            return GazeResult(False,'双眼或虹膜未确认')
        candidates.sort(key=lambda row:row[0])
        if len(candidates)>1:
            return GazeResult(False,'多人眼部关联不明确')
        eye = evaluate_eyes(candidates[0][1],original_size,self.reference,
                            self.horizontal,self.vertical,self.min_eye_width)
        screen_reference = getattr(self,'screen_reference',None)
        if not eye.looking and eye.ratios is not None and screen_reference is not None:
            screen_eye = evaluate_eyes(candidates[0][1],original_size,screen_reference,
                                       self.horizontal,self.vertical,self.min_eye_width)
            if screen_eye.looking:
                return GazeResult(True,'视线接近已校准屏幕',screen_eye.ratios)
        return eye

    def calibrate(self, measurements, kind='camera'):
        reference = np.median([row.ratios for row in measurements],axis=0)
        if kind == 'camera':
            self.reference = reference
        elif kind == 'screen':
            self.screen_reference = reference
        else:
            raise ValueError('不支持的视线校准类型。')

    def reset_calibration(self):
        """切换身份时只清除参考，不重建原生模型。"""
        self.reference = None
        self.screen_reference = None

    def close(self):
        self.landmarker.close()
