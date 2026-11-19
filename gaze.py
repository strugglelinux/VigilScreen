"""使用眼角、眼睑和虹膜相对位置估计是否注视镜头；不是精确眼动仪。"""
from dataclasses import dataclass
from PIL import Image
import numpy as np
import os
import tempfile
from pathlib import Path


@dataclass
class GazeResult:
    looking: bool
    reason: str
    ratios: tuple | None = None
    pose: tuple | None = None
    box: tuple | None = None


def head_pose(landmarks):
    """在双眼局部坐标系检查鼻、嘴位置，避免轻微歪头造成水平坐标误判。"""
    points=np.asarray(landmarks,dtype=float)
    if points.shape!=(5,2) or not np.isfinite(points).all():
        return None
    eyes=points[:2][np.argsort(points[:2,0])]
    axis=eyes[1]-eyes[0]
    distance=float(np.linalg.norm(axis))
    if distance<1:
        return None
    axis/=distance
    normal=np.array([-axis[1],axis[0]])
    middle=eyes.mean(axis=0)
    nose=(points[2]-middle)/distance
    mouth=(points[3:].mean(axis=0)-middle)/distance
    yaw,pitch=float(nose@axis),float(nose@normal)
    roll=float(np.arctan2(axis[1],axis[0]))
    mouth_width=float(np.linalg.norm(points[3]-points[4])/distance)
    if (abs(roll)>np.deg2rad(40) or abs(yaw)>.28 or not .1<=pitch<=1.2
            or float(mouth@normal)<=pitch+.08 or abs(float(mouth@axis))>.35
            or not .25<=mouth_width<=1.3):
        return None
    return yaw,pitch,roll


def matches_reference(ratios,reference,horizontal,vertical):
    errors=(np.asarray(ratios)-np.asarray(reference))/[horizontal,vertical]
    # 每只眼独立通过椭圆范围，同时限制双眼相对参考的分歧。
    return (bool(np.all(np.sum(errors*errors,axis=1)<=1+1e-9))
            and bool(np.all(np.abs(errors[0]-errors[1])<=1.25)))


def calibration_location_matches(box,anchor):
    if anchor is None:
        return True
    if box is None:
        return False
    box,anchor=np.asarray(box,dtype=float),np.asarray(anchor,dtype=float)
    if (box.shape!=(4,) or anchor.shape!=(4,) or not np.isfinite([box,anchor]).all()
            or np.any(box[2:]<=0) or np.any(anchor[2:]<=0)):
        return False
    movement=(box[:2]+box[2:]/2-anchor[:2]-anchor[2:]/2)/anchor[2:]
    return bool(np.all(np.abs(movement)<=.6) and np.all(box[2:]/anchor[2:]>=.65)
                and np.all(box[2:]/anchor[2:]<=1.5))


def closest_reference(ratios,references,horizontal,vertical):
    """在已校准的双眼四维特征凸包内插值，同一组权重用于两只眼，禁止外推。"""
    references=np.asarray(references,dtype=float)
    residual=((references-np.asarray(ratios))/[horizontal,vertical]).reshape(len(references),4)
    weights=np.zeros(len(references))
    weights[np.argmin(np.sum(residual*residual,axis=1))]=1
    for _ in range(32):
        current=weights@residual
        vertex=int(np.argmin(residual@current))
        direction=residual[vertex]-current
        length=float(direction@direction)
        if length<1e-12:
            break
        step=float(np.clip(-(current@direction)/length,0,1))
        if step<1e-8:
            break
        weights*=1-step
        weights[vertex]+=step
    return np.sum(references*weights[:,None,None],axis=0),weights


def evaluate_eyes(points, image_size, reference=None, horizontal=0.12, vertical=0.25,
                  min_eye_width=12,processing_size=None):
    points=np.asarray(points,dtype=float)
    if points.shape!=(478,2) or not np.isfinite(points).all():
        return GazeResult(False,'眼部关键点不足')
    if horizontal<=0 or vertical<=0:
        raise ValueError('视线阈值必须大于 0。')
    original=points
    points=points*np.asarray(image_size)
    ratios=[]
    definitions=[(33,133,[(160,144),(159,145),(158,153)],468),
                 (362,263,[(385,380),(386,374),(387,373)],473)]
    for outer,inner,lids,iris in definitions:
        a,b=points[outer],points[inner]
        if a[0]>b[0]:
            a,b=b,a
        width=float(np.linalg.norm(b-a))
        processed_width=(np.linalg.norm((original[outer]-original[inner])*processing_size)
                         if processing_size is not None else width)
        if width<min_eye_width or processed_width<8:
            return GazeResult(False,'距离过远或眼部像素不足，视线无法确认，请靠近镜头')
        axis=(b-a)/width
        normal=np.array([-axis[1],axis[0]])
        upper=np.array([points[u] for u,_ in lids])
        lower=np.array([points[l] for _,l in lids])
        heights=(lower-upper)@normal
        height=float(np.median(heights))
        if height<2 or height/width<.12 or np.count_nonzero(heights>0)<2:
            return GazeResult(False,'闭眼或眼睑不可见')
        ring=points[iris+1:iris+5]
        center=np.median(ring,axis=0)
        radius=np.linalg.norm(ring-center,axis=1)
        if (np.linalg.norm(center-points[iris])>width*.1
                or np.min(radius)<max(.5,width*.025) or np.max(radius)>width*.35):
            return GazeResult(False,'虹膜轮廓不稳定')
        # 中心与环的中心共同估计，避免单个虹膜点抖动。
        center=(center+points[iris])/2
        x=float((center-a)@axis/width)
        eyelid_middle=float(np.median(((upper+lower)/2-a)@normal))
        offset=float((center-a)@normal-eyelid_middle)
        # 竖直归一化固定使用眼宽的 1/4，避免眼睑高度随眨眼变化放大误差。
        y=.5+offset/(width*.25)
        if not (.05<=x<=.95) or abs(offset)>height*.75:
            return GazeResult(False,'虹膜位置未确认')
        ratios.append((x,y))
    reference=reference if reference is not None else ((.5,.5),(.5,.5))
    looking=matches_reference(ratios,reference,horizontal,vertical)
    return GazeResult(looking,'视线接近镜头' if looking else '视线偏离参考或双眼方向不一致',tuple(ratios))


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
        self.reference_pose = None
        self.reference_box = None
        self.screen_box = None
        self.screen_samples = []
        self.screen_poses = []
        self.horizontal, self.vertical, self.min_eye_width = horizontal, vertical, min_eye_width
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path),delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE, num_faces=2,
            min_face_detection_confidence=.6,min_face_presence_confidence=.6,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def measure(self, picture, face):
        pose=head_pose(np.asarray(face[4:14]).reshape(5,2))
        if pose is None:
            return GazeResult(False,'脸部朝向未确认')
        x,y,w,h = [int(value) for value in face[:4]]
        left,top = max(0,x-int(w*.3)),max(0,y-int(h*.3))
        right,bottom = min(picture.width,x+int(w*1.3)),min(picture.height,y+int(h*1.3))
        if right <= left or bottom <= top:
            return GazeResult(False,'人脸区域无效')
        crop = picture.crop((left,top,right,bottom)).convert('RGB')
        original_size = crop.size
        # 放大目标区域辅助关键点检测；清晰度检查仍使用原始像素。
        scale = 512 / max(original_size)
        size = tuple(max(1,round(value*scale)) for value in original_size)
        crop = crop.resize(size, Image.Resampling.LANCZOS)
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
            expected=np.asarray(face[4:8]).reshape(2,2)
            observed=np.array([eye_points[:2].mean(axis=0),eye_points[2:].mean(axis=0)])
            expected=expected[np.argsort(expected[:,0])]
            observed=observed[np.argsort(observed[:,0])]
            distance=float(np.linalg.norm(center-expected.mean(axis=0)))
            separation=float(np.linalg.norm(expected[0]-expected[1]))
            if (separation>=1 and distance<=w*.25
                    and np.max(np.linalg.norm(observed-expected,axis=1))<=w*.22
                    and .6<=np.linalg.norm(observed[0]-observed[1])/separation<=1.6):
                candidates.append((distance,points))
        if not candidates:
            return GazeResult(False,'双眼或虹膜未确认')
        if len(candidates)>1:
            return GazeResult(False,'多人眼部关联不明确')
        box=(x/picture.width,y/picture.height,w/picture.width,h/picture.height)
        camera_location_ok=calibration_location_matches(box,getattr(self,'reference_box',None))
        eye=evaluate_eyes(candidates[0][1],original_size,self.reference if camera_location_ok else None,
                          self.horizontal,self.vertical,self.min_eye_width,processing_size=crop.size)
        eye.pose=pose
        eye.box=box
        if eye.ratios is None:
            return eye
        reference_pose=getattr(self,'reference_pose',None) if camera_location_ok else None
        camera_pose_ok=(abs(pose[0])<=.18 if reference_pose is None else
                        np.all(np.abs(np.asarray(pose[:2])-reference_pose[:2])<=[.16,.25]))
        if eye.looking and camera_pose_ok:
            return eye
        eye.looking=False
        eye.reason='视线偏离镜头 / 屏幕参考' if camera_pose_ok else '头部朝向与镜头参考不一致'
        screen_reference=getattr(self,'screen_reference',None)
        if screen_reference is not None and calibration_location_matches(box,getattr(self,'screen_box',None)):
            references=getattr(self,'screen_samples',None) or [screen_reference]
            horizontal,vertical=self.horizontal,self.vertical
            poses=getattr(self,'screen_poses',[])
            # 保留已有位置的接受范围；新增位置不能让原有中心参考失效。
            for index,reference in enumerate(references):
                pose_ok=(index>=len(poses) or poses[index] is None or
                         np.all(np.abs(np.asarray(pose[:2])-np.asarray(poses[index])[:2])<=[.18,.25]))
                if pose_ok and matches_reference(eye.ratios,reference,horizontal,vertical):
                    return GazeResult(True,'视线接近已校准屏幕',eye.ratios,eye.pose,eye.box)
            reference,weights=closest_reference(eye.ratios,references,horizontal,vertical)
            pose_ok=True
            if len(poses)==len(references) and all(item is not None for item in poses):
                expected_pose=weights@np.asarray(poses)[:,:2]
                pose_ok=bool(np.all(np.abs(np.asarray(pose[:2])-expected_pose)<=[.18,.25]))
            if pose_ok and matches_reference(eye.ratios,reference,horizontal,vertical):
                return GazeResult(True,'视线接近已校准屏幕',eye.ratios,eye.pose,eye.box)
        return eye

    def calibrate(self, measurements, kind='camera'):
        if kind not in {'camera','screen','screen_extend'}:
            raise ValueError('不支持的视线校准类型。')
        if len(measurements)<5:
            raise ValueError('至少需要 5 帧清晰、稳定的双眼采样。')
        if any(row.ratios is None for row in measurements):
            raise ValueError('采样中有闭眼或无法确认的眼睛。')
        samples=np.asarray([row.ratios for row in measurements],dtype=float)
        if samples.shape!=(len(measurements),2,2) or not np.isfinite(samples).all():
            raise ValueError('视线采样无效。')
        reference=np.median(samples,axis=0)
        residual=(samples-reference)/[self.horizontal*.6,self.vertical*.6]
        stable=np.max(np.abs(residual),axis=(1,2))<=1
        if np.count_nonzero(stable)<max(4,int(np.ceil(len(samples)*.8))):
            raise ValueError('视线变化过大，请保持注视同一位置。')
        reference=np.median(samples[stable],axis=0)
        if np.any(reference[:,0]<.1) or np.any(reference[:,0]>.9) or np.any(np.abs(reference[:,1]-.5)>1):
            raise ValueError('视线偏转过大，请调整摄像头与屏幕位置。')
        poses=[row.pose for row in measurements]
        pose=None
        if all(item is not None for item in poses):
            poses=np.asarray(poses,dtype=float)
            if poses.shape!=(len(measurements),3) or not np.isfinite(poses).all():
                raise ValueError('头部姿态采样无效。')
            pose=np.median(poses[stable],axis=0)
            if np.any(np.max(np.abs(poses[stable]-pose),axis=0)>[.08,.12,.25]):
                raise ValueError('头部移动过大，请保持姿态稳定。')
        box=None
        boxes=[row.box for row in measurements]
        if all(item is not None for item in boxes):
            boxes=np.asarray(boxes,dtype=float)
            if boxes.shape!=(len(measurements),4) or not np.isfinite(boxes).all():
                raise ValueError('人脸位置采样无效。')
            centers=boxes[:,:2]+boxes[:,2:]/2
            size=np.median(boxes[:,2:],axis=0)
            if np.any(size<=0) or np.any(np.ptp(centers,axis=0)>size*.3) or np.any(np.ptp(boxes[:,2:],axis=0)>size*.35):
                raise ValueError('采样人脸移动或变化，请保持同一个人在画面中。')
            box=tuple(np.median(boxes,axis=0))
        if kind=='camera':
            self.reference=reference
            self.reference_pose=pose
            self.reference_box=box
        elif kind=='screen':
            self.screen_reference=reference
            self.screen_samples=[reference]
            self.screen_poses=[pose]
            self.screen_box=box
        else:
            if getattr(self,'screen_reference',None) is None:
                raise ValueError('请先校准屏幕中心，再扩展屏幕边角。')
            if not calibration_location_matches(box,getattr(self,'screen_box',None)):
                raise ValueError('人脸位置或距离已变化，请重新校准屏幕中心。')
            references=list(getattr(self,'screen_samples',None) or [self.screen_reference])
            poses=list(getattr(self,'screen_poses',None) or [None])
            if min(np.linalg.norm((reference-item)/[self.horizontal,self.vertical]) for item in references)<.2:
                return
            references.append(reference);poses.append(pose)
            # 最多中心加八个边界位置，限制实时插值开销。
            self.screen_samples=references[:1]+references[-8:] if len(references)>9 else references
            self.screen_poses=poses[:1]+poses[-8:] if len(poses)>9 else poses

    def reset_calibration(self):
        """切换身份时只清除参考，不重建原生模型。"""
        self.reference = None
        self.screen_reference = None
        self.reference_pose = None
        self.reference_box = None
        self.screen_box = None
        self.screen_samples = []
        self.screen_poses = []

    def close(self):
        self.landmarker.close()
