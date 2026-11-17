"""后台读取摄像头；界面线程只接收最新帧。"""
import queue
import threading
from uuid import uuid4
from pathlib import Path

from PIL import Image
from labeling import LabelStore, selection_box, prepare_capture


class CameraStream:
    def __init__(self, index=0):
        self.index = index
        self.events = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def publish(self, kind, value):
        try:
            self.events.get_nowait()
        except queue.Empty:
            pass
        self.events.put_nowait((kind, value))

    def run(self):
        capture = None
        try:
            import cv2
            capture = cv2.VideoCapture(self.index)
            if not capture.isOpened():
                raise RuntimeError('无法打开摄像头，请检查设备编号、占用情况和系统摄像头权限。')
            while not self.stopped.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('摄像头读取失败，请重新打开摄像头。')
                picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.publish('frame', picture)
                self.stopped.wait(0.03)
        except ImportError:
            self.publish('error', '缺少 OpenCV，请运行 python -m pip install opencv-python。')
        except Exception as error:
            self.publish('error', str(error))
        finally:
            if capture is not None:
                capture.release()

    def stop(self):
        self.stopped.set()


def save_capture(picture: Image.Image, output: Path, category: str, box=None, max_side=0) -> Path:
    """裁剪并缩小采集图，再保存至归档和类别数据集。"""
    picture = prepare_capture(picture, box, max_side)
    store = LabelStore(output)
    raw_dir = store.root.parent / (store.root.name + '_captures')
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = raw_dir / (uuid4().hex + '.png')
    picture.save(source)
    try:
        return store.save(source, category)
    except Exception:
        source.unlink(missing_ok=True)
        raise
