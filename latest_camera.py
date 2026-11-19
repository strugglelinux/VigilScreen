"""采集与推理解耦，仅保留最近一帧，摄像头由采集线程持有和释放。"""
import threading
import time

from PIL import Image


class LatestCamera:
    def __init__(self, index, width=1920, height=1080):
        self.index = index
        self.width, self.height = width, height
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.latest = None
        self.sequence = 0
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def publish(self, picture, captured_at):
        with self.condition:
            self.sequence += 1
            self.latest = (picture, captured_at, self.sequence)
            self.condition.notify_all()

    def read(self, after=0, timeout=5):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.error is not None or self.stopped.is_set() or self.sequence > after,
                timeout=timeout)
            if self.error is not None:
                raise RuntimeError(self.error)
            if self.stopped.is_set():
                return None
            if not ready:
                raise RuntimeError('等待新摄像头画面超时，请检查设备连接。')
            return self.latest

    def run(self):
        capture = None
        try:
            import cv2
            capture = cv2.VideoCapture(self.index)
            if not capture.isOpened():
                raise RuntimeError('无法打开摄像头，请检查设备编号、占用情况和系统摄像头权限。')
            # 不支持时使用设备实际返回的分辨率。
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 后端不支持时仍由最近帧机制避免应用层积压。
            while not self.stopped.is_set():
                ok, frame = capture.read()
                captured_at = time.monotonic()
                if not ok:
                    raise RuntimeError('摄像头读取失败，请检查连接后重启程序。')
                self.publish(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), captured_at)
        except Exception as error:
            with self.condition:
                self.error = str(error)
                self.condition.notify_all()
        finally:
            if capture is not None:
                capture.release()

    def stop(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()

    def close(self):
        self.stop()
        self.thread.join(timeout=2)
