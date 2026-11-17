"""由 Tk 主线程管理的 macOS 原生菜单栏状态提示。"""
import time


class MenuBarIndicator:
    @property
    def scope(self):
        return '全员' if self.target is None else f'指定：{self.target}'

    def __init__(self, root, target):
        from AppKit import NSStatusBar, NSVariableStatusItemLength

        self.root = root
        self.bar = NSStatusBar.systemStatusBar()
        self.item = self.bar.statusItemWithLength_(NSVariableStatusItemLength)
        self.timer = None
        self.reset(target)

    def reset(self, target):
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None
        self.target = target
        self.hit_count = 0
        self.last_hit_at = None
        self.item.button().setTitle_(f'VigilScreen · {self.scope}')
        self.item.button().setToolTip_(f'{self.scope}模式：等待注视镜头或已校准屏幕')

    def show_hit(self, target, path, count=1):
        if target != self.target:
            self.reset(target)
        elif self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None
        now = time.monotonic()
        if self.last_hit_at is None or now-self.last_hit_at > 10:
            self.hit_count = 0
        self.hit_count += count
        self.last_hit_at = now
        self.item.button().setTitle_(f'Peek {self.hit_count} · {self.scope}')
        self.item.button().setToolTip_(f'{self.scope} · 第 {self.hit_count} 次 Peek（相邻命中间隔不超过 10 秒时累计）\n截图：{path}')
        self.timer = self.root.after(3000, self.clear_hit)

    def clear_hit(self):
        self.timer = None
        self.item.button().setTitle_(f'VigilScreen · {self.scope}')

    def pause(self,target):
        self.reset(target)
        self.item.button().setTitle_(f'已暂停 · {self.scope}')
        self.item.button().setToolTip_('监测已暂停，已请求释放摄像头。')

    def close(self):
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None
        if self.item is not None:
            self.bar.removeStatusItem_(self.item)
            self.item = None
