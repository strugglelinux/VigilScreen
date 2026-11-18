"""由 Tk 主线程管理的 macOS 原生菜单栏状态提示。"""
import queue
import sys
import time

_action_target_class = None


def native_menu_tracking():
    if sys.platform!='darwin':
        return False
    from AppKit import NSEventTrackingRunLoopMode
    from Foundation import NSRunLoop
    return NSRunLoop.currentRunLoop().currentMode()==NSEventTrackingRunLoopMode


def make_action_target(actions):
    global _action_target_class
    if _action_target_class is None:
        import objc
        from Foundation import NSObject

        class VigilScreenTrayActionTarget(NSObject):
            @objc.python_method
            def dispatch(self,name):
                # Cocoa 的菜单跟踪循环中不能调用任何 Tk 方法，包括 after/after_idle。
                self.actions.put(name)

            def showWindow_(self,sender):
                self.dispatch('show')

            def hideWindow_(self,sender):
                self.dispatch('hide')

            def quitApp_(self,sender):
                self.dispatch('quit')

        _action_target_class = VigilScreenTrayActionTarget
    target = _action_target_class.alloc().init()
    target.actions = actions
    return target


class MenuBarIndicator:
    @property
    def scope(self):
        return '全员' if self.target is None else f'指定：{self.target}'

    def __init__(self, root, target):
        from AppKit import NSStatusBar, NSVariableStatusItemLength,NSImage,NSImageLeft

        self.root = root
        self.bar = NSStatusBar.systemStatusBar()
        self.item = self.bar.statusItemWithLength_(NSVariableStatusItemLength)
        self.menu = None
        self.menu_actions = None
        self.action_queue = queue.Queue()
        self.action_callbacks = {}
        self.action_timer = None
        self.compact = False
        self.title = ''
        image = NSImage.imageWithSystemSymbolName_accessibilityDescription_('eye','VigilScreen 视线监控')
        self.has_image = image is not None
        if image is not None:
            image.setSize_((18,18))
            image.setTemplate_(True)
            self.item.button().setImage_(image)
            self.item.button().setImagePosition_(NSImageLeft)
        self.timer = None
        self.reset(target)

    def set_title(self,title):
        self.title = title
        compact = self.compact and self.has_image and not title.startswith('Peek ')
        self.item.button().setTitle_('' if compact else title)

    def set_compact(self,compact):
        self.compact = compact
        self.set_title(self.title)

    def configure_window_menu(self,show,hide,quit):
        from AppKit import NSMenu,NSMenuItem
        self.action_callbacks = {'show':show,'hide':hide,'quit':quit}
        self.menu_actions = make_action_target(self.action_queue)
        self.menu = NSMenu.alloc().init()
        for title,action in [('显示主窗口','showWindow:'),('隐藏到菜单栏','hideWindow:'),
                             (None,None),('退出 VigilScreen','quitApp:')]:
            if title is None:
                self.menu.addItem_(NSMenuItem.separatorItem())
            else:
                item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title,action,'')
                item.setTarget_(self.menu_actions)
                self.menu.addItem_(item)
        self.item.setMenu_(self.menu)
        if self.action_timer is None:
            self.action_timer = self.root.winfo_toplevel().after(50,self.poll_actions)

    def poll_actions(self):
        self.action_timer = None
        if self.item is None:
            return
        try:
            if native_menu_tracking():
                return
            while self.item is not None:
                action = self.action_queue.get_nowait()
                callback = self.action_callbacks.get(action)
                if callback is not None:
                    callback()
        except queue.Empty:
            pass
        finally:
            if self.item is not None:
                self.action_timer = self.root.winfo_toplevel().after(50,self.poll_actions)

    def reset(self, target):
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None
        self.target = target
        self.hit_count = 0
        self.last_hit_at = None
        self.set_title(f'VigilScreen · {self.scope}')
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
        self.set_title(f'Peek {self.hit_count} · {self.scope}')
        self.item.button().setToolTip_(f'{self.scope} · 第 {self.hit_count} 次 Peek（相邻命中间隔不超过 10 秒时累计）\n截图：{path}')
        self.timer = self.root.after(3000, self.clear_hit)

    def clear_hit(self):
        self.timer = None
        self.set_title(f'VigilScreen · {self.scope}')

    def pause(self,target):
        self.reset(target)
        self.set_title(f'已暂停 · {self.scope}')
        self.item.button().setToolTip_('监测已暂停，已请求释放摄像头。')

    def close(self):
        if self.action_timer is not None:
            self.root.winfo_toplevel().after_cancel(self.action_timer)
            self.action_timer = None
        self.action_callbacks = {}
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None
        if self.item is not None:
            self.bar.removeStatusItem_(self.item)
            self.item = None
        self.menu_actions = None
        self.menu = None
