"""最小化时隐藏到 macOS 状态栏，保留后台任务与恢复入口。"""
import sys
import time


class TrayWindow:
    def __init__(self,root,indicator,on_quit):
        from AppKit import NSApplication,NSApplicationActivationPolicyAccessory,NSApplicationActivationPolicyRegular,NSWindowAnimationBehaviorNone
        self.root = root
        self.application = NSApplication.sharedApplication()
        self.accessory_policy = NSApplicationActivationPolicyAccessory
        self.regular_policy = NSApplicationActivationPolicyRegular
        self.no_animation = NSWindowAnimationBehaviorNone
        self.hidden_windows = []
        self.on_quit = on_quit
        self.indicator = None
        self.hidden = False
        self.restoring = False
        self.transition = 0
        self.closed = False
        self.pending_hide = None
        self.settle_timer = None
        self.restore_timer = None
        self.binding = root.bind('<Unmap>',self.on_unmap,add='+')
        self.attach(indicator)

    def attach(self,indicator):
        self.indicator = indicator
        indicator.configure_window_menu(self.restore,self.hide,self.on_quit)

    def on_unmap(self,event):
        if (not self.closed and not self.hidden and not self.restoring and event.widget is self.root
                and self.root.state()=='iconic' and self.pending_hide is None):
            revision = self.transition
            self.pending_hide = self.root.after_idle(lambda:self.hide_if_minimized(revision))

    def hide_if_minimized(self,revision=None):
        if revision is not None and revision!=self.transition:
            return
        self.pending_hide = None
        if not self.closed and not self.restoring and self.root.state()=='iconic':
            self.hide()

    def window_records(self):
        records=list(self.hidden_windows)
        for window in self.application.windows():
            if str(window.title())==str(self.root.title()) and not any(window==saved for saved,_ in records):
                records.append((window,window.animationBehavior()))
        return records

    def cancel_timer(self,name):
        timer=getattr(self,name)
        if timer is not None:
            self.root.after_cancel(timer)
            setattr(self,name,None)

    def hide(self):
        if self.closed or self.hidden or self.indicator.item is None:
            return False
        # 确认/初始化等模态窗口仍需用户处理，不能把它们一起藏到后台。
        if self.root.grab_current() is not None:
            self.restore()
            return False
        self.transition += 1
        revision = self.transition
        for timer in ('pending_hide','settle_timer','restore_timer'):
            self.cancel_timer(timer)
        self.restoring = False
        self.hidden = True
        if not self.application.setActivationPolicy_(self.accessory_policy):
            self.hidden = False
            self.restore()
            return False
        self.indicator.set_compact(True)
        self.hidden_windows = self.window_records()
        for window,_ in self.hidden_windows:
            window.setAnimationBehavior_(self.no_animation)
        # 隐藏只做隐藏；deminiaturize 会重新显示窗口，必须留到明确恢复时执行。
        self.application.hide_(None)
        self.root.withdraw()
        for window,_ in self.hidden_windows:
            window.orderOut_(None)
        self.hide_started = time.monotonic()
        self.hide_attempts = 0
        self.settle_timer = self.root.after(120,lambda:self.settle_hidden(revision))
        return True

    def settle_hidden(self,revision=None):
        if revision is not None and revision!=self.transition:
            return
        self.settle_timer = None
        if self.closed or not self.hidden:
            return
        self.hide_attempts += 1
        if self.root.winfo_ismapped():
            self.root.withdraw()
        for window,_ in self.hidden_windows:
            if window.isVisible():
                window.orderOut_(None)
        if not self.application.isHidden():
            self.application.hide_(None)
        # 应用隐藏也会隐藏它的最小化窗口，不必为清除 mini 标志反复显示主窗口。
        pending = (time.monotonic()-self.hide_started<.5 or self.root.winfo_ismapped()
                   or not self.application.isHidden())
        if pending and self.hide_attempts<12:
            revision = self.transition
            self.settle_timer = self.root.after(120,lambda:self.settle_hidden(revision))

    def restore(self):
        if self.closed or self.restoring:
            return
        self.transition += 1
        revision = self.transition
        self.restoring = True
        self.restore_started = time.monotonic()
        for timer in ('pending_hide','settle_timer','restore_timer'):
            self.cancel_timer(timer)
        self.hidden_windows = self.window_records()
        self.application.setActivationPolicy_(self.regular_policy)
        self.hidden = False
        self.indicator.set_compact(False)
        self.present_window(remap=not self.window_visible())
        self.restore_timer = self.root.after(100,lambda:self.settle_visible(revision))

    def present_window(self,remap=False):
        self.application.unhide_(None)
        if remap and self.root.grab_current() is None:
            # Tk 可能仍记录 normal，但 NSWindow 已被 orderOut；强制重建映射。
            self.root.withdraw()
        self.root.deiconify()
        self.root.lift()
        for window,_ in self.hidden_windows:
            if window.isMiniaturized():
                window.deminiaturize_(None)
            window.makeKeyAndOrderFront_(None)
        self.application.activateIgnoringOtherApps_(True)

    def window_visible(self):
        return (bool(self.root.winfo_ismapped()) and not self.application.isHidden()
                and all(window.isVisible() and not window.isMiniaturized() for window,_ in self.hidden_windows))

    def settle_visible(self,revision):
        if self.closed or revision!=self.transition or not self.restoring:
            return
        self.restore_timer = None
        self.hidden_windows = self.window_records()
        visible = self.window_visible()
        if not visible:
            self.present_window(remap=bool(self.root.winfo_ismapped()))
        elapsed = time.monotonic()-self.restore_started
        if elapsed<.6 or (not visible and elapsed<3):
            self.restore_timer = self.root.after(100,lambda:self.settle_visible(revision))
            return
        self.restoring = False
        if not self.window_visible():
            # 保留窗口引用，下次点击仍可修复映射，不把仅恢复 Dock 当成成功。
            return
        for window,animation in self.hidden_windows:
            window.setAnimationBehavior_(animation)
        self.hidden_windows = []
        focus = self.root.grab_current() or self.root
        def focus_window():
            if (not self.closed and revision==self.transition and not self.hidden
                    and not self.restoring and focus.winfo_exists()):
                focus.focus_force()
        self.root.after_idle(focus_window)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.transition += 1
        for timer in ('pending_hide','settle_timer','restore_timer'):
            self.cancel_timer(timer)
        self.root.unbind('<Unmap>',self.binding)
        self.indicator = None
        self.hidden_windows = []


def attach_tray(root,indicator,on_quit,controller=None):
    if sys.platform!='darwin' or indicator is None:
        return controller
    if controller is None:
        return TrayWindow(root,indicator,on_quit)
    controller.attach(indicator)
    return controller
