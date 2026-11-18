"""原生托盘与 Dock 策略验证；只使用模拟摄像头。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import json
import time
import traceback
import tkinter as tk
from unittest.mock import Mock,patch
import numpy as np
from main import VigilScreen


def watch_hidden_window(controller,title):
    """记录隐藏期间的原生解除最小化/重新显示，而不是仅检查最后一帧。"""
    from Foundation import NSObject,NSNotificationCenter
    from AppKit import NSWindowDidDeminiaturizeNotification,NSApplicationDidUnhideNotification

    class VigilScreenHiddenWindowTestObserver(NSObject):
        def deminiaturized_(self,notification):
            if (self.controller.hidden and not self.controller.restoring and not self.controller.closed
                    and str(notification.object().title())==self.title):
                self.unexpected.append('deminiaturize')

        def unhidden_(self,notification):
            if self.controller.hidden and not self.controller.restoring and not self.controller.closed:
                self.unexpected.append('unhide')

    observer=VigilScreenHiddenWindowTestObserver.alloc().init()
    observer.controller=controller;observer.title=title;observer.unexpected=[]
    center=NSNotificationCenter.defaultCenter()
    center.addObserver_selector_name_object_(observer,'deminiaturized:',NSWindowDidDeminiaturizeNotification,None)
    center.addObserver_selector_name_object_(observer,'unhidden:',NSApplicationDidUnhideNotification,controller.application)
    return observer,center


def tracked_restore(indicator,actor=None,burst=1):
    """从 Cocoa 计时器展开真实菜单，在事件跟踪模式下选中恢复。"""
    from AppKit import NSEventTrackingRunLoopMode
    from Foundation import NSObject,NSTimer,NSRunLoop,NSDefaultRunLoopMode

    if actor is None:
        class VigilScreenTrackingTestAction(NSObject):
            def menuWillOpen_(self,menu):
                self.opened=True

            def menuDidClose_(self,menu):
                self.menu_closed=True

            def selectWindow_(self,timer):
                self.fired=True
                self.tracking_mode=NSRunLoop.currentRunLoop().currentMode()==NSEventTrackingRunLoopMode
                for _ in range(self.burst):
                    self.menu.performActionForItemAtIndex_(0)
                self.menu.cancelTracking()

            def openMenu_(self,timer):
                self.item.popUpStatusItemMenu_(self.menu)
                self.popup_returned=True

        actor=VigilScreenTrackingTestAction.alloc().init()
    actor.menu=indicator.menu;actor.item=indicator.item
    actor.burst=burst
    actor.opened=False;actor.fired=False;actor.menu_closed=False
    actor.popup_returned=False;actor.tracking_mode=False
    indicator.menu.setDelegate_(actor)
    actor.timer=NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(.2,actor,'selectWindow:',None,False)
    NSRunLoop.currentRunLoop().addTimer_forMode_(actor.timer,NSEventTrackingRunLoopMode)
    actor.open_timer=NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(.05,actor,'openMenu:',None,False)
    NSRunLoop.currentRunLoop().addTimer_forMode_(actor.open_timer,NSDefaultRunLoopMode)
    return actor


def main(tracking=False,repeat=1,burst=1,late_hide=False):
    state={'opened':0,'released':0}
    def camera(index):
        state['opened']+=1
        capture=Mock();capture.isOpened.return_value=True
        def read():
            time.sleep(.03)
            return True,np.zeros((240,320,3),dtype=np.uint8)
        capture.read.side_effect=read
        def release():state['released']+=1
        capture.release.side_effect=release
        return capture
    root=tk.Tk();errors=[]
    root.report_callback_exception=lambda *args:errors.append(str(args[1]))
    with patch('cv2.VideoCapture',side_effect=camera),patch('chi_camera.FaceFinder') as finder,\
         patch('chi_camera.GazeEstimator'):
        finder.return_value.detect.return_value=[]
        app=VigilScreen(root)
        observer,notification_center=watch_hidden_window(app.tray,root.title())
        started=time.monotonic();stage=[0];actors=[];completed=[0];injected=[False]
        def tick():
            try:
                if time.monotonic()-started>max(20,repeat*4+5):
                    raise AssertionError((stage,root.state(),app.tray.hidden,app.tray.application.activationPolicy(),
                                          [(actor.opened,actor.fired,actor.menu_closed) for actor in actors]))
                if stage[0]==0:
                    assert state['opened']==0
                    assert app.monitor.menu_indicator.item.button().image() is not None
                    assert app.monitor.menu_indicator.menu.numberOfItems()==4
                    app.monitor.start_button.invoke()
                    stage[0]=1
                elif stage[0]==1 and app.monitor.last_frame is not None:
                    root.iconify()
                    stage[0]=2
                elif stage[0]==2 and app.tray.hidden and app.tray.settle_timer is None:
                    windows=[window for window in app.tray.application.windows() if 'VigilScreen' in str(window.title())]
                    assert windows
                    assert root.winfo_ismapped()==0,(root.state(),windows)
                    assert app.tray.application.isHidden()
                    assert all(not window.isVisible() or app.tray.application.isHidden() for window in windows)
                    assert app.tray.application.activationPolicy()==app.tray.accessory_policy
                    assert app.monitor.menu_indicator.item.isVisible()
                    assert app.monitor.want_running and app.monitor.worker.is_alive()
                    assert state['opened']==1 and state['released']==0
                    assert str(app.monitor.menu_indicator.item.button().title())==''
                    app.monitor.menu_indicator.show_hit(None,'simulated.jpg')
                    assert 'Peek 1' in str(app.monitor.menu_indicator.item.button().title())
                    if tracking:
                        actor=tracked_restore(app.monitor.menu_indicator,actors[0] if actors else None,burst)
                        if not actors:
                            actors.append(actor)
                    else:
                        for _ in range(burst):
                            app.monitor.menu_indicator.menu.performActionForItemAtIndex_(0)
                    injected[0]=False
                    stage[0]=3
                elif stage[0]==3 and late_hide and app.tray.restoring and not injected[0]:
                    # Tk 仍认为 normal，但 Cocoa 窗口已隐藏，模拟用户遇到的状态。
                    for window,_ in app.tray.hidden_windows:
                        window.orderOut_(None)
                    app.tray.application.hide_(None)
                    injected[0]=True
                elif stage[0]==3 and not app.tray.hidden and not app.tray.restoring:
                    if tracking:
                        assert actors[0].fired and actors[0].tracking_mode and actors[0].popup_returned,(actors[0].opened,actors[0].fired,actors[0].menu_closed,actors[0].popup_returned,actors[0].tracking_mode)
                        actors[0].timer.invalidate()
                        actors[0].open_timer.invalidate()
                        app.monitor.menu_indicator.menu.setDelegate_(None)
                    assert root.state()=='normal'
                    assert root.winfo_ismapped() and app.notebook.winfo_viewable()
                    windows=[window for window in app.tray.application.windows() if str(window.title())==root.title()]
                    assert windows and all(window.isVisible() and not window.isMiniaturized() for window in windows)
                    assert not app.tray.application.isHidden()
                    assert app.tray.application.activationPolicy()==app.tray.regular_policy
                    assert app.monitor.want_running and state['opened']==1
                    assert not app.closing
                    assert app.monitor.menu_indicator.hit_count==1
                    completed[0]+=1
                    app.monitor.menu_indicator.reset(None)
                    if completed[0]<repeat and completed[0]%2:
                        root.iconify()
                    else:
                        app.monitor.menu_indicator.menu.performActionForItemAtIndex_(1)
                    stage[0]=2 if completed[0]<repeat else 4
                elif stage[0]==4 and app.tray.hidden and app.tray.settle_timer is None:
                    assert app.monitor.want_running
                    assert state['opened']==1 and state['released']==0
                    app.monitor.menu_indicator.menu.performActionForItemAtIndex_(3)
                    stage[0]=5
                    return
                root.after(75,tick)
            except Exception:
                errors.append(traceback.format_exc())
                app.close()
        root.after(200,tick)
        root.mainloop()
        notification_center.removeObserver_(observer)
    if errors:
        raise AssertionError(errors)
    assert stage[0]==5
    assert state['opened']==state['released']==1
    assert not observer.unexpected,observer.unexpected
    print(json.dumps({'ok':True,'camera_accessed':False,'menu_image':True,'minimize_hides_dock':True,
                      'restore_shows_window_and_dock':True,'monitoring_continues_in_tray':True,
                      'menu_actions_verified':True,'native_menu_tracking_verified':tracking,
                      'restore_cycles':completed[0],'show_request_burst':burst,'late_native_hide_tested':late_hide,
                      'unexpected_reappearances_while_hidden':len(observer.unexpected),**state}))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking',action='store_true')
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--burst',type=int,default=1)
    parser.add_argument('--late-hide',action='store_true')
    args=parser.parse_args()
    if min(args.repeat,args.burst)<1:
        parser.error('--repeat 和 --burst 至少为 1')
    main(tracking=args.tracking,repeat=args.repeat,burst=args.burst,late_hide=args.late_hide)
