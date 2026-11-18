from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock,patch

from tray_window import TrayWindow,attach_tray


class TrayWindowTests(unittest.TestCase):
    def controller(self):
        root=Mock()
        root.grab_current.return_value=None
        root.after_idle.return_value='idle'
        root.state.return_value='iconic'
        root.winfo_ismapped.return_value=True
        indicator=Mock();application=Mock()
        application.windows.return_value=[]
        application.isHidden.return_value=False
        application.setActivationPolicy_.return_value=True
        appkit=SimpleNamespace(NSApplication=Mock(),NSApplicationActivationPolicyAccessory=1,
                               NSApplicationActivationPolicyRegular=0,NSWindowAnimationBehaviorNone=2)
        appkit.NSApplication.sharedApplication.return_value=application
        with patch.dict(sys.modules,{'AppKit':appkit}):
            controller=TrayWindow(root,indicator,Mock())
        return controller,root,indicator,application

    def test_hide_removes_dock_entry_and_restore_brings_window_back(self):
        controller,root,indicator,application=self.controller()
        self.assertTrue(controller.hide())
        self.assertTrue(controller.hidden)
        application.setActivationPolicy_.assert_called_with(1)
        root.withdraw.assert_called_once()
        controller.on_quit.assert_not_called()
        root.deiconify.reset_mock()
        controller.restore()
        application.setActivationPolicy_.assert_called_with(0)
        root.deiconify.assert_called_once()
        self.assertFalse(controller.hidden)
        indicator.configure_window_menu.assert_called_once_with(controller.restore,controller.hide,controller.on_quit)

    def test_child_unmap_does_not_hide_and_cancelled_minimize_is_ignored(self):
        controller,root,_,application=self.controller()
        controller.on_unmap(SimpleNamespace(widget=Mock()))
        root.after_idle.assert_not_called()
        controller.on_unmap(SimpleNamespace(widget=root))
        controller.on_unmap(SimpleNamespace(widget=root))
        root.after_idle.assert_called_once()
        root.state.return_value='normal'
        controller.hide_if_minimized()
        application.setActivationPolicy_.assert_not_called()

    def test_modal_or_failed_policy_never_loses_window(self):
        controller,root,_,application=self.controller()
        root.grab_current.return_value=Mock()
        self.assertFalse(controller.hide())
        root.withdraw.assert_not_called()
        root.grab_current.return_value=None
        application.setActivationPolicy_.return_value=False
        self.assertFalse(controller.hide())
        root.withdraw.assert_not_called()
        self.assertFalse(controller.hidden)

    def test_rebuild_reuses_minimize_binding_and_quit_action(self):
        controller,root,_,_=self.controller()
        replacement=Mock()
        with patch('tray_window.sys.platform','darwin'):
            self.assertIs(attach_tray(root,replacement,controller.on_quit,controller),controller)
        self.assertEqual(root.bind.call_count,1)
        replacement.configure_window_menu.assert_called_once()
        controller.on_unmap(SimpleNamespace(widget=root))
        controller.close();controller.close()
        root.after_cancel.assert_called_once_with('idle')
        root.unbind.assert_called_once()
        controller.restore()
        root.deiconify.assert_not_called()

    def test_other_platform_keeps_normal_minimization(self):
        with patch('tray_window.sys.platform','linux'):
            self.assertIsNone(attach_tray(Mock(),Mock(),Mock()))

    def test_hiding_never_deminiaturizes_and_restore_clears_mini_state(self):
        controller,root,_,application=self.controller()
        root.title.return_value='VigilScreen'
        window=Mock();window.title.return_value='VigilScreen'
        window.animationBehavior.return_value=3
        window.isMiniaturized.return_value=True
        window.deminiaturize_.side_effect=lambda sender:setattr(window.isMiniaturized,'return_value',False)
        application.windows.return_value=[window]
        controller.hide()
        window.isMiniaturized.return_value=True  # 模拟随后到达的 Cocoa 最小化完成事件。
        with patch('tray_window.time.monotonic',return_value=controller.hide_started+1):
            root.winfo_ismapped.return_value=False
            application.isHidden.return_value=True
            controller.settle_hidden()
        window.deminiaturize_.assert_not_called()
        root.deiconify.assert_not_called()
        self.assertIsNone(controller.settle_timer)
        controller.restore()
        self.assertEqual(window.deminiaturize_.call_count,1)
        root.winfo_ismapped.return_value=True
        application.isHidden.return_value=False
        with patch('tray_window.time.monotonic',return_value=controller.restore_started+1):
            controller.settle_visible(controller.transition)
        window.setAnimationBehavior_.assert_called_with(3)

    def test_old_focus_callback_cannot_bring_hidden_window_back(self):
        controller,root,_,_=self.controller()
        controller.restore()
        with patch('tray_window.time.monotonic',return_value=controller.restore_started+1):
            controller.settle_visible(controller.transition)
        focus_callback=root.after_idle.call_args.args[0]
        controller.hide()
        focus_callback()
        root.focus_force.assert_not_called()

    def test_repeated_show_requests_share_one_restore_and_keep_native_window(self):
        controller,root,_,application=self.controller()
        root.title.return_value='VigilScreen'
        window=Mock();window.title.return_value='VigilScreen'
        window.isMiniaturized.return_value=False;window.isVisible.return_value=True
        application.windows.return_value=[window]
        for _ in range(25):
            controller.restore()
        self.assertEqual(root.deiconify.call_count,1)
        self.assertEqual(len(controller.hidden_windows),1)
        window.makeKeyAndOrderFront_.assert_called_once()
        self.assertTrue(controller.restoring)

    def test_stale_hide_or_restore_callbacks_cannot_reverse_new_request(self):
        controller,root,_,_=self.controller()
        controller.hide();old_hide=controller.transition
        controller.restore();old_restore=controller.transition
        root.withdraw.reset_mock()
        controller.settle_hidden(old_hide)
        root.withdraw.assert_not_called()
        controller.hide()
        root.deiconify.reset_mock()
        controller.settle_visible(old_restore)
        root.deiconify.assert_not_called()
        self.assertTrue(controller.hidden)

    def test_native_invisible_window_is_shown_even_when_tk_reports_normal(self):
        controller,root,_,application=self.controller()
        root.title.return_value='VigilScreen';root.state.return_value='normal'
        window=Mock();window.title.return_value='VigilScreen'
        window.isMiniaturized.return_value=False;window.isVisible.return_value=False
        window.makeKeyAndOrderFront_.side_effect=lambda sender:setattr(window.isVisible,'return_value',True)
        application.windows.return_value=[window]
        controller.restore()
        window.isVisible.return_value=False  # 模拟迟到的原生隐藏事件。
        with patch('tray_window.time.monotonic',return_value=controller.restore_started+.2):
            controller.settle_visible(controller.transition)
        self.assertEqual(window.makeKeyAndOrderFront_.call_count,2)
        self.assertTrue(controller.restoring)
        with patch('tray_window.time.monotonic',return_value=controller.restore_started+1):
            controller.settle_visible(controller.transition)
        self.assertFalse(controller.restoring)
        self.assertTrue(controller.window_visible())


if __name__=='__main__':
    unittest.main()
