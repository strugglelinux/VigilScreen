if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

import sys
import queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from menu_bar import MenuBarIndicator,make_action_target


class MenuBarTests(unittest.TestCase):
    def test_multiple_people_count_and_pause_reset(self):
        root = Mock()
        root.after.return_value = 'timer'
        bar = Mock()
        appkit = SimpleNamespace(NSStatusBar=Mock(),NSVariableStatusItemLength=-1,NSImage=Mock(),NSImageLeft=2)
        appkit.NSStatusBar.systemStatusBar.return_value = bar
        with patch.dict(sys.modules,{'AppKit':appkit}):
            indicator = MenuBarIndicator(root,None)
        button = indicator.item.button()
        button.setImage_.assert_called_once()
        indicator.show_hit(None,'one.jpg',count=2)
        button.setTitle_.assert_called_with('Peek 2 · 全员')
        indicator.show_hit(None,'two.jpg',count=1)
        button.setTitle_.assert_called_with('Peek 3 · 全员')
        indicator.show_hit('chi','three.jpg')
        button.setTitle_.assert_called_with('Peek 1 · 指定：chi')
        indicator.clear_hit()
        button.setTitle_.assert_called_with('VigilScreen · 指定：chi')
        indicator.pause(None)
        button.setTitle_.assert_called_with('已暂停 · 全员')
        self.assertEqual(indicator.hit_count,0)
        indicator.close()
        bar.removeStatusItem_.assert_called_once()

    def test_tray_menu_keeps_restore_hide_and_exit_callbacks(self):
        indicator=MenuBarIndicator.__new__(MenuBarIndicator)
        indicator.root=Mock();indicator.item=Mock()
        indicator.action_queue=queue.Queue();indicator.action_timer=None
        appkit=SimpleNamespace(NSMenu=Mock(),NSMenuItem=Mock())
        callbacks=[Mock(),Mock(),Mock()]
        with patch.dict(sys.modules,{'AppKit':appkit}),patch('menu_bar.make_action_target') as target:
            indicator.configure_window_menu(*callbacks)
        target.assert_called_once_with(indicator.action_queue)
        self.assertEqual(indicator.action_callbacks,dict(zip(['show','hide','quit'],callbacks)))
        self.assertEqual(indicator.menu.addItem_.call_count,4)
        indicator.item.setMenu_.assert_called_once_with(indicator.menu)

    def test_native_callbacks_only_enqueue_actions_without_tk_calls(self):
        class NativeObject:
            @classmethod
            def alloc(cls):
                return cls()
            def init(self):
                return self
        actions=queue.Queue()
        with patch('menu_bar._action_target_class',None),patch.dict(sys.modules,{
            'objc':SimpleNamespace(python_method=lambda method:method),
            'Foundation':SimpleNamespace(NSObject=NativeObject),
        }):
            target=make_action_target(actions)
            target.showWindow_(None)
            target.hideWindow_(None)
            target.quitApp_(None)
        self.assertEqual(list(actions.queue),['show','hide','quit'])
        self.assertFalse(hasattr(target,'root'))

    def test_actions_run_only_on_tk_poll_and_quit_stops_rescheduling(self):
        indicator=MenuBarIndicator.__new__(MenuBarIndicator)
        indicator.root=Mock();indicator.item=Mock();indicator.action_timer=None
        indicator.action_queue=queue.Queue()
        show=Mock()
        def quit():
            indicator.item=None
        indicator.action_callbacks={'show':show,'quit':quit}
        indicator.action_queue.put('show')
        show.assert_not_called()
        indicator.poll_actions()
        show.assert_called_once()
        indicator.root.winfo_toplevel().after.assert_called_once()
        indicator.root.winfo_toplevel().after.reset_mock()
        indicator.action_queue.put('quit')
        indicator.poll_actions()
        indicator.root.winfo_toplevel().after.assert_not_called()

    def test_open_native_menu_defers_restore_until_tracking_finishes(self):
        indicator=MenuBarIndicator.__new__(MenuBarIndicator)
        indicator.root=Mock();indicator.item=Mock();indicator.action_timer=None
        indicator.action_queue=queue.Queue();show=Mock()
        indicator.action_callbacks={'show':show}
        indicator.action_queue.put('show')
        with patch('menu_bar.native_menu_tracking',return_value=True):
            indicator.poll_actions()
        show.assert_not_called()
        self.assertEqual(indicator.action_queue.qsize(),1)
        with patch('menu_bar.native_menu_tracking',return_value=False):
            indicator.poll_actions()
        show.assert_called_once()

    def test_compact_icon_keeps_peek_count_and_restores_text(self):
        indicator=MenuBarIndicator.__new__(MenuBarIndicator)
        indicator.root=Mock();indicator.item=Mock();indicator.timer=None
        indicator.compact=False;indicator.has_image=True;indicator.title=''
        indicator.reset(None)
        indicator.set_compact(True)
        indicator.item.button().setTitle_.assert_called_with('')
        indicator.show_hit(None,'hit.jpg',count=2)
        indicator.item.button().setTitle_.assert_called_with('Peek 2 · 全员')
        indicator.set_compact(False)
        self.assertEqual(indicator.hit_count,2)
        indicator.set_compact(True)
        indicator.clear_hit()
        indicator.item.button().setTitle_.assert_called_with('')
        indicator.set_compact(False)
        indicator.item.button().setTitle_.assert_called_with('VigilScreen · 全员')


if __name__=='__main__':
    unittest.main()
