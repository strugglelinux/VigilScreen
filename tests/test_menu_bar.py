if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from menu_bar import MenuBarIndicator


class MenuBarTests(unittest.TestCase):
    def test_multiple_people_count_and_pause_reset(self):
        root = Mock()
        root.after.return_value = 'timer'
        bar = Mock()
        appkit = SimpleNamespace(NSStatusBar=Mock(),NSVariableStatusItemLength=-1)
        appkit.NSStatusBar.systemStatusBar.return_value = bar
        with patch.dict(sys.modules,{'AppKit':appkit}):
            indicator = MenuBarIndicator(root,None)
        button = indicator.item.button()
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


if __name__=='__main__':
    unittest.main()
