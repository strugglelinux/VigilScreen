if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from gui import TrainingWindow


class Value:
    def __init__(self,value=''):
        self.value=value
    def get(self):
        return self.value
    def set(self,value):
        self.value=value


class ValidationUiTests(unittest.TestCase):
    def create_model(self,root,name):
        directory=root/name
        directory.mkdir()
        profile=directory/'active_profile.json'
        profile.write_text(json.dumps({'format_version':1,'profile_id':name,'classes':[name],'validation':'validation.json'}))
        report={'profile_id':name,'images':1,'known_identity_match_rate':1.,'unknown_false_accept_rate':None,
                'training_content_overlaps':0,'per_class':{name:{'unusable':0}},
                'details':[{'expected':name,'predicted':name,'similarity':.8,'training_content_overlap':False,'source':f'{name}.png'}]}
        (directory/'validation.json').write_text(json.dumps(report))
        return directory,profile.resolve()

    def make_window(self,directory):
        window=TrainingWindow.__new__(TrainingWindow)
        window.checkpoints=Value(str(directory))
        window.validation_status=Value()
        window.validation_expected_choice=Mock()
        window.validation_table=Mock()
        window.validation_table.get_children.return_value=['old-row']
        window.last_validation=None
        window.last_validation_profile=None
        return window

    def test_changed_model_never_displays_another_models_cached_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            old,old_profile=self.create_model(root,'alice')
            new,new_profile=self.create_model(root,'bob')
            window=self.make_window(new)
            window.last_validation=old/'validation.json'
            window.last_validation_profile=old_profile
            window.display_validation()
            values=window.validation_table.insert.call_args.kwargs['values']
            self.assertEqual(values[:2],('bob','bob'))
            self.assertIn('模型：bob',window.validation_status.get())

    def test_wrong_report_model_id_is_rejected_and_old_table_cleared(self):
        with tempfile.TemporaryDirectory() as directory:
            model,profile=self.create_model(Path(directory),'alice')
            path=model/'validation.json'
            report=json.loads(path.read_text());report['profile_id']='bob'
            path.write_text(json.dumps(report))
            window=self.make_window(model)
            window.display_validation()
            window.validation_table.insert.assert_not_called()
            window.validation_table.delete.assert_any_call('old-row')
            self.assertIn('不属于当前',window.validation_status.get())

    def test_directory_change_clears_cached_validation(self):
        window=self.make_window(Path('models'))
        window.last_validation=Path('old.json')
        window.last_validation_profile=Path('old-profile.json')
        window.validation_model_changed()
        self.assertIsNone(window.last_validation)
        self.assertIsNone(window.last_validation_profile)
        window.validation_table.delete.assert_called_with('old-row')


if __name__=='__main__':
    unittest.main()
