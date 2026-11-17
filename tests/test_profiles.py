if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from PIL import Image

from identity_training import available_identities,evaluate,heldout_records,main,train_profile,validation_records


class ProfileTests(unittest.TestCase):
    def test_identity_training_defaults_to_folder_materials(self):
        with patch('identity_training.train_profile') as trainer:
            main(['train','--data-dir','materials','--output-dir','models'])
        self.assertEqual(trainer.call_args.args[0].dataset,'folder')

    def test_malformed_optional_profile_has_no_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'profile.json'
            for value in [[],{'profile':[]},{'format_version':1,'classes':42}]:
                path.write_text(json.dumps(value))
                self.assertEqual(available_identities(path),[])

    def test_heldout_snapshot_paths_survive_project_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root/'old'
            dataset = old/'run/dataset'
            photo = old/'run/materials/alice/one.png'
            dataset.mkdir(parents=True)
            photo.parent.mkdir(parents=True)
            Image.new('RGB',(20,20)).save(photo)
            (dataset/'preparation.json').write_text(json.dumps({'samples':[
                {'split':'val','category':'alice','source':'../materials/alice/one.png'}]}))
            old.rename(root/'moved')
            rows = heldout_records(root/'moved/run/profile.json',{'dataset':'dataset'})
            self.assertTrue(Path(rows[0]['source']).is_file())
            self.assertTrue(Path(rows[0]['source']).is_relative_to((root/'moved').resolve()))

    def test_missing_profile_does_not_block_all_mode_choices(self):
        self.assertEqual(available_identities('/no/such/profile.json'),[])

    def test_validation_uses_all_images_in_denominator_and_marks_training_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for name,color in [('known','red'),('stranger','blue'),('unusable','green')]:
                path = root/f'{name}.png'
                Image.new('RGB',(30,30),color).save(path)
                files.append(path)
            verifier = Mock()
            verifier.gallery = {'alice':[]}
            verifier.identify.side_effect = [('alice',.8),('alice',.6)]
            finder = Mock()
            finder.detect.side_effect = [[object()],[object()],[]]
            report = evaluate(verifier,[{'source':str(p),'category':category}
                              for p,category in zip(files,['alice','unknown','alice'])],finder,
                              [hashlib.sha256(files[0].read_bytes()).hexdigest()])
            self.assertEqual(report['images'],3)
            self.assertEqual(report['known_identity_match_rate'],.5)
            self.assertEqual(report['unknown_false_accept_rate'],1.)
            self.assertEqual(report['training_content_overlaps'],1)
            self.assertEqual(report['per_class']['alice']['unusable'],1)

    def test_failed_training_does_not_replace_active_identity_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source,output = root/'input',root/'output'
            (source/'alice').mkdir(parents=True)
            output.mkdir()
            active = output/'active_profile.json'
            active.write_text('{"old":true}')
            args = SimpleNamespace(data_dir=source,output_dir=output,dataset='folder',
                                   face_model=root/'face.onnx',identity_model=root/'id.onnx',seed=42,val_ratio=.2)
            with patch('prepare_face_training.prepare',return_value={'classes':['alice']}),\
                 patch('chi_camera.FaceFinder'),patch('identity_training.IdentityVerifier'),\
                 patch('test.train',side_effect=RuntimeError('中途停止')):
                with self.assertRaises(RuntimeError):
                    train_profile(args)
            self.assertEqual(json.loads(active.read_text()),{'old':True})

    def test_flat_validation_directory_requires_explicit_expected_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Image.new('RGB',(20,20)).save(path/'one.png')
            with self.assertRaises(ValueError):
                validation_records(path)
            self.assertEqual(validation_records(path,'alice')[0]['category'],'alice')

    def test_mixed_validation_directory_is_rejected_instead_of_silently_skipping_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            Image.new('RGB',(20,20)).save(root/'one.png')
            (root/'alice').mkdir()
            Image.new('RGB',(20,20)).save(root/'alice/two.png')
            with self.assertRaisesRegex(ValueError,'混有'):
                validation_records(root,'alice')

    def test_validation_resolves_active_profile_once_and_records_model_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            profile=root/'run-a/profile.json'
            metadata={'profile_id':'run-a','classes':['alice'],'training_hashes':[]}
            verifier=SimpleNamespace(threshold=.5,margin=.06)
            output=root/'report.json'
            with patch('identity_training.read_profile',return_value=(profile,metadata)) as reader,\
                 patch('identity_training.load_profile',return_value=(verifier,metadata)) as loader,\
                 patch('identity_training.heldout_records',return_value=[{'source':'one.png','category':'alice'}]) as heldout,\
                 patch('chi_camera.FaceFinder'),patch('identity_training.evaluate',return_value={'details':[]}):
                main(['validate','--profile',str(root/'active_profile.json'),'--output',str(output)])
            reader.assert_called_once()
            self.assertEqual(loader.call_args.args[0],profile)
            heldout.assert_called_once_with(profile,metadata)
            self.assertEqual(json.loads(output.read_text())['profile_id'],'run-a')


if __name__=='__main__':
    unittest.main()
