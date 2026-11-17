if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from identity import match_identity, enrollment_paths
from chi_camera import face_color


class IdentityTests(unittest.TestCase):
    def test_enrollment_only_reads_training_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / 'original_train.png'
            validation = root / 'original_val.png'
            training.touch()
            validation.touch()
            records = [
                {'split':'train','category':'me','source':str(training),'output':'unused'},
                {'split':'val','category':'me','source':str(validation),'output':'unused'},
                {'split':'train','category':'chi','source':str(validation),'output':'unused'},
            ]
            (root/'preparation.json').write_text(json.dumps({'samples':records}))
            self.assertEqual(enrollment_paths(root/'train','me'), [training])

    def test_stranger_is_unknown_even_with_closed_set_candidate(self):
        gallery={'me':np.array([[1.,0,0]]*3), 'chi':np.array([[0.,1,0]]*3)}
        self.assertEqual(match_identity([0,0,1],gallery)[0], 'unknown')
        self.assertEqual(match_identity([1,0,0],gallery)[0], 'me')
        self.assertEqual(match_identity([0,1,0],gallery)[0], 'chi')
        self.assertEqual(match_identity([1,1,0],gallery)[0], 'unknown')
        self.assertEqual(match_identity([0,0,0],gallery)[0], 'unknown')

    def test_single_accidental_match_is_not_enough(self):
        gallery={'me':np.array([[1.,0,0],[0.,1,0],[0.,0,1]])}
        self.assertEqual(match_identity([1,0,0],gallery)[0], 'unknown')
        self.assertNotEqual(face_color('unknown'), face_color('me'))
        self.assertNotEqual(face_color('chi',True), face_color('me'))

class EnrollmentAuditTests(unittest.TestCase):
    def test_counts_skip_reasons_and_content_fingerprint(self):
        from unittest.mock import patch,Mock
        from PIL import Image
        from identity import IdentityVerifier
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=root/'model.onnx';model.write_bytes(b'test')
            source=root/'samples'
            for category in ['chi','me']:
                (source/category).mkdir(parents=True)
                for index in range(4):Image.new('RGB',(40,40),'red').save(source/category/f'{index}.png')
            face=np.zeros(15,dtype=np.float32)
            finder=Mock();finder.detect.side_effect=[[face],[face],[face],[],[face],[face],[face],[]]
            with patch('cv2.FaceRecognizerSF'),patch.object(IdentityVerifier,'feature',return_value=np.array([1.,0.])):
                verifier=IdentityVerifier(model,source,finder,['chi','me'])
                for audit in verifier.enrollment_report['classes'].values():
                    self.assertEqual(audit['candidates'],4)
                    self.assertEqual(audit['accepted'],3)
                    self.assertEqual(len(audit['skipped']),1)
                report=root/'enrollment.json';verifier.save_report(report)
                self.assertEqual(json.loads(report.read_text())['fingerprint'],verifier.enrollment_report['fingerprint'])
                Image.new('RGB',(40,40),'blue').save(source/'chi/0.png')
                finder.detect.side_effect=None;finder.detect.return_value=[face]
                updated=IdentityVerifier(model,source,finder,['chi','me'])
                self.assertNotEqual(updated.enrollment_report['fingerprint'],verifier.enrollment_report['fingerprint'])
                self.assertNotEqual(updated.enrollment_report['snapshot_id'],verifier.enrollment_report['snapshot_id'])


if __name__=='__main__':
    unittest.main()
