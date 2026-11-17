from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from download_models import prepare_models


class DownloadModelsTests(unittest.TestCase):
    def manifest(self,directory,content):
        Path(directory,'models.json').write_text(json.dumps({'format_version':1,'models':[
            {'filename':'model.onnx','url':'https://example.test/model.onnx','sha256':hashlib.sha256(content).hexdigest()}]}))

    def test_existing_correct_model_does_not_access_network(self):
        with tempfile.TemporaryDirectory() as directory:
            self.manifest(directory,b'correct')
            Path(directory,'model.onnx').write_bytes(b'correct')
            with patch('download_models.urlopen') as request:
                prepare_models(directory,check_only=True)
                request.assert_not_called()

    def test_download_verifies_and_replaces_file(self):
        with tempfile.TemporaryDirectory() as directory:
            self.manifest(directory,b'correct')
            with patch('download_models.urlopen',return_value=BytesIO(b'correct')):
                prepare_models(directory)
            self.assertEqual(Path(directory,'model.onnx').read_bytes(),b'correct')

    def test_bad_download_preserves_existing_file_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            self.manifest(directory,b'correct')
            Path(directory,'model.onnx').write_bytes(b'previous')
            with patch('download_models.urlopen',return_value=BytesIO(b'wrong')):
                with self.assertRaisesRegex(ValueError,'SHA-256'):
                    prepare_models(directory)
            self.assertEqual(Path(directory,'model.onnx').read_bytes(),b'previous')
            self.assertFalse(list(Path(directory).glob('*.tmp')))

    def test_check_missing_model_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            self.manifest(directory,b'correct')
            with patch('download_models.urlopen') as request:
                with self.assertRaises(ValueError):
                    prepare_models(directory,check_only=True)
                request.assert_not_called()


if __name__=='__main__':
    unittest.main()
