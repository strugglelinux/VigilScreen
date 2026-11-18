"""标记副本、重新分类与原图保护。"""
if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

from pathlib import Path
import tempfile
import unittest
from PIL import Image
from labeling import LabelStore,existing_categories
from data import build_datasets


class LabelingTests(unittest.TestCase):
    def test_existing_categories_merge_directories_and_manifest_without_modifying_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('chi','me','.cache','train','val','test'):
                (root/name).mkdir()
            manifest=root/'.labels.json'
            content='{"one":"chi","two":"旧类别","bad":[],"empty":"","reserved":"train"}'
            manifest.write_text(content,encoding='utf-8')
            self.assertEqual(existing_categories(root),['chi','me','旧类别'])
            self.assertEqual(manifest.read_text(encoding='utf-8'),content)
            manifest.write_text('invalid json',encoding='utf-8')
            self.assertEqual(existing_categories(root),['chi','me'])
            missing=root/'missing'
            self.assertEqual(existing_categories(missing),[])
            self.assertFalse(missing.exists())

    def test_crop_persists_and_source_is_preserved_when_relabeling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / 'source.png'
            picture = Image.new('RGB', (800, 400), 'blue')
            picture.paste('red', (200, 100, 600, 300))
            picture.save(source)
            original = source.read_bytes()
            store = LabelStore(root / 'labeled')
            target = store.save(source, 'cat', box=(200, 100, 600, 300), max_side=128)
            with Image.open(target) as cropped:
                self.assertEqual(cropped.size, (128, 64))
                self.assertEqual(cropped.getpixel((0, 0)), (255, 0, 0))
            reopened = LabelStore(store.root)
            region = reopened.regions[str(source)]
            self.assertEqual(region, {'box': [200, 100, 600, 300], 'max_side': 128})
            renamed = reopened.save(source, 'dog', box=region['box'], max_side=region['max_side'])
            self.assertFalse(target.exists())
            with Image.open(renamed) as cropped:
                self.assertEqual(cropped.size, (128, 64))
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(ValueError):
                reopened.save(source, 'dog', box=(-1, 0, 40, 40))
            self.assertEqual(source.read_bytes(), original)

    def test_relabel_persist_and_train_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            store = LabelStore(root / 'labeled')
            for index in range(4):
                source = root / f'{index}.jpg'
                Image.new('RGB', (40, 40), (index * 40, 0, 0)).save(source)
                original = source.read_bytes()
                store.save(source, '临时')
                target = store.save(source, '猫' if index < 2 else '狗')
                self.assertTrue(target.exists())
                self.assertEqual(source.read_bytes(), original)
            self.assertFalse((store.root / '临时').exists())
            self.assertEqual(LabelStore(store.root).labels, store.labels)
            bundle = build_datasets('folder', store.root)
            self.assertEqual(len(bundle.train), 2)
            self.assertEqual(len(bundle.val), 2)
            for invalid in ('', '..', '../cat', 'a/b', 'train'):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    store.save(source, invalid)


if __name__=='__main__':
    unittest.main()
