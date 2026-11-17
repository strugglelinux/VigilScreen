"""离线验证数据隔离、类别检查和训练到预测的完整流程。"""
if __name__=='__main__':
    import sys
    from pathlib import Path as _TestPath
    sys.path.insert(0,str(_TestPath(__file__).resolve().parents[1]))

import contextlib
import io
from unittest.mock import patch
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
import torch
from torch.utils.data import Subset

from data import build_datasets, stratified_split
from model import ImageClassifier
from test import load_checkpoint, train, build_parser


ROOT = Path(__file__).resolve().parents[1]


def make_images(root: Path, per_class: int, classes=("cat", "dog")) -> None:
    """生成有空间变化的图片，使随机增强与验证预处理可以区分。"""
    for label, name in enumerate(classes):
        directory = root / name
        directory.mkdir(parents=True)
        for index in range(per_class):
            image = Image.new("RGB", (41, 35))
            image.putdata([
                ((x * 7 + label * 97) % 256, (y * 9 + index * 31) % 256, label * 170)
                for y in range(35) for x in range(41)
            ])
            image.save(directory / f"{index}.png")


class DatasetTests(unittest.TestCase):
    def test_zero_classes_are_rejected(self):
        with self.assertRaises(ValueError):
            ImageClassifier(0)

    def test_stratified_split_has_no_leakage_and_is_reproducible(self):
        targets = [0] * 9 + [1] * 3 + [2] * 2
        train, val = stratified_split(targets, val_ratio=0.2, seed=42)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(len(targets))))
        self.assertEqual(len(train) + len(val), len(targets))
        self.assertEqual({targets[i] for i in train}, {0, 1, 2})
        self.assertEqual({targets[i] for i in val}, {0, 1, 2})
        self.assertEqual((train, val), stratified_split(targets, 0.2, 42))

    def test_invalid_split_and_too_few_images_are_rejected(self):
        for ratio in (0, 1, -0.2, float("nan")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                stratified_split([0, 0, 1, 1], ratio, seed=42)
        with self.assertRaisesRegex(ValueError, "至少需要 2 张"):
            stratified_split([0, 0, 1], 0.2, seed=42)

    def test_folder_views_have_separate_transforms_and_disjoint_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_images(root, per_class=5)
            bundle = build_datasets("folder", root, seed=42)
            self.assertEqual(bundle.classes, ["cat", "dog"])
            self.assertIsNone(bundle.test)
            self.assertIsInstance(bundle.train, Subset)
            self.assertIsInstance(bundle.val, Subset)
            self.assertIsNot(bundle.train.dataset, bundle.val.dataset)
            self.assertIsNot(bundle.train.dataset.transform, bundle.val.dataset.transform)
            train_paths = {bundle.train.dataset.samples[i][0] for i in bundle.train.indices}
            val_paths = {bundle.val.dataset.samples[i][0] for i in bundle.val.indices}
            self.assertFalse(train_paths & val_paths)
            self.assertEqual(len(train_paths | val_paths), 10)
            first_image, first_label = bundle.val[0]
            self.assertEqual(tuple(first_image.shape), (3, 32, 32))
            for _ in range(3):
                image, label = bundle.val[0]
                self.assertTrue(torch.equal(first_image, image))
                self.assertEqual(first_label, label)

    def test_explicit_splits_require_matching_class_mapping(self):
        for mismatched_split in ("val", "test"):
            with self.subTest(split=mismatched_split), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_images(root / "train", per_class=2)
                make_images(root / "val", per_class=1,
                            classes=("cat", "horse") if mismatched_split == "val" else ("cat", "dog"))
                make_images(root / "test", per_class=1,
                            classes=("cat", "horse") if mismatched_split == "test" else ("cat", "dog"))
                with self.assertRaisesRegex(ValueError, "类别与训练集不一致"):
                    build_datasets("folder", root)


class CommandLineTests(unittest.TestCase):
    @staticmethod
    def run_cli(*arguments, expected_code=0):
        environment = os.environ.copy()
        environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        result = subprocess.run(
            [sys.executable, str(ROOT / "test.py"), *map(str, arguments)],
            cwd=ROOT, env=environment, text=True, capture_output=True, timeout=60,
        )
        if result.returncode != expected_code:
            raise AssertionError(
                f"命令退出码 {result.returncode}，预期 {expected_code}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def test_equal_validation_accuracy_selects_lower_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, output = root / 'images', root / 'output'
            make_images(images, per_class=3)
            args = build_parser().parse_args([
                'train', '--dataset', 'folder', '--data-dir', str(images),
                '--output-dir', str(output), '--epochs', '3', '--device', 'cpu',
            ])
            metrics = []
            for loss in (0.9, 0.2, 0.5):
                metrics.extend([{'loss': 1.0, 'accuracy': 0.5}, {'loss': loss, 'accuracy': 1.0}])
            with patch('test.run_epoch', side_effect=metrics), contextlib.redirect_stdout(io.StringIO()):
                train(args)
            checkpoint = torch.load(output / 'best_model.pt', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual(checkpoint['val_loss'], 0.2)

    def test_single_class_train_reload_evaluate_and_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, output = root / "images", root / "output"
            make_images(images, per_class=3, classes=("cat",))
            bundle = build_datasets("folder", images)
            self.assertEqual(bundle.classes, ["cat"])
            self.assertEqual(len(bundle.train), 2)
            self.assertEqual(len(bundle.val), 1)
            trained = self.run_cli(
                "train", "--dataset", "folder", "--data-dir", images,
                "--output-dir", output, "--epochs", 1, "--device", "cpu",
            )
            self.assertIn("Epoch 1/1", trained.stdout)
            self.assertIn("当前只有一个类别", trained.stdout)
            checkpoint = output / "best_model.pt"
            model, classes, transform = load_checkpoint(checkpoint, torch.device("cpu"))
            self.assertEqual(classes, ["cat"])
            with Image.open(images / "cat" / "0.png") as picture:
                with torch.inference_mode():
                    logits = model(transform(picture).unsqueeze(0))
            self.assertEqual(tuple(logits.shape), (1, 1))
            self.assertTrue(torch.isfinite(logits).all())
            history = json.loads((output / "history.json").read_text())
            self.assertEqual(history[0]["train"]["loss"], 0)
            evaluated = self.run_cli(
                "evaluate", "--dataset", "folder", "--data-dir", images,
                "--checkpoint", checkpoint, "--device", "cpu",
            )
            self.assertIn("评估图片：3", evaluated.stdout)
            predicted = self.run_cli(
                "predict", "--checkpoint", checkpoint, "--image", images / "cat" / "0.png",
                "--device", "cpu",
            )
            self.assertIn("cat: 100.00%", predicted.stdout)
            self.assertIn("准确率不代表识别能力", predicted.stdout)

    def test_cpu_train_save_reload_evaluate_and_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, output = root / "images", root / "output"
            for split, count in (("train", 3), ("val", 2), ("test", 2)):
                make_images(images / split, per_class=count)

            trained = self.run_cli(
                "train", "--dataset", "folder", "--data-dir", images,
                "--output-dir", output, "--epochs", 1, "--batch-size", 3,
                "--device", "cpu", "--seed", 42,
            )
            self.assertIn("Epoch 1/1", trained.stdout)
            self.assertIn("独立测试集", trained.stdout)
            checkpoint_path = output / "best_model.pt"
            self.assertTrue(checkpoint_path.is_file())
            history = json.loads((output / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["epoch"], 1)
            for split in ("train", "val"):
                self.assertTrue(math.isfinite(history[0][split]["loss"]))
                self.assertGreaterEqual(history[0][split]["accuracy"], 0)
                self.assertLessEqual(history[0][split]["accuracy"], 1)

            # 验证权重确实经过了优化，并且模型可在独立进程之外安全加载。
            torch.manual_seed(42)
            initial_model = ImageClassifier(2)
            model, classes, transform = load_checkpoint(checkpoint_path, torch.device("cpu"))
            self.assertEqual(classes, ["cat", "dog"])
            self.assertFalse(model.training)
            self.assertTrue(any(
                not torch.equal(initial, trained_parameter)
                for initial, trained_parameter in zip(initial_model.parameters(), model.parameters())
            ))
            with Image.open(images / "test" / "cat" / "0.png") as image:
                inputs = transform(image).unsqueeze(0)
            with torch.inference_mode():
                logits = model(inputs)
            self.assertEqual(tuple(logits.shape), (1, 2))
            self.assertTrue(torch.isfinite(logits).all())

            evaluated = self.run_cli(
                "evaluate", "--dataset", "folder", "--data-dir", images / "test",
                "--checkpoint", checkpoint_path, "--device", "cpu",
            )
            self.assertIn("评估图片：4", evaluated.stdout)

            grayscale = root / "grayscale.png"
            Image.new("L", (47, 39), color=100).save(grayscale)
            predicted = self.run_cli(
                "predict", "--checkpoint", checkpoint_path, "--image", grayscale,
                "--top-k", 99, "--device", "cpu",
            )
            self.assertIn("cat:", predicted.stdout)
            self.assertIn("dog:", predicted.stdout)
            self.assertEqual(predicted.stdout.count("%"), 2)

            mismatched = root / "mismatched"
            make_images(mismatched, per_class=1, classes=("cat", "horse"))
            rejected = self.run_cli(
                "evaluate", "--dataset", "folder", "--data-dir", mismatched,
                "--checkpoint", checkpoint_path, "--device", "cpu", expected_code=1,
            )
            self.assertIn("类别与训练集不一致", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
