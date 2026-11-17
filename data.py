"""数据加载与分层划分；验证集和测试集不使用训练增强。"""

import random
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, ImageFolder

from model import build_transform


@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset | None
    classes: list[str]


def stratified_split(
    targets: list[int], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """每类至少保留一张训练图和一张验证图，且索引互斥。"""
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio 必须大于 0 且小于 1。")
    grouped: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        grouped.setdefault(target, []).append(index)
    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for target, indices in sorted(grouped.items()):
        if len(indices) < 2:
            raise ValueError(
                f"类别索引 {target} 只有 {len(indices)} 张图片；自动划分时每类至少需要 2 张。"
                "也可提供独立的 train/val 文件夹。"
            )
        rng.shuffle(indices)
        val_count = max(1, min(len(indices) - 1, round(len(indices) * val_ratio)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    return train_indices, val_indices


def check_classes(actual: list[str], expected: list[str], name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} 的类别与训练集不一致：预期 {expected}，实际 {actual}。")


def build_datasets(
    dataset: str,
    data_dir: Path,
    download: bool = False,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    train_transform = build_transform(training=True)
    eval_transform = build_transform(training=False)
    if dataset == "cifar10":
        # 对同一训练数据建立两个视图，避免 Subset 共享随机增强。
        train_view = CIFAR10(data_dir, train=True, download=download, transform=train_transform)
        val_view = CIFAR10(data_dir, train=True, download=False, transform=eval_transform)
        test = CIFAR10(data_dir, train=False, download=download, transform=eval_transform)
        train_indices, val_indices = stratified_split(train_view.targets, val_ratio, seed)
        return DatasetBundle(
            Subset(train_view, train_indices), Subset(val_view, val_indices),
            test, list(train_view.classes),
        )
    if dataset != "folder":
        raise ValueError(f"不支持的数据集：{dataset}")
    if not data_dir.is_dir():
        raise ValueError(f"数据目录不存在：{data_dir}")

    has_splits = (data_dir / "train").is_dir()
    train_dir = data_dir / "train" if has_splits else data_dir
    train_view = ImageFolder(train_dir, transform=train_transform)
    if len(train_view.classes) < 1:
        raise ValueError("训练数据至少需要一个类别文件夹。")
    classes = list(train_view.classes)
    if has_splits and (data_dir / "val").is_dir():
        train = train_view
        val = ImageFolder(data_dir / "val", transform=eval_transform)
        check_classes(val.classes, classes, "验证集")
    else:
        val_view = ImageFolder(train_dir, transform=eval_transform)
        train_indices, val_indices = stratified_split(train_view.targets, val_ratio, seed)
        train, val = Subset(train_view, train_indices), Subset(val_view, val_indices)

    test = None
    if has_splits and (data_dir / "test").is_dir():
        test = ImageFolder(data_dir / "test", transform=eval_transform)
        check_classes(test.classes, classes, "测试集")
    return DatasetBundle(train, val, test, classes)


def build_evaluation_dataset(
    dataset: str, data_dir: Path, download: bool = False, transform=None
) -> Dataset:
    """folder 的 data_dir 直接指向待评估的类别目录，例如 my_images/test。"""
    if transform is None:
        transform = build_transform(training=False)
    if dataset == "cifar10":
        return CIFAR10(data_dir, train=False, download=download, transform=transform)
    if dataset == "folder":
        return ImageFolder(data_dir, transform=transform)
    raise ValueError(f"不支持的数据集：{dataset}")
