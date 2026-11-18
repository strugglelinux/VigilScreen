"""PyTorch 图像分类入口：train / evaluate / predict。"""

import argparse
import json
import math
import random
from pathlib import Path

from runtime_support import (configure_process_environment,cpu_thread_budget,
                             native_architecture)
import sys

configure_process_environment()

import torch
torch.set_num_threads(cpu_thread_budget())
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader

from data import build_datasets, build_evaluation_dataset, check_classes
from model import IMAGE_SIZE, MEAN, STD, ImageClassifier, build_transform

from runtime_paths import data_root
ROOT = data_root()


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if sys.platform=='darwin' and native_architecture()=='arm64'
            and getattr(torch.backends,'mps',None) is not None and torch.backends.mps.is_available() else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请选择 --device cpu 或 auto。")
    if name == "mps" and (getattr(torch.backends,'mps',None) is None or not torch.backends.mps.is_available()):
        raise ValueError("MPS 不可用，请选择 --device cpu 或 auto。")
    return torch.device(name)


def run_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, count = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            size = labels.size(0)
            loss_sum += loss.item() * size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += size
    if count == 0:
        raise ValueError("数据集为空，无法训练或评估。")
    return {"loss": loss_sum / count, "accuracy": correct / count}


def load_checkpoint(path: Path, device: torch.device):
    # 先加载到 CPU，兼容在 CUDA/MPS 设备上训练后保存的权重。
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("模型文件格式不支持，请使用本项目的 train 命令生成 checkpoint。")
    classes = checkpoint["classes"]
    if not isinstance(classes, list) or len(classes) < 1 or not all(
        isinstance(name, str) for name in classes
    ) or len(set(classes)) != len(classes):
        raise ValueError("模型文件的类别信息无效。")
    model = ImageClassifier(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    preprocess = checkpoint["preprocess"]
    transform = build_transform(training=False, **preprocess)
    return model, classes, transform


def report_single_class(classes: list[str]) -> None:
    if len(classes) == 1:
        print(
            "提示：当前只有一个类别，softmax 输出恒为 100%，交叉熵损失为 0；"
            "该训练不能学习区分其他图片，准确率不代表识别能力。"
            "如需区分目标与其他图片，请增加其他类别的样本。", flush=True,
        )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    bundle = build_datasets(
        args.dataset, args.data_dir, args.download, args.val_ratio, args.seed
    )
    report_single_class(bundle.classes)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        bundle.train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=generator,
    )
    val_loader = DataLoader(
        bundle.val, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = ImageClassifier(len(bundle.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "history.json"
    best_accuracy = -1.0
    best_loss = float("inf")
    history = []
    print(f"设备：{device} | 类别：{bundle.classes}", flush=True)
    print(
        f"训练集：{len(bundle.train)} | 验证集：{len(bundle.val)} | "
        f"测试集：{len(bundle.test) if bundle.test is not None else 0}", flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if (val_metrics["accuracy"] > best_accuracy or (
            val_metrics["accuracy"] == best_accuracy and val_metrics["loss"] < best_loss
        )):
            best_accuracy = val_metrics["accuracy"]
            best_loss = val_metrics["loss"]
            torch.save({
                "format_version": 1,
                "model_state_dict": model.state_dict(),
                "classes": bundle.classes,
                "preprocess": {"image_size": IMAGE_SIZE, "mean": list(MEAN), "std": list(STD)},
                "epoch": epoch,
                "val_accuracy": best_accuracy,
                "val_loss": best_loss,
                "history": history,
                "train_config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            }, checkpoint_path)
        temporary_history = history_path.with_suffix(".tmp")
        temporary_history.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_history.replace(history_path)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"训练 loss={train_metrics['loss']:.4f}, 准确率={train_metrics['accuracy']:.2%} | "
            f"验证 loss={val_metrics['loss']:.4f}, 准确率={val_metrics['accuracy']:.2%}",
            flush=True,
        )
    print(f"最佳验证准确率：{best_accuracy:.2%}；模型已保存：{checkpoint_path}")
    if bundle.test is not None:
        model, _, _ = load_checkpoint(checkpoint_path, device)
        test_loader = DataLoader(
            bundle.test, batch_size=args.batch_size, num_workers=args.num_workers
        )
        metrics = run_epoch(model, test_loader, device)
        print(f"独立测试集 loss={metrics['loss']:.4f}, 准确率={metrics['accuracy']:.2%}")


def evaluate(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    model, classes, transform = load_checkpoint(args.checkpoint, device)
    report_single_class(classes)
    dataset = build_evaluation_dataset(args.dataset, args.data_dir, args.download, transform)
    check_classes(dataset.classes, classes, "评估数据")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    metrics = run_epoch(model, loader, device)
    print(f"评估图片：{len(dataset)} | loss={metrics['loss']:.4f}, 准确率={metrics['accuracy']:.2%}")


def predict(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    model, classes, transform = load_checkpoint(args.checkpoint, device)
    report_single_class(classes)
    with Image.open(args.image) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        inputs = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(inputs).softmax(dim=1)[0]
        scores, indices = probabilities.topk(min(args.top_k, len(classes)))
    print(f"图片：{args.image}")
    for score, index in zip(scores.tolist(), indices.tolist()):
        print(f"{classes[index]}: {score:.2%}")


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数。")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("不能为负数。")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须为有限正数。")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch 图像分类：训练、评估和单张图片预测。")
    commands = parser.add_subparsers(dest="command")
    train_parser = commands.add_parser("train", help="训练 CNN 并保存最佳模型")
    evaluate_parser = commands.add_parser("evaluate", help="在有标签的图片集上评估模型")
    predict_parser = commands.add_parser("predict", help="预测单张图片的类别")
    for command in (train_parser, evaluate_parser, predict_parser):
        command.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    for command in (train_parser, evaluate_parser):
        command.add_argument("--dataset", choices=("cifar10", "folder"), default="cifar10")
        command.add_argument("--data-dir", type=Path, default=ROOT / "data")
        command.add_argument("--download", action="store_true", help="允许下载 CIFAR-10 数据")
        command.add_argument("--batch-size", type=positive_int, default=64)
        command.add_argument("--num-workers", type=nonnegative_int, default=0)
    train_parser.add_argument("--epochs", type=positive_int, default=20)
    train_parser.add_argument("--learning-rate", type=positive_float, default=0.001)
    train_parser.add_argument("--val-ratio", type=float, default=0.2)
    train_parser.add_argument("--seed", type=nonnegative_int, default=42)
    train_parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints")
    for command in (evaluate_parser, predict_parser):
        command.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best_model.pt")
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument("--top-k", type=positive_int, default=3)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    try:
        if args.command == "train" and not 0 < args.val_ratio < 1:
            raise ValueError("--val-ratio 必须大于 0 且小于 1。")
        {"train": train, "evaluate": evaluate, "predict": predict}[args.command](args)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        hint = ""
        if args.command in ("train", "evaluate") and args.dataset == "cifar10" and not args.download:
            hint = "\n首次使用 CIFAR-10 请加 --download；自有图片请加 --dataset folder --data-dir 路径。"
        parser.exit(1, f"错误：{error}{hint}\n")


if __name__ == "__main__":
    main()
