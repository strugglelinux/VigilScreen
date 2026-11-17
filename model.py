"""适用于 CIFAR-10 和自定义图片分类的小型卷积网络。"""

from collections.abc import Sequence

from torch import Tensor, nn
from torchvision import transforms

IMAGE_SIZE = 32
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class ImageClassifier(nn.Module):
    """输入 [batch, 3, 32, 32]，输出未经 softmax 的类别分数。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("图像分类至少需要一个类别。")
        blocks = []
        in_channels = 3
        for out_channels in (32, 64, 128):
            blocks.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((2, 2)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.features(images))


def build_transform(
    training: bool,
    image_size: int = IMAGE_SIZE,
    mean: Sequence[float] = MEAN,
    std: Sequence[float] = STD,
) -> transforms.Compose:
    """验证和预测共享确定性预处理，随机增强仅用于训练。"""
    operations = [transforms.Resize((image_size, image_size))]
    if training:
        operations.extend([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(operations)
