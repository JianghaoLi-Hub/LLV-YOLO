# LLV-YOLO

LLV-YOLO 是一种以 YOLO11n 为基线、面向导管架平台裂纹与变形检测的轻量化单阶段目标检测模型。模型结合轻量化网络结构重设计（LNSR）、LLVF 特征提取模块和 SA-CSPP 空间池化模块，以降低模型复杂度，并加强对损伤细节的特征表达。

[English README](README.md)

## 优势与适用场景

- **轻量化结构**：在两类检测、640×640 输入下，参数量为 1.78 M，计算量为 5.75 GFLOPs，低于原生 YOLO11n 的 2.59 M 和 6.44 GFLOPs。
- **面向损伤细节**：通过 LLVF 特征提取与 SA-CSPP 空间池化，结合局部损伤特征和空间信息。
- **适用场景**：导管架平台水上、水下图像中的裂纹与变形检测，以及使用相应标注数据训练后的类似结构表面损伤检测任务。

## 配置文件

| 文件 | 用途 |
|---|---|
| [configs/LLV-YOLO-11n.yaml](configs/LLV-YOLO-11n.yaml) | LLV-YOLO 模型结构 |
| [configs/yolo11n.yaml](configs/yolo11n.yaml) | YOLO11n 基线模型结构 |
| [ultralytics/nn/modules/llv_yolo.py](ultralytics/nn/modules/llv_yolo.py) | 自定义模块源码 |

## 训练

在仓库根目录运行：

```bash
python train.py
```

[train.py](train.py) 直接调用 Ultralytics Python 接口：

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("configs/LLV-YOLO-11n.yaml")
    model.train(data="coco128.yaml")
```

示例数据：[COCO128 官方说明](https://docs.ultralytics.com/datasets/detect/coco128/) · [下载地址](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip)。COCO128 的训练与验证使用同一批图像，仅用于检查流程；本仓库不包含 JPD 数据集。

## UIEC²-Net 离线数据增强

UIEC²-Net 请使用作者的 [UWEnhancement 官方仓库](https://github.com/BIGWangYuDong/UWEnhancement)，按照其说明安装并获取权重。本仓库不包含 UIEC²-Net 的代码与权重，训练、验证及推理入口不会自动调用该增强网络。

## 许可证

本项目基于 Ultralytics，遵循 [AGPL-3.0 许可证](LICENSE)。
