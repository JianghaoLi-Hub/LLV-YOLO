# LLV-YOLO

LLV-YOLO is a lightweight, YOLO11n-based single-stage detector for crack and deformation detection on jacket platforms. It combines Lightweight Network Structure Redesign (LNSR), LLVF and SA-CSPP to reduce model complexity and enhance damage-detail feature extraction.

[中文说明](README.zh-CN.md)

## Advantages and applications

- **Lightweight:** with two classes and a 640×640 input, LLV-YOLO has 1.78 M parameters and 5.75 GFLOPs, compared with 2.59 M and 6.44 GFLOPs for native YOLO11n.
- **Damage-oriented features:** LLVF and SA-CSPP combine local damage features with spatial information.
- **Applications:** crack and deformation detection in above-water and underwater jacket-platform images, and related structural surface-damage tasks after training on suitable labeled data.

## Model files

| File | Purpose |
|---|---|
| [configs/LLV-YOLO-11n.yaml](configs/LLV-YOLO-11n.yaml) | LLV-YOLO architecture |
| [configs/yolo11n.yaml](configs/yolo11n.yaml) | YOLO11n baseline architecture |
| [ultralytics/nn/modules/llv_yolo.py](ultralytics/nn/modules/llv_yolo.py) | Custom modules |

## Training

Run `python train.py` from the repository root. [train.py](train.py) directly uses the Ultralytics Python API:

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("configs/LLV-YOLO-11n.yaml")
    model.train(data="coco128.yaml")
```

Demo dataset: [COCO128 documentation](https://docs.ultralytics.com/datasets/detect/coco128/) · [Download](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip). COCO128 uses the same images for training and validation and is only a pipeline demo. JPD is not included.

## UIEC²-Net offline augmentation

Use the authors' [official UWEnhancement repository](https://github.com/BIGWangYuDong/UWEnhancement) for installation and weights. This repository does not include UIEC²-Net code or weights, and the training, validation and prediction scripts do not invoke it automatically.

## License

This project is based on Ultralytics and retains its [AGPL-3.0 license](LICENSE) and source copyright notices. The upstream version and source archive are recorded in [UPSTREAM.json](UPSTREAM.json).
