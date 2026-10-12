# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("configs/LLV-YOLO-11n.yaml")  # Baseline: configs/yolo11n.yaml
    model.train(data="coco128.yaml")
