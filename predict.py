# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    model.predict(source="ultralytics/assets", save=True)
