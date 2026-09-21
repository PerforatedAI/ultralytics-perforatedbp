# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    cfg = Path(__file__).parent / "yolo26n_cityscapes_pai.yaml"
    YOLO("yolo26n.pt").train(cfg=str(cfg), perforate=True, name="yolo26n_cityscapes_pai")
