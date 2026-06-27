from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import os
 
def main():
    model = YOLO("yolo26x.pt")
 
    dataset_yaml = '/home/daft/Documents/detection people/data2/data.yaml'
 
    run_name = f"yolo26n_person_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
 
    model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=os.cpu_count(),
        classes=[0],
        project="danger_zone",
        name=run_name,
        exist_ok=False,
    )
 
    best = Path("danger_zone") / run_name / "weights" / "best.pt"
    print(f"\n✅ Готово! Лучшие веса: {best.resolve()}")
 
if __name__ == "__main__":
    main()
