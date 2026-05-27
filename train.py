from ultralytics import YOLO

# 1. Load model
model = YOLO("yolov8n.pt")

# 2. Train
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)

# 3. Validate (metrics)
metrics = model.val()

print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP@50:", metrics.box.map50)
print("mAP@50-95:", metrics.box.map)