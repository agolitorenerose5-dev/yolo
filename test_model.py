from ultralytics import YOLO

model = YOLO(r"C:\Users\AGOLITO\Downloads\yolo\runs\detect\train-4\weights\best.pt")

# Run detection on an image
results = model("test.jpg", save=True)

# Print results
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(model.names[cls], conf)
