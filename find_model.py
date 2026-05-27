import os

base = r"C:\Users\AGOLITO\Downloads\yolo\runs\detect"

for root, dirs, files in os.walk(base):
    if "best.pt" in files:
        print("FOUND:", os.path.join(root, "best.pt"))