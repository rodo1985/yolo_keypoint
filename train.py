from ultralytics import YOLO

model = YOLO('yolov8x-pose.pt')  # load a pretrained model (recommended for training)

model.train(data='config.yaml', epochs=50, imgsz=640, batch=8)
