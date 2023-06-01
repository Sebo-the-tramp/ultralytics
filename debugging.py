from ultralytics import YOLO

# Load a model
model = YOLO('yolov8.yaml')  # build a new model from YAML
#model = YOLO('yolov8n_phi.yaml')  # build a new model from YAML

# Train the model
model.train(data='coco128.yaml', epochs=1, imgsz=320)
