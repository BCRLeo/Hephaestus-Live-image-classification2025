from ultralytics import YOLO

# Create a YOLO model from the yolov8n.yaml configuration
model = YOLO('yolov8n.yaml')

# Save the model weights to yolov8r.pt
model.save('yolov8r.pt')

print("Model created and weights saved to yolov8r.pt")