from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Predict with the model
results = model('image.png')

# Print results
print(results)
print("__" * 20)
results[0].show()

print(results[0].full_preds)

print(results[0].full_summary())