from ultralytics import YOLO

model = YOLO('distilled_yolov8.pt')

img = 'test_images/image.png'  # Path to your image

results = model(img)

results[0].show()