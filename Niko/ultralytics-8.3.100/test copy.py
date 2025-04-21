from ultralytics import YOLO
import torch
import cv2

def load_image(path, img_size=640):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = torch.tensor(img).float().permute(2, 0, 1) / 255.0
        return img
    
# model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8 model

# result = model('test_images/image.png')

# print(result[0].raw_predictions.shape)

# print("_"*50)

device = 'mps' if torch.backends.mps.is_available() else 'cuda'
model = YOLO('yolov8n.yaml').to(device)

img1 = load_image('test_images/image.png').unsqueeze(0).to(device)

# batch = torch.stack([img1, img2]).to(device)  # Shape: [2, 3, 640, 640]


model.model.train().to(device)

# Perform forward pass
result = model.model.forward(img1)

for res in result:
    print(f"Result shape: {res.shape}")
    print(f"Result device: {res.device}")
    print(f"Result dtype: {res.dtype}")
    print(f"Result requires_grad: {res.requires_grad}")
    


# # Try backpropagation
# try:
#     dummy_loss = result[0].sum()  # Compute a dummy loss
#     dummy_loss.backward()  # Attempt backpropagation
#     print("Backpropagation successful!")
# except RuntimeError as e:
#     print(f"Backpropagation failed: {e}")