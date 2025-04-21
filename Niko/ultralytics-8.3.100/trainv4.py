import torch
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch.nn.functional as F

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

teacher = YOLO('yolov8n.pt').to(device)
teacher.model.eval().to(device)

student = YOLO('yolov8n.pt').to(device)
student.model.train().to(device)

pil_image = Image.open(r"test_images/image.png")
torch_tensor = TF.to_tensor(pil_image)[np.newaxis, ...]
x = TF.resize(torch_tensor, [640, 640]).to(device)
x.requires_grad_()

# Perform forward pass
with torch.no_grad():
    teacher_preds = teacher.model(x)[0].to(device)
    teacher_preds = teacher_preds.transpose(1, 2)
    print(f"Teacher preds shape: {teacher_preds.shape}")

# Perform forward pass
student_preds = student.model(x)[0].to(device)
student_preds = student_preds.transpose(1, 2)
print(f"Student preds shape: {student_preds.shape}")

# Compute loss
loss = F.mse_loss(student_preds, teacher_preds)
print(f"Loss: {loss.item()}")

# Backpropagation
loss.backward()
print(f"Gradient shape: {x.grad.shape}")
print(f"Gradient device: {x.grad.device}")
print(f"Gradient dtype: {x.grad.dtype}")
