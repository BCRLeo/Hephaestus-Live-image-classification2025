import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms.functional as TF
from ultralytics.utils import IterableSimpleNamespace

model = YOLO("yolov8n.pt")
# model.model.requires_grad_()
model.model.train()
model.model.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

pil_image = Image.open(r"test_images/image.png")
torch_tensor = TF.to_tensor(pil_image)[np.newaxis, ...]
x = TF.resize(torch_tensor, [640, 640])
x.requires_grad_()

pred = model.model(x)
p = pred[0]
print(f"Pred shape: {p.shape}")
print(f"Pred device: {p.device}")
print(f"Pred dtype: {p.dtype}")
print(f"Pred requires_grad: {p.requires_grad}")
train_batch = {'cls': torch.zeros((0, 1)), 'bboxes': torch.zeros((0, 4)), 'batch_idx': torch.zeros(0)}
loss = model.model.loss(train_batch, pred)[0]
loss.backward()
print(x.grad)