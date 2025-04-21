import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import cv2

class YOLOKDDataset(Dataset):
    def __init__(self, image_dir, teacher_model, img_size=640, device= 'mps' if torch.backends.mps.is_available() else 'cuda'):
        self.device = device
        self.image_dir = image_dir
        self.img_size = img_size
        self.teacher = teacher_model
        self.image_paths = [
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not self.image_paths:
            raise ValueError(f"No valid image files found in directory: {self.image_dir}")
        
        # Precompute teacher predictions
        self.teacher_preds = []
        with torch.no_grad():
            for img_path in self.image_paths:
                img = self.load_image(img_path)
                results = self.teacher(img)
                self.teacher_preds.append(results[0].raw_predictions)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = self.load_image(img_path)
        return img.to(self.device), self.teacher_preds[idx].to(self.device)

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.tensor(img).float().permute(2, 0, 1) / 255.0
        return img

class YOLOKDLoss(nn.Module):
    def __init__(self, box_weight=1.0, cls_weight=1.0, conf_thresh=0.01):
        super().__init__()
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.conf_thresh = conf_thresh

    def forward(self, student_preds, teacher_preds):

        # Split predictions
        s_boxes = student_preds[:, :, :4]
        s_cls = student_preds[:, :, 4:]
        
        t_boxes = teacher_preds[:, :, :4].to(student_preds.device)
        t_cls = teacher_preds[:, :, 4:].to(student_preds.device)

        # Create confidence mask
        t_conf = t_cls.max(dim=-1)[0]
        mask = t_conf > self.conf_thresh

        # Box loss (MSE)
        box_loss = F.mse_loss(s_boxes[mask], t_boxes[mask]) * self.box_weight

        # Classification loss (KL divergence)
        cls_loss = F.kl_div(
            F.log_softmax(s_cls[mask], dim=-1),
            F.softmax(t_cls[mask], dim=-1),
            reduction='batchmean'
        ) * self.cls_weight

        return box_loss + cls_loss

def train_distillation(
    teacher_weights='yolov8n.pt',
    student_config='yolov8n.yaml',
    data_dir='test_images',
    epochs=5,
    batch_size=8
):
    device = 'mps' if torch.backends.mps.is_available() else 'cuda'
    # Initialize models
    teacher = YOLO(teacher_weights).to(device)
    student = YOLO(student_config).to(device)
    
    # Create dataset
    dataset = YOLOKDDataset(data_dir, teacher, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = YOLOKDLoss(box_weight=1.0, cls_weight=0.5)
    optimizer = torch.optim.AdamW(student.model.parameters(), lr=1e-4)

    # Training loop
    student.model.train().to(device)
    
    teacher.model.eval().to(device)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for images, targets in loader:
            images = images
            targets = targets

            student_preds = []
            for image in images:
                image = image.unsqueeze(0)
                preds = student.model.predict(image)
                print(len(preds), preds[0].shape)
                student_preds.append(preds[0].raw_predictions)

            # Stack predictions into a single tensor
            student_preds = torch.stack(student_preds)
            print(f"student_preds.requires_grad: {student_preds.requires_grad}, targets.requires_grad: {targets.requires_grad}")
            
            # Compute loss
            for student_pred, target in zip(student_preds, targets):
                with torch.set_grad_enabled(True):
                    loss = criterion(student_pred, target)
                    print(f"loss.grad_fn: {loss.grad_fn}")
                    
                    # Optimize
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}')

    # Save distilled model
    student.save('distilled_yolov8.pt')

if __name__ == '__main__':
    train_distillation()