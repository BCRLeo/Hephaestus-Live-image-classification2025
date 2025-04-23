import os
import cv2
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO


# --------------------------------------------------------------
# Helper: raw YOLO head  →  (B, 8400, 4 + nc)  (obj column removed)
# --------------------------------------------------------------
def raw_preds(model: torch.nn.Module, imgs, detach: bool = False):
    """
    Works for Ultralytics‑YOLOv8 Detect models.

    Args
    ----
    model   : the inner .model of a YOLO object
    imgs    : (B, 3, H, W) tensor
    detach  : if True, wraps the forward pass in torch.no_grad()
              and returns a detached tensor (use for teacher)
    """
    ctx = torch.no_grad() if detach else torch.enable_grad()
    with ctx:
        was_training = model.training
        model.train()                                # raw head, no NMS
        p = model(imgs)[0]                           # raw output tensor
        model.train(mode=was_training)               # restore mode

    # ----------------------------------------------------------
    # Bring to (B, anchors, no)  where no = 5 + nc
    # ----------------------------------------------------------
    if p.ndim == 4:                                  # (B, no, ny, nx)
        bs, no, ny, nx = p.shape
        p = p.view(bs, no, -1).permute(0, 2, 1)      # (B, anchors, no)
    elif p.ndim == 3 and p.shape[-1] >= 6:           # already good
        pass
    elif p.ndim == 3 and p.shape[1] >= 6:            # (B, no, anchors)
        p = p.permute(0, 2, 1)
    else:
        raise ValueError(f"Unexpected head shape {p.shape}")

    # ----------------------------------------------------------
    # Drop objectness column (index 4)
    # ----------------------------------------------------------
    p = torch.cat([p[..., :4], p[..., 5:]], dim=-1)  # (B, anchors, 4 + nc)

    return p.detach() if detach else p


# --------------------------------------------------------------
# Dataset
# --------------------------------------------------------------
class YOLOKDDataset(Dataset):
    def __init__(self, image_dir, teacher_yolo,
                 img_size=640,
                 device='mps' if torch.backends.mps.is_available() else 'cpu',
                 max_images=None):
        self.device = device
        self.img_size = img_size
        self.teacher = teacher_yolo

        self.paths = [os.path.join(image_dir, f)
                      for f in os.listdir(image_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not self.paths:
            raise ValueError(f"No images found in {image_dir}")
        
        if max_images is not None:
            self.paths = random.sample(self.paths, max_images)
            print(f"✓ Using {len(self.paths)} images from {image_dir}")

        # Pre‑compute teacher predictions
        self.teacher.model.to(device).eval()
        feats = []
        for i, p in enumerate(self.paths):
            img = self._load(p).unsqueeze(0).to(device)
            feats.append(raw_preds(self.teacher.model, img, detach=True)
                         .squeeze(0))                 # (8400, 4+nc)
            print(f"\r✓ {i+1}/{len(self.paths)}  images loaded", end='')
            
        print(f'\nLoaded {len(self.paths)} images from {image_dir}')
        self.teacher_preds = torch.stack(feats)        # (N, 8400, 4+nc)

    # PyTorch API
    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        return self._load(self.paths[idx]).to(self.device), \
               self.teacher_preds[idx]

    # helper
    def _load(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return torch.tensor(img).float().permute(2, 0, 1) / 255.0


# --------------------------------------------------------------
# KD loss
# --------------------------------------------------------------
class YOLOKDLoss(nn.Module):
    def __init__(self, box_w=1.0, cls_w=1.0, conf_th=0.01):
        super().__init__()
        self.box_w = box_w
        self.cls_w = cls_w
        self.conf_th = conf_th

    def forward(self, s, t):
        print("student:", s.shape, "teacher:", t.shape)

        sb, sc = s[..., :4], s[..., 4:]
        tb, tc = t[..., :4], t[..., 4:]
        mask = tc.max(-1).values > self.conf_th

        box_loss = F.mse_loss(sb[mask], tb[mask]) * self.box_w

        T = 100.0
        cls_loss = F.kl_div(
            F.log_softmax(sc[mask] / T, -1),
            F.softmax(tc[mask] / T, -1),
            reduction='batchmean') * self.cls_w * T ** 2

        print(f"box {box_loss.item():.4f}  cls {cls_loss.item():.4f}")
        return box_loss + cls_loss


# --------------------------------------------------------------
# Training loop
# --------------------------------------------------------------
def train_distillation(teacher_w='yolov8n.pt',
                       student_cfg='yolov8n.yaml',
                       data_dir='test_images',
                       epochs=5,
                       batch_size=8):

    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    teacher = YOLO(teacher_w)
    student = YOLO(student_cfg)

    ds = YOLOKDDataset(data_dir, teacher, device=dev, max_images=100)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    loss_fn = YOLOKDLoss(box_w=1.0, cls_w=5.0)
    opt = torch.optim.AdamW(student.model.parameters(), lr=1e-4)

    student.model.custom_train = True
    student.model.train().to(dev)
    teacher.model.eval().to(dev)

    for ep in range(epochs):
        tot = 0.0
        for imgs, tgt in dl:
            imgs, tgt = imgs.to(dev), tgt.to(dev)

            spred = raw_preds(student.model, imgs)      # keeps grad_fn
            assert spred.shape == tgt.shape

            loss = loss_fn(spred, tgt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            tot += loss.item()

        print(f"Epoch {ep+1}/{epochs}  mean_loss {tot/len(dl):.4f}")

    student.save('distilled_yolov8.pt')
    print("✓ Distillation complete → distilled_yolov8.pt")


# --------------------------------------------------------------
# Entry‑point
# --------------------------------------------------------------
if __name__ == "__main__":
    train_distillation(
        teacher_w='yolov8n.pt',
        student_cfg='yolov8n.yaml',
        data_dir=r'Niko/ultralytics-8.3.100/test2014',
        epochs=5,
        batch_size=8
    )
