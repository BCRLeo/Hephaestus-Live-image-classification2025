import os, cv2, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO


# ------------------------------------------------------------------
# Helper: get flat predictions  (bs, anchors, 4 + n_cls)
# ------------------------------------------------------------------
def raw_preds(model: torch.nn.Module, imgs, *, detach: bool = True):
    """
    For Ultralytics‑YOLOv8 Detect.  Returns:
        (batch, anchors, 4 + n_cls)    — objectness column already removed.
    detach=True  ->   wrapped in no_grad() and .detach()
    detach=False ->   keeps autograd graph for back‑prop
    """
    was_training = model.training
    model.train()                # forward in train‑mode ⇒ raw head

    if detach:
        with torch.no_grad():
            p = model(imgs)[0]
    else:
        p = model(imgs)[0]

    if not was_training:
        model.eval()

    # ------------------------------------------ drop obj column (index 4)
    # p shape could be (bs, no, h, w) or (bs, anchors, no)
    if p.ndim == 4:                               # (bs, no, h, w)
        bs, ch, h, w = p.shape                    # ch = 5 + n_cls
        p = p.view(bs, ch, h * w).permute(0, 2, 1)  # (bs, anchors, no)
    elif p.ndim != 3:
        raise ValueError(f"Unexpected Detect output shape {p.shape}")

    # now p = (bs, anchors, 5 + n_cls)
    boxes_cls = torch.cat([p[..., :4], p[..., 5:]], dim=-1)  # drop obj

    return boxes_cls.detach() if detach else boxes_cls


# ------------------------------------------------------------------
# Dataset ------------------------------------------------------------------
class YOLOKDDataset(Dataset):
    def __init__(self, image_dir, teacher_yolo,
                 img_size=640,
                 device='mps' if torch.backends.mps.is_available() else 'cpu'):
        self.device = device
        self.img_size = img_size
        self.teacher = teacher_yolo
        self.image_paths = [os.path.join(image_dir, f)
                            for f in os.listdir(image_dir)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        # -------- pre‑compute teacher predictions  ------------------
        self.teacher.model.to(device).eval()
        self.teacher_preds = []
        for pth in self.image_paths:
            img = self._load_image(pth)
            feats = raw_preds(self.teacher.model, img.unsqueeze(0), detach=True)
            self.teacher_preds.append(feats.squeeze(0))        # (8400, 4+nc)
        self.teacher_preds = torch.stack(self.teacher_preds)   # (N, 8400, 4+nc)

    # Dataset API ---------------------------------------------------
    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])
        return img.to(self.device), self.teacher_preds[idx]

    # helper --------------------------------------------------------
    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return torch.tensor(img).float().permute(2, 0, 1) / 255.0


# ------------------------------------------------------------------
# KD‑loss ------------------------------------------------------------------
class YOLOKDLoss(nn.Module):
    def __init__(self, box_w=1.0, cls_w=1.0, conf_th=0.01):
        super().__init__()
        self.bw, self.cw, self.th = box_w, cls_w, conf_th

    def forward(self, s, t):
        print("student:", s.shape, "teacher:", t.shape)  # sanity print

        s_box, s_cls = s[..., :4], s[..., 4:]
        t_box, t_cls = t[..., :4], t[..., 4:]

        mask = t_cls.max(dim=-1)[0] > self.th

        box_loss = F.mse_loss(s_box[mask], t_box[mask]) * self.bw

        T = 100.0
        cls_loss = F.kl_div(
            F.log_softmax(s_cls[mask] / T, dim=-1),
            F.softmax(t_cls[mask] / T, dim=-1),
            reduction='batchmean') * self.cw * T**2

        print(f"box {box_loss.item():.4f}  cls {cls_loss.item():.4f}")
        return box_loss + cls_loss


# ------------------------------------------------------------------
# Training loop ----------------------------------------------------
def train_distillation(teacher_w='yolov8n.pt', student_cfg='yolov8n.yaml',
                       data_dir='test_images', epochs=5, batch=8):

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    teacher = YOLO(teacher_w)
    student = YOLO(student_cfg)

    ds = YOLOKDDataset(data_dir, teacher, device=device)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    criterion = YOLOKDLoss(box_w=1.0, cls_w=5.0)
    optim = torch.optim.AdamW(student.model.parameters(), lr=1e-4)

    student.model.custom_train = True
    student.model.train().to(device)
    teacher.model.eval().to(device)

    for ep in range(1, epochs + 1):
        tot = 0.0
        for imgs, tgt in dl:
            imgs, tgt = imgs.to(device), tgt.to(device)

            s_pred = raw_preds(student.model, imgs, detach=False)  # keep grad
            assert s_pred.shape == tgt.shape

            loss = criterion(s_pred, tgt)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optim.step()
            tot += loss.item()

        print(f"Epoch {ep}/{epochs}  mean_loss {tot/len(dl):.4f}")

    student.save('distilled_yolov8.pt')
    print("✓ Distillation complete → distilled_yolov8.pt")


# ------------------------------------------------------------------
# Run --------------------------------------------------------------
if __name__ == "__main__":
    train_distillation(
        teacher_w='yolov8n.pt',
        student_cfg='yolov8n.yaml',
        data_dir=r'C:\Users\leona\AppData\Local\Programs\Python\Hephaestus-Live-image-classification2025\Niko\ultralytics-8.3.100\test_images',
        epochs=5,
        batch=8
    )
