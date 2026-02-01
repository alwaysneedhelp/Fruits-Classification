import cv2
import torch
from ultralytics import YOLO
from torchvision import models, transforms
import os
import torch.nn.functional as F
import torch.nn as nn

# Load models
yolo = YOLO("models/yolo_best.pt")


resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(2048, 116)

state_dict = torch.load("models/resnet50.pth", map_location="cpu")
resnet.load_state_dict(state_dict)

resnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

# -------- reconstruct ResNet class names (ImageFolder logic) --------
RESNET_TRAIN_DIR = "fruits/fruits-360_original-size/fruits-360-original-size/Training"

# ImageFolder sorts folder names alphabetically
RESNET_CLASSES = sorted([
    d for d in os.listdir(RESNET_TRAIN_DIR)
    if os.path.isdir(os.path.join(RESNET_TRAIN_DIR, d))
])

# -------- YOLO class names --------
YOLO_CLASSES = yolo.names  # id -> name


def detect_crop_classify(image_path):
    img = cv2.imread(image_path)
    results = yolo(img)

    best = None
    best_conf = -1.0
    best_yolo_cls = None

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best = box
                best_yolo_cls = int(box.cls[0])

    if best is None:
        return None

    # -------- crop --------
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    crop = img[y1:y2, x1:x2]

    tensor = transform(crop).unsqueeze(0)

    # -------- ResNet inference --------
    with torch.no_grad():
        logits = resnet(tensor)
        probs = F.softmax(logits, dim=1)

    # -------- constrain ResNet using YOLO --------
    yolo_name = YOLO_CLASSES[best_yolo_cls]

    # allow only ResNet classes that start with YOLO class name
    allowed_idxs = [
        i for i, name in enumerate(RESNET_CLASSES)
        if name.lower().startswith(yolo_name.lower())
    ]

    if allowed_idxs:
        mask = torch.zeros_like(probs)
        mask[:, allowed_idxs] = 1.0
        probs = probs * mask

    pred_idx = probs.argmax(dim=1).item()
    pred_name = RESNET_CLASSES[pred_idx]

    return {
        "bbox": (x1, y1, x2, y2),
        "yolo_class": yolo_name,
        "resnet_class": pred_name,
        "confidence": best_conf,
    }


print(detect_crop_classify('manual_test/test_banana.jpg'))