import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
IMG_SIZE = 224
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats —
                          std=[0.229, 0.224, 0.225]),   # required since we reuse its pretrained weights
])

def build_model(device, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False   # freeze everything...
    model.fc = nn.Linear(model.fc.in_features, 2)  # ...then replace fc (new layer is trainable by default)
    return model.to(device)

def train_model(train_df, device, epochs=8, lr=1e-4, batch_size=16):
    from dataset import BottleDataset
    ds = BottleDataset(train_df, transform=TRANSFORM)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = build_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in dl:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
        print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/len(ds):.4f}")
    return model