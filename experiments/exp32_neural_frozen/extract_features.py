"""Extract frozen CIFAR-10 features from an ImageNet-pretrained ResNet-18
(penultimate 512-d layer), on MPS. Saves features.npz."""
import numpy as np
import torch
import torchvision
from torchvision import transforms
from pathlib import Path

HERE = Path(__file__).resolve().parent
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Identity()
model.eval().to(dev)

out = {}
for split, train in (("train", True), ("test", False)):
    ds = torchvision.datasets.CIFAR10(str(HERE / "data"), train=train,
                                      download=True, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=4)
    feats, labels = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            feats.append(model(x.to(dev)).cpu().numpy())
            labels.append(y.numpy())
            if i % 20 == 0:
                print(split, i * 256, flush=True)
    out[f"X_{split}"] = np.concatenate(feats)
    out[f"y_{split}"] = np.concatenate(labels)
np.savez_compressed(HERE / "features.npz", **out)
print("saved", {k: v.shape for k, v in out.items()})
