import os
import glob
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from .nima.model import *

test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])

base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_state(model_path=None):
    if not model_path:
        model_path = glob.glob(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '*.pkl')
        )[0]
    global model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()


def load_image(img):
    im = Image.open(img) if isinstance(img, str) else img
    if len(im.getbands()) != 3:
        im2 = Image.new('RGB', im.size)
        im2.paste(im)
        return im2
    return im


def predict(images):
    for img in images:
        try:
            im = load_image(img)
            imt = test_transform(im)
            imt = imt.unsqueeze(dim=0)
            imt = imt.to(device)
            with torch.no_grad():
                out = model(imt)
            out = out.view(10, 1)
            mean = 0
            for j, e in enumerate(out, 1):
                mean += j * e.item()
            yield (img, mean)
        except Exception:
            yield img, -1
