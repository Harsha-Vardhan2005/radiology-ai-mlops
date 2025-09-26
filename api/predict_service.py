# api/predict_service.py
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_learning import get_resnet50, get_vit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # adjust if needed

def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

def load_model(model_type="baseline", checkpoint_path=None):
    if model_type == "baseline":
        model = BaselineCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    elif model_type == "resnet50":
        model, _, _ = get_resnet50(num_classes=len(CLASS_NAMES), device=DEVICE)
    elif model_type == "vit":
        model, _, _ = get_vit(num_classes=len(CLASS_NAMES), device=DEVICE)
    else:
        raise ValueError("Invalid model type. Choose: baseline | resnet50 | vit")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model

def predict_image(image_path, model):
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    return CLASS_NAMES[pred_idx], confidence
