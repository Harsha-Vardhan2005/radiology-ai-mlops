import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from models.baseline_cnn import BaselineCNN
from models.transfer_learning import get_resnet50, get_vit

# -----------------------------
# Configuration
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # adjust if needed


# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_image(image_path, img_size=IMG_SIZE):
    """
    Load and preprocess a single image for model inference.
    """
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0)  # add batch dimension
    return img_tensor.to(DEVICE)


# -----------------------------
# Load Model function
# -----------------------------
def load_model(model_type="baseline", checkpoint_path=None):
    """
    Load a trained model for inference.
    model_type: baseline | resnet50 | vit
    checkpoint_path: path to saved .pth file
    """
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


# -----------------------------
# Prediction function
# -----------------------------
def predict(image_path, model, class_names=CLASS_NAMES):
    """
    Predict the class of a single image using the given model.
    """
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    return class_names[pred_idx], confidence


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example: Predict using ResNet50
    model_path = "models/checkpoints/baseline_best.pth"
    model = load_model(model_type="resnet50", checkpoint_path=model_path)

    test_image = "data/chest_xray/test/PNEUMONIA/person1_bacteria_1.jpeg"
    pred_class, conf = predict(test_image, model)
    print(f"Prediction: {pred_class}, Confidence: {conf:.4f}")
