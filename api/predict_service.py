# api/predict_service.py
import torch
import boto3
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_learning import get_resnet50, get_vit
from pathlib import Path
import logging

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# AWS Configuration
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')  
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET = os.getenv('S3_BUCKET_NAME')

def download_model_from_s3():
    """Download model from S3 if not exists locally"""
    model_filename = "ResNet50_best.pth"
    local_model_path = f"models/{model_filename}"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Check if model already exists locally
    if os.path.exists(local_model_path):
        logger.info(f"Model {model_filename} already exists locally")
        return local_model_path
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        
        logger.info(f"Downloading {model_filename} from S3 bucket: {S3_BUCKET}")
        
        # Download model from S3
        s3_client.download_file(S3_BUCKET, model_filename, local_model_path)
        
        logger.info(f"Successfully downloaded {model_filename}")
        return local_model_path
        
    except Exception as e:
        logger.error(f"Error downloading model from S3: {str(e)}")
        raise Exception(f"Failed to download model: {str(e)}")

def preprocess_image(image_path):
    """Preprocess image for model input"""
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

def load_model(model_type="resnet50", checkpoint_path=None):
    """Load model with automatic S3 download"""
    
    # If no checkpoint path provided, download from S3
    if checkpoint_path is None:
        checkpoint_path = download_model_from_s3()
    
    if model_type == "baseline":
        model = BaselineCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    elif model_type == "resnet50":
        model, _, _ = get_resnet50(num_classes=len(CLASS_NAMES), device=DEVICE)
    elif model_type == "vit":
        model, _, _ = get_vit(num_classes=len(CLASS_NAMES), device=DEVICE)
    else:
        raise ValueError("Invalid model type. Choose: baseline | resnet50 | vit")

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()
        logger.info(f"Successfully loaded {model_type} model")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise Exception(f"Failed to load model: {str(e)}")

def predict_image(image_path, model):
    """Make prediction on image"""
    try:
        img_tensor = preprocess_image(image_path)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        logger.info(f"Prediction: {CLASS_NAMES[pred_idx]}, Confidence: {confidence:.4f}")
        return CLASS_NAMES[pred_idx], confidence
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise Exception(f"Prediction failed: {str(e)}")