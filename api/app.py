# api/app.py
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import base64
import json
from groq import Groq
from .predict_service import load_model, predict_image
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chest X-Ray Pneumonia Detection")

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Templates
templates = Jinja2Templates(directory="api/templates")

# Load environment variables
load_dotenv()

# Initialize model (will download from S3 automatically)
MODEL_TYPE = "resnet50"
try:
    logger.info("Loading model from S3...")
    model = load_model(model_type=MODEL_TYPE)  # No checkpoint_path needed - will auto-download
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

UPLOAD_FOLDER = "api/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found. Chat functionality will be disabled.")
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)

# Store current session data (in production, use proper session management)
current_session = {
    "prediction": None,
    "confidence": None,
    "image_base64": None
}

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint for deployment monitoring"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    }

# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Please upload a valid image file"
            })
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Processing image: {file.filename}")
        
        # Convert image to base64 for LLM (if chat is enabled)
        img_base64 = None
        if client:  # Only if Groq is available
            with open(file_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Make prediction
        pred_class, confidence = predict_image(file_path, model)
        
        # Store session data for chat
        current_session.update({
            "prediction": pred_class,
            "confidence": confidence,
            "image_base64": img_base64
        })
        
        # Remove uploaded file
        os.remove(file_path)
        
        logger.info(f"Prediction completed: {pred_class} ({confidence:.4f})")
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "chat_enabled": client is not None
        })
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Prediction failed: {str(e)}"
        })

# Chat endpoint
@app.post("/chat")
async def chat_with_llm(request: Request):
    if not client:
        return JSONResponse({"error": "Chat service not available - missing GROQ_API_KEY"})
    
    data = await request.json()
    user_message = data.get("message", "")
    
    if not current_session["prediction"]:
        return JSONResponse({"error": "No prediction available for chat"})
    
    try:
        # Create system prompt with medical context
        system_prompt = f"""You are a medical AI assistant analyzing chest X-ray results. 

Current Analysis:
- Diagnosis: {current_session['prediction']}
- Confidence Score: {current_session['confidence']}

Instructions:
- Act as a knowledgeable medical assistant
- Explain the diagnosis in simple terms
- Answer questions about pneumonia, X-rays, and the results
- Always remind users to consult healthcare professionals for medical decisions
- Be empathetic and supportive
- If asked about the confidence score, explain what it means
- Keep responses concise but informative

Remember: This is AI-assisted diagnosis tool, not a replacement for professional medical advice."""

        # Prepare messages for vision model
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add image context if available
        if current_session["image_base64"]:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Here's the chest X-ray image. My AI model diagnosed it as: {current_session['prediction']} with confidence: {current_session['confidence']}. Now the user asks: {user_message}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{current_session['image_base64']}"
                        }
                    }
                ]
            })
        else:
            messages.append({
                "role": "user", 
                "content": user_message
            })
        
        # Call Groq Vision API
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",  # Vision model
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        llm_response = completion.choices[0].message.content
        
        return JSONResponse({
            "response": llm_response,
            "diagnosis": current_session["prediction"],
            "confidence": current_session["confidence"]
        })
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        # Fallback to text-only model if vision fails
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            llm_response = completion.choices[0].message.content
            return JSONResponse({
                "response": llm_response,
                "diagnosis": current_session["prediction"], 
                "confidence": current_session["confidence"]
            })
        except Exception as fallback_error:
            logger.error(f"Chat fallback failed: {str(fallback_error)}")
            return JSONResponse({
                "error": f"Chat service unavailable: {str(fallback_error)}"
            })

# Clear session endpoint
@app.post("/clear-session")
async def clear_session():
    current_session.update({
        "prediction": None,
        "confidence": None, 
        "image_base64": None
    })
    return JSONResponse({"status": "cleared"})

# Model info endpoint (useful for monitoring)
@app.get("/model-info")
def model_info():
    """Get information about the loaded model"""
    return {
        "model_type": MODEL_TYPE,
        "classes": ["NORMAL", "PNEUMONIA"],
        "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    }