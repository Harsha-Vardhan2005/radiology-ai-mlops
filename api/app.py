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

app = FastAPI(title="Chest X-Ray Pneumonia Detection")

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Templates
templates = Jinja2Templates(directory="api/templates")

# Load model once
MODEL_TYPE = "resnet50"
CHECKPOINT_PATH = "models/checkpoints/ResNet50_best.pth"
model = load_model(model_type=MODEL_TYPE, checkpoint_path=CHECKPOINT_PATH)

UPLOAD_FOLDER = "api/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Groq client - Replace with your actual API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY. Please set it in your .env file.")

client = Groq(api_key=GROQ_API_KEY)

# Store current session data (in production, use proper session management)
current_session = {
    "prediction": None,
    "confidence": None,
    "image_base64": None
}

# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Convert image to base64 for LLM
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
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": pred_class,
        "confidence": round(confidence, 4)
    })

# Chat endpoint
@app.post("/chat")
async def chat_with_llm(request: Request):
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
        print(f"Error in chat: {str(e)}")
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