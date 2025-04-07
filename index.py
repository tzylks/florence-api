import os
import logging
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import jwt
import datetime
import traceback

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])
logging.basicConfig(level=logging.DEBUG)  # Log to Railway console
logger = logging.getLogger(__name__)

# Load secret key from environment variable
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "your-secret-key-for-dev-only")
PORT = int(os.getenv("PORT", 8080))  # Railway assigns PORT dynamically

# Florence-2 setup (lazy-loaded)
device = "cpu"  # Railway free tier has no GPU
torch_dtype = torch.float16  # Reduce memory usage
model_name = "microsoft/Florence-2-base"
model = None
processor = None

def load_model():
    global model, processor
    if model is None or processor is None:
        try:
            logger.info(f"Loading {model_name} model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch_dtype, 
                trust_remote_code=True
            ).to(device)
            processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            model.eval()  # Optimize for inference
            logger.info("Model and processor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}\n{traceback.format_exc()}")
            raise

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if email == "test@example.com" and password == "password123":
        token = jwt.encode({
            'email': email,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        }, SECRET_KEY, algorithm="HS256")
        response = make_response(jsonify({'message': 'Login successful'}), 200)
        is_secure = os.getenv('FLASK_ENV') != 'development'
        response.set_cookie(
            'token',
            token,
            httponly=True,
            samesite='None' if is_secure else 'Lax',
            secure=is_secure,
            max_age=3600,
            path='/'
        )
        logger.info("User logged in successfully")
        return response
    logger.warning("Invalid login attempt")
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout', methods=['POST'])
def logout():
    response = make_response(jsonify({'message': 'Logged out'}), 200)
    is_secure = os.getenv('FLASK_ENV') != 'development'
    response.set_cookie(
        'token',
        '',
        httponly=True,
        samesite='None' if is_secure else 'Lax',
        secure=is_secure,
        max_age=0,
        path='/'
    )
    logger.info("User logged out")
    return response

@app.route('/predict', methods=['POST'])
def predict():
    token = request.cookies.get('token')
    if not token:
        logger.warning("No token provided")
        return jsonify({'error': 'Authentication required'}), 401
    try:
        jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        logger.warning("Invalid or expired token")
        return jsonify({'error': 'Invalid or expired token'}), 401

    if 'image' not in request.files:
        logger.warning("No image in request")
        return jsonify({'error': 'No image provided'}), 400

    load_model()  # Lazy load on first request

    try:
        image_file = request.files['image']
        logger.info(f"Received image: {image_file.filename}")
        image = Image.open(image_file).convert("RGB")
        
        task_prompt = "<CAPTION>"
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]  # Already in float16
        
        logger.info("Image processed, running model inference...")
        with torch.no_grad():  # Reduce memory usage
            generated_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1024,
                num_beams=3
