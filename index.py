import os
import logging
import transformers  # Add this line
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import traceback

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info(f"Using transformers version: {transformers.__version__}")
logger.info(f"Using torch version: {torch.__version__}")

PORT = int(os.getenv("PORT", 8080))

device = "cpu"
model_name = "openai/clip-vit-base-patch32"
model = None
processor = None

def load_model():
    global model, processor
    if model is None or processor is None:
        try:
            logger.info(f"Loading {model_name} model...")
            model = CLIPModel.from_pretrained(model_name).to(device)
            processor = CLIPProcessor.from_pretrained(model_name)
            model.eval()
            logger.info("Model and processor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}\n{traceback.format_exc()}")
            raise

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        logger.warning("No image in request")
        return jsonify({'error': 'No image provided'}), 400

    logger.info("Loading model for prediction...")
    load_model()

    try:
        image_file = request.files['image']
        logger.info(f"Received image: {image_file.filename}")
        image = Image.open(image_file).convert("RGB")

        # Predefined caption candidates
        candidate_captions = [
            "A dog in a park",
            "A cat on a couch",
            "A person walking",
            "A car on the road",
            "A sunny beach",
        ]

        # Process image and text with CLIP
        inputs = processor(
            text=candidate_captions,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        logger.info("Image and text processed, running model inference...")
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # Select the best caption
        best_caption_idx = probs.argmax().item()
        result = candidate_captions[best_caption_idx]
        logger.info(f"Caption selected: {result}")
        return jsonify({'result': result})
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Server error during prediction'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT)
