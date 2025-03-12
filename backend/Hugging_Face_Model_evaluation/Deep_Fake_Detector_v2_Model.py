import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image

# Global model instances
MODEL_V2 = None
FEATURE_EXTRACTOR_V2 = None

def _load_model():
    """Load model into global variables if not already loaded"""
    global MODEL_V2, FEATURE_EXTRACTOR_V2
    try:
        if MODEL_V2 is None or FEATURE_EXTRACTOR_V2 is None:
            print("Loading V2 model...")
            model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
            MODEL_V2 = AutoModelForImageClassification.from_pretrained(model_name)
            FEATURE_EXTRACTOR_V2 = AutoFeatureExtractor.from_pretrained(model_name)
            print("V2 model loaded successfully")
    except Exception as e:
        print(f"Error loading V2 model: {str(e)}")
        raise

def predict_image(image_path):
    """
    Predict using cached model
    Args:
        image_path (str): Path to the image file
    Returns:
        tuple: (str, float) - (prediction ['REAL' or 'FAKE'], confidence percentage)
    """
    try:
        # Ensure model is loaded
        _load_model()

        # Process image
        image = Image.open(image_path)
        inputs = FEATURE_EXTRACTOR_V2(image, return_tensors="pt")

        # Get prediction
        with torch.no_grad():
            outputs = MODEL_V2(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
            confidence = torch.nn.functional.softmax(logits, dim=-1).max().item()
        
        # Return result
        result = "REAL" if predicted_class == 0 else "FAKE"
        return result, confidence * 100

    except Exception as e:
        print(f"Error in V2 prediction: {str(e)}")
        return "ERROR", 0