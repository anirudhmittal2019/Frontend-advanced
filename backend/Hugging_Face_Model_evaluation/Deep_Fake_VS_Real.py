import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image

# Global model instances
MODEL = None
FEATURE_EXTRACTOR = None

def _load_model():
    """Load model into global variables if not already loaded"""
    global MODEL, FEATURE_EXTRACTOR
    try:
        if MODEL is None or FEATURE_EXTRACTOR is None:
            print("Loading VS Real model...")
            model_name = "dima806/deepfake_vs_real_image_detection"
            MODEL = AutoModelForImageClassification.from_pretrained(model_name)
            FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(model_name)
            print("VS Real model loaded successfully")
    except Exception as e:
        print(f"Error loading VS Real model: {str(e)}")
        raise

def _get_analysis(result, confidence):
    """Internal function to get analysis context"""
    if result == "FAKE":
        if confidence > 90:
            return "Strong indicators of image manipulation detected"
        elif confidence > 70:
            return "Moderate indicators of image manipulation detected"
        return "Weak indicators of image manipulation detected"
    else:
        if confidence > 90:
            return "Strong indicators of authentic image"
        elif confidence > 70:
            return "Moderate indicators of authentic image"
        return "Analysis inconclusive - low confidence score"

def predict_image(image_path):
    """
    Single entry point for image prediction using cached model
    Args:
        image_path (str): Path to the image file
    Returns:
        dict: {
            'prediction': 0 or 1 (0=real, 1=fake),
            'confidence': float (0-100),
            'label': str ('REAL' or 'FAKE'),
            'analysis': str (detailed analysis)
        }
    """
    try:
        # Ensure model is loaded
        _load_model()
        
        # Process image
        image = Image.open(image_path)
        inputs = FEATURE_EXTRACTOR(image, return_tensors="pt")

        # Get prediction
        with torch.no_grad():
            outputs = MODEL(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = logits.argmax(-1).item()
            confidence = probabilities[0][predicted_class].item() * 100

        # Format results
        label = "REAL" if predicted_class == 1 else "FAKE"
        binary_result = 0 if label == "REAL" else 1
        analysis = _get_analysis(label, confidence)

        return {
            'prediction': binary_result,
            'confidence': confidence,
            'label': label,
            'analysis': analysis
        }

    except Exception as e:
        print(f"Error in VS Real prediction: {str(e)}")
        return {
            'prediction': None,
            'confidence': 0,
            'label': 'ERROR',
            'analysis': f'Error occurred: {str(e)}'
        }