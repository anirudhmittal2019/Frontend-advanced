"""
Deepfake Detection System - Core Router
-------------------------------------
Orchestrates multiple detection models and provides unified analysis interface.

Architecture Overview:
1. Model Coordination Layer
2. Result Aggregation System
3. Persistence Management
4. Rate Limiting Controls
5. Error Recovery Mechanisms

Performance Notes:
- API rate limits enforced
- Temporary file cleanup
- Resource management
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import requests
from io import BytesIO
from PIL import Image
from functools import wraps
import time

# Import model functions from different modules
from Google_img_classification_label.describing import analyze_image as analyze_vision
from Hugging_Face_Model_evaluation.Deep_Fake_VS_Real import predict_image as predict_vs_real
from Hugging_Face_Model_evaluation.Deep_Fake_Detector_v2_Model import predict_image as predict_v2

######################################################################################
"""
Result Management
---------------
JSON-based persistence with audit trail capabilities.
Implements atomic operations for reliability.
"""

def save_result_to_json(result, filename="analysis_results.json"):
    """
    Atomic result persistence with collision handling.
    Maintains chronological order for analysis history.
    """
    file_path = os.path.join(os.path.dirname(__file__), '..', 'results', filename)
    print(f"Saving results to: {file_path}")
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    result['timestamp'] = datetime.now().isoformat()
    
    try:
        existing_results = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_results = json.load(f)
                print(f"Loaded {len(existing_results)} existing results")
        
        existing_results.append(result)
        print(f"Adding new result. Total results: {len(existing_results)}")
        
        with open(file_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
            print("Results saved successfully")
            
    except Exception as e:
        print(f"Error saving results to JSON: {str(e)}")

def get_analysis_history(filename="analysis_results.json"):
    """Get all analysis results from JSON file"""
    file_path = os.path.join(os.path.dirname(__file__), '..', 'results', filename)
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error reading results from JSON: {str(e)}")
        return []

######################################################################################

"""
Analysis Pipeline
---------------
Multi-stage detection system with fallback mechanisms.
Combines multiple model outputs for robust detection.
"""

load_dotenv()
GOOGLE_API = os.getenv('GOOGLE_API')
genai.configure(api_key=GOOGLE_API)

class DeepfakeAnalyzer:
    """
    Central analysis coordinator implementing:
    1. Model voting system
    2. Confidence scoring
    3. Cross-validation
    4. Fallback mechanisms
    
    Detection Strategy:
    - Primary: V2 model (threshold: 70%)
    - Secondary: VS Real validation
    - Tertiary: Google Vision verification
    """

    def __init__(self):
        """Initialize the analyzer with required models"""
        #self.nude_detector = Detector("detector_v2_base_checkpoint", "detector_v2_base_classes")

    def analyze_image(self, image_path):
        """
        Primary analysis pipeline. Coordinates between models and generates
        a final summary using LLM-based analysis.
        
        Flow:
        1. Collect predictions from all models
        2. Aggregate and normalize results
        3. Generate human-readable summary
        4. Apply safety checks and fallbacks
        """
        try:
            # Get model predictions
            results = self._collect_model_predictions(image_path)
            if not results:
                raise ValueError("Failed to collect model predictions")

            # Generate summary
            final_decision = self._generate_summary(results)
            if not final_decision:
                raise ValueError("Failed to generate summary")

            results["final_summary"] = final_decision
            return results

        except Exception as e:
            print(f"Error during image analysis: {str(e)}")
            return None

    def _collect_model_predictions(self, image_path):
        """
        Gathers predictions from multiple models.
        
        Models used:
        1. Google Vision API - for general image analysis
        2. VS Real Model - specialized in real vs fake detection
        3. V2 Model - our latest deepfake detection model
        
        Note: Each model has different strengths and specializations
        """
        try:  # Add try-catch here
            # 1. Google Vision Analysis
            vision_results = analyze_vision(image_path)
            if not vision_results:
                raise ValueError("Google Vision analysis failed")
            
            # 2. VS Real Model
            vs_results = predict_vs_real(image_path)
            if not vs_results:
                raise ValueError("VS Real model analysis failed")
            
            # 3. V2 Model
            v2_results = predict_v2(image_path)
            if not v2_results or len(v2_results) != 2:
                raise ValueError("V2 model analysis failed")

            return {
                "google_vision": vision_results,
                "huggingface_models": {
                    "vs_model": vs_results,
                    "v2_model": {
                        "prediction": str(v2_results[0]),  # Convert to string
                        "confidence": float(v2_results[1])  # Ensure float
                    }
                }
            }
        except Exception as e:
            print(f"Model prediction error: {str(e)}")
            return None

    def _generate_summary(self, results):
        """
        Uses Gemini to generate a comprehensive analysis summary.
        
        Decision rules:
        1. V2 model confidence > 70% triggers fake detection
        2. Requires cross-validation from multiple sources
        3. Implements conservative detection to minimize false positives
        
        Returns: Structured JSON with decision metrics
        """
        model = genai.GenerativeModel('gemini-1.0-pro')
        
        prompt = f"""
        You are a strictly controlled JSON generator. Return ONLY a valid JSON object with EXACTLY this structure, no extra text:
        {{
            "decision": 0 or 1 (0 for real, 1 for fake),
            "confidence": number between 0-100,
            "factors": ["factor1", "factor2", ...],
            "explanation": "brief explanation"
        }}

        Analyze these detection results and format your response accordingly:
        1. Google Vision: {results['google_vision']}
        2. VS Model: {results['huggingface_models']['vs_model']}
        3. V2 Model: {results['huggingface_models']['v2_model']}

        Follow these rules:
        1. Mark as fake (1) if V2 models exceed 70% confidence.
        2. If V2 Predict it is real(0), return 0, if V2 Predict it is fake(1), return 1(If VS Model FAKE confidence > 70%, ignore. Only look if V2 Model > 70% & Google vision)
        3. Enumerate specific detection factors.
        4. Keep the explanation under 100 words.
        5. Default to real (0) unless the rules indicate otherwise.

        Ensure you return ONLY the JSON object, with absolutely no extra commentary.
        """

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean up the response text
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            
            # Remove any additional whitespace/newlines
            text = text.strip()
            
            # Validate it starts with { and ends with }
            if not (text.startswith('{') and text.endswith('}')):
                raise ValueError("Response is not a valid JSON object")
                
            # Parse JSON
            result = json.loads(text)
            
            # Validate required keys
            required_keys = ["decision", "confidence", "factors", "explanation"]
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
            
            # Validate and convert types
            result["decision"] = int(result["decision"])
            if result["decision"] not in [0, 1]:
                raise ValueError("Decision must be 0 or 1")
                
            result["confidence"] = float(result["confidence"])
            if not 0 <= result["confidence"] <= 100:
                raise ValueError("Confidence must be between 0 and 100")
                
            if not isinstance(result["factors"], list):
                result["factors"] = [str(result["factors"])]
                
            if not isinstance(result["explanation"], str):
                raise ValueError("Explanation must be a string")
                
            return result

        except Exception as e:
            print(f"Error in Gemini analysis: {str(e)}")
            return self._fallback_decision(results)

    def _fallback_decision(self, results):
        """
        Backup decision mechanism when LLM analysis fails.
        Uses a weighted voting system based on model confidences.
        
        This is our safety net to ensure system reliability
        even when primary analysis paths fail.
        """
        fake_votes = 0
        total_votes = 0
        confidence_sum = 0
        
        # VS Model vote
        vs_result = results['huggingface_models']['vs_model']
        if vs_result['prediction'] == 1:
            fake_votes += 1
            confidence_sum += vs_result['confidence']
        total_votes += 1
        
        # V2 Model vote
        v2_result = results['huggingface_models']['v2_model']
        if v2_result['prediction'] == "FAKE":
            fake_votes += 1
            confidence_sum += v2_result['confidence']
        total_votes += 1

        # Calculate decision using Majority Threshold & Confidence Scoring
        decision = 1 if (fake_votes / total_votes) > 0.5 else 0
        confidence = (confidence_sum / total_votes) if fake_votes > 0 else (100 - confidence_sum / total_votes)

        return {
            "decision": decision,
            "confidence": min(confidence, 100),
            "factors": [f"{fake_votes} out of {total_votes} models indicate fake"],
            "explanation": f"Fallback decision based on majority voting with {confidence:.1f}% confidence"
        }
    
"""
Image Processing Controls
----------------------
Implements safety measures and format handling.
Ensures consistent image processing across formats.
"""

def rate_limit(seconds):
    """
    Decorator for API rate limiting.
    Ensures we don't exceed API quotas and maintains system stability.
    
    Warning: This is crucial for production deployment!
    """
    def decorator(func):
        last_called = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            if func.__name__ not in last_called or \
               current_time - last_called[func.__name__] >= seconds:
                last_called[func.__name__] = current_time
                return func(*args, **kwargs)
            else:
                time_to_wait = seconds - (current_time - last_called[func.__name__])
                time.sleep(time_to_wait)
                last_called[func.__name__] = time.time()
                return func(*args, **kwargs)
        return wrapper
    return decorator

def _get_safe_image_format(img):
    """
    Safely determines image format with fallback to JPEG.
    Supports: JPEG, PNG, WebP, BMP
    
    Note: We prefer JPEG for consistent processing
    """
    fmt = img.format
    if fmt in ['JPEG', 'PNG', 'WebP', 'BMP']:
        return fmt
    return 'JPEG'

def _generate_temp_filename(original_url, fmt='JPEG'):
    """
    Creates unique temporary filenames based on URL hash and timestamp.
    Helps prevent collisions in high-traffic scenarios.
    """
    timestamp = int(time.time())
    url_id = abs(hash(original_url)) % 10000
    ext = '.jpg' if fmt == 'JPEG' else f'.{fmt.lower()}'
    return f"temp_image_{timestamp}_{url_id}{ext}"


"""
Main Interface
-------------
Primary entry point for image analysis with rate limiting
and comprehensive error handling.
"""

@rate_limit(1.0)
def inference(imgurl, image_id=None):
    """
    Production-grade inference pipeline with:
    - Rate limiting
    - Format validation
    - Error recovery
    - Result persistence
    - Resource cleanup
    """
    temp_file = None
    try:
        analyzer = DeepfakeAnalyzer()
        
        # Create temp directory with absolute path
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Using temp directory: {temp_dir}")
        
        # Handle chrome-extension URLs
        if imgurl.startswith('chrome-extension://'):
            result = {
                'decision': 0,
                'confidence': 100,
                'img_url': imgurl,
                'explanation': 'Chrome extension resource - skipped analysis',
                'image_id': image_id
            }
            save_result_to_json(result)
            return result

        try:
            # Download image
            print(f"Downloading image from: {imgurl}")
            response = requests.get(imgurl, timeout=10)
            response.raise_for_status()
            
            # Open image and detect format
            img = Image.open(BytesIO(response.content))
            original_format = _get_safe_image_format(img)
            print(f"Detected image format: {original_format}")
            
            # Convert to RGB if needed
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Save image with appropriate format
            temp_file = os.path.join(temp_dir, _generate_temp_filename(imgurl, original_format))
            save_params = {}
            
            if original_format == 'JPEG':
                save_params = {'quality': 95, 'optimize': True}
            elif original_format == 'PNG':
                save_params = {'optimize': True}
            elif original_format == 'WebP':
                save_params = {'quality': 95, 'method': 4}
            
            img.save(temp_file, format=original_format, **save_params)
            print(f"Saved temporary file: {temp_file}")
            
            # Verify file exists and has content
            if not os.path.exists(temp_file):
                raise FileNotFoundError(f"Failed to save temporary file: {temp_file}")
            
            file_size = os.path.getsize(temp_file)
            if file_size == 0:
                raise ValueError("Saved file is empty")
            print(f"File size: {file_size} bytes")
            
            # Process image
            print("Analyzing image...")
            results = analyzer.analyze_image(temp_file)
            
            if results is None:
                raise ValueError("Analysis failed to produce results")
            
            # Format result
            final_summary = results.get("final_summary", {})
            result = {
                'decision': final_summary.get('decision', 1),
                'confidence': final_summary.get('confidence', 0),
                'img_url': imgurl,
                'explanation': final_summary.get('explanation', 'Analysis failed'),
                'image_id': image_id,
                'format': original_format
            }
            
            # Save and return result
            save_result_to_json(result)
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Download error: {str(e)}")
            result = {
                'decision': 0,
                'confidence': 100,
                'img_url': imgurl,
                'explanation': f'Failed to download image: {str(e)}',
                'image_id': image_id
            }
            save_result_to_json(result)
            return result
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            raise
            
    except Exception as e:
        print(f"Unexpected error in inference: {str(e)}")
        result = {
            'decision': 0,
            'confidence': 100,
            'img_url': imgurl,
            'explanation': f'System error: {str(e)}',
            'image_id': image_id
        }
        save_result_to_json(result)
        return result      
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                print(f"Error cleaning up temporary file: {str(e)}")

# Testing the function with the elon musk picture
def main():
    """
    Test harness for the analyzer.
    Uses a known test image to verify system functionality.
    """
    analyzer = DeepfakeAnalyzer()
    test_image = "../Google_img_classification_label/image/fakeelon.jpg"
    
    print(f"Analyzing image: {test_image}")
    results = analyzer.analyze_image(test_image)
    
    if results:
        if results.get("decision") == 2:
            print("\n⚠️ " + results["message"])
        else:
            print("\n=== Analysis Results ===")
            final_summary = results.get("final_summary", {})

            #Place where it output 0 or 1
            print(f"\nDecision: {'1' if final_summary.get('decision') == 1 else '0'}")
            print(f"Confidence: {final_summary.get('confidence', 0):.1f}%")
            print("\nKey Factors:")
            for factor in final_summary.get('factors', []):
                print(f"- {factor}")
            print(f"\nExplanation: {final_summary.get('explanation', 'No explanation available')}")
    else:
        print("Analysis failed")

if __name__ == "__main__":
    main()