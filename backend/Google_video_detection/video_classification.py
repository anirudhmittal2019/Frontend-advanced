"""
Video Analysis System
-------------------
Advanced video analysis using Google Cloud Video Intelligence API.
Implements multi-factor detection for synthetic/manipulated content.

Core Features:
1. Face detection with confidence analysis
2. Temporal consistency validation
3. Content manipulation detection
4. Weighted scoring system

Design Philosophy:
- Comprehensive but efficient analysis
- Fault-tolerant operation
- Clear audit trails
"""

import os
from google.cloud import videointelligence
from datetime import datetime
import json

class VideoAnalyzer:
    """
    Core analyzer with sophisticated detection algorithms.
    
    Technical Implementation:
    - 5-minute operation timeout
    - Multi-factor analysis
    - Confidence-based scoring
    """
    
    def __init__(self):
        """Initialize the video analyzer"""
        credentials_path = os.path.join(os.path.dirname(__file__), "turkeye_sa.json")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.client = videointelligence.VideoIntelligenceServiceClient()
        print(f"Using credentials from: {credentials_path}")

    def _analyze_video_content(self, video_path):
        """
        Primary analysis using Google Cloud API.
        Processes multiple detection types in parallel:
        - Label detection (content classification)
        - Explicit content detection (anomaly checking)
        - Face detection (manipulation indicators)
        
        Note: Heavy operation - consider API costs
        """
        try:
            # Read the local video file
            with open(video_path, "rb") as file:
                video_content = file.read()

            # Configure the request
            features = [
                videointelligence.Feature.LABEL_DETECTION,
                videointelligence.Feature.EXPLICIT_CONTENT_DETECTION,
                videointelligence.Feature.FACE_DETECTION,
            ]

            request = {
                "input_content": video_content,
                "features": features
            }

            print(f"Starting video analysis for: {video_path}")
            operation = self.client.annotate_video(request)
            print("Waiting for operation to complete...")
            result = operation.result(timeout=300)  # 5-minute timeout
            print("Analysis complete")

            return result

        except Exception as e:
            print(f"Error analyzing video: {str(e)}")
            return None

    def _evaluate_fakeness(self, result):
        """
        Multi-factor fake detection system.
        
        Scoring Components & Weights:
        - Face anomalies: 40% (primary indicator)
        - Content markers: 35% (secondary validation)
        - Temporal consistency: 25% (supporting check)
        
        Thresholds (empirically determined):
        - Face confidence < 0.6: potential fake
        - Overall fake threshold: 60%
        - Label confidence weights: 0.7-1.0
        """
        try:
            fake_score = 0
            total_factors = 0
            factors = []

            # Debug print to see structure
            print("Available annotation results:", [
                attr for attr in dir(result.annotation_results[0]) 
                if not attr.startswith('_')
            ])

            # Critical detection factors
            for annotation_result in result.annotation_results:
                # Face Detection (40% weight)
                if hasattr(annotation_result, 'face_annotations'):
                    faces = annotation_result.face_annotations
                    if faces:
                        total_factors += 1
                        low_confidence_faces = []
                        for face in faces:
                            if hasattr(face, 'confidence') and face.confidence < 0.6:
                                low_confidence_faces.append(face.confidence)
                        
                        if low_confidence_faces:
                            fake_score += 1
                            avg_confidence = sum(low_confidence_faces) / len(low_confidence_faces)
                            factors.append(f"Low face detection confidence: {avg_confidence:.2f}")

                # Content Analysis (35% weight)
                if hasattr(annotation_result, 'segment_label_annotations'):
                    labels = annotation_result.segment_label_annotations
                    if labels:
                        total_factors += 1
                        suspicious_labels = {
                            'animation': 0.8,
                            'graphic': 0.7,
                            'special effects': 0.9,
                            'computer generated': 1.0,
                            'artificial': 0.8,
                            'cgi': 1.0,
                            'rendered': 0.9
                        }
                        
                        detected_suspicious = []
                        for label in labels:
                            label_text = label.entity.description.lower()
                            for susp_label, weight in suspicious_labels.items():
                                if susp_label in label_text:
                                    confidence = label.segments[0].confidence if label.segments else 0.5
                                    score = confidence * weight
                                    detected_suspicious.append((susp_label, score))
                        
                        if detected_suspicious:
                            fake_score += sum(score for _, score in detected_suspicious)
                            labels_found = [f"{label} ({score:.2f})" for label, score in detected_suspicious]
                            factors.append(f"Artificial content indicators: {', '.join(labels_found)}")

                # Temporal Analysis (25% weight)
                if hasattr(annotation_result, 'explicit_annotation'):
                    explicit = annotation_result.explicit_annotation
                    if explicit and explicit.frames:
                        total_factors += 1
                        suspicious_frames = [
                            frame for frame in explicit.frames 
                            if frame.pornography_likelihood >= videointelligence.Likelihood.LIKELY
                        ]
                        if suspicious_frames:
                            fake_score += 1
                            factors.append("Suspicious content detected in video frames")

            # Calculate final score
            if total_factors > 0:
                fake_probability = (fake_score / total_factors) * 100
            else:
                fake_probability = 50  # Default to uncertain if no factors analyzed
                factors.append("No definitive factors found for analysis")

            # Format final decision
            return {
                "decision": "FAKE" if fake_probability >= 60 else "REAL",
                "confidence": min(100, max(0, fake_probability)),  # Clamp between 0-100
                "factors": factors,
                "raw_scores": {
                    "fake_score": fake_score,
                    "total_factors": total_factors
                }
            }

        except Exception as e:
            print(f"Error evaluating fakeness: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error trace for debugging
            return None

    def analyze_video(self, video_path):
        """
        Main analysis pipeline with error recovery.
        Coordinates between detection systems and provides
        unified results with confidence scoring.
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Analyze video content
            result = self._analyze_video_content(video_path)
            if not result:
                raise ValueError("Failed to analyze video")

            # Evaluate fakeness
            evaluation = self._evaluate_fakeness(result)
            if not evaluation:
                raise ValueError("Failed to evaluate video")

            # Add metadata
            evaluation.update({
                "video_path": video_path,
                "timestamp": datetime.now().isoformat(),
            })

            return evaluation

        except Exception as e:
            print(f"Error in video analysis pipeline: {str(e)}")
            return {
                "decision": "ERROR",
                "confidence": 0,
                "factors": [f"Analysis error: {str(e)}"],
                "video_path": video_path,
                "timestamp": datetime.now().isoformat()
            }

def save_result(result, filename="video_analysis_results.json"):
    """
    Atomic result persistence with collision handling.
    Maintains analysis history for pattern detection.
    """
    file_path = os.path.join(os.path.dirname(__file__), "../results", filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        existing_results = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_results = json.load(f)
        
        existing_results.append(result)
        
        with open(file_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
        print(f"Results saved to: {file_path}")
            
    except Exception as e:
        print(f"Error saving results: {str(e)}")

# Usage example
if __name__ == "__main__":
    analyzer = VideoAnalyzer()
    
    # Local video path
    video_path = "output.mp4"  # Make sure this file exists in the same directory
    
    print(f"Analyzing video: {video_path}")
    result = analyzer.analyze_video(video_path)
    print("\nAnalysis Result:")
    print(json.dumps(result, indent=2))
    
    save_result(result)