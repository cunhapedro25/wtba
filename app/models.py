"""
Model management and YOLO integration
"""


import requests
import base64
from pathlib import Path
from ultralytics import YOLO
import io

class DetectionModel:
    def __init__(self):
        self.model = None
        self.class_names = ['hog', 'rabbit', 'pigeon', 'deer']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0)]
        self.model_path = None
        self._setup_model_path()

    def _setup_model_path(self):
        """Setup path to the YOLO model"""
        current_dir = Path.cwd()
        project_dir = current_dir.parent if current_dir.name == 'src' else current_dir
        self.model_path = project_dir / "models" / "best.pt"

    def load_model(self):
        """Load the YOLO model"""
        try:
            if not self.model_path.exists():
                print(f"Model not found: {self.model_path}")
                return False
            self.model = YOLO(str(self.model_path))
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None

    def predict(self, *args, **kwargs):
        """Run prediction on image"""
        if not self.is_loaded():
            raise Exception("Model not loaded")
        return self.model(*args, **kwargs)

    def predict_silent(self, image, conf_threshold=0.5):
        """Run prediction without verbose output"""
        if not self.is_loaded():
            raise Exception("Model not loaded")
        return self.model(image, conf=conf_threshold, verbose=False)


class LLMAnalyzer:
    def __init__(self, ollama_url="http://localhost:11434", model_name="llava:7b"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self._test_connection()

    def _test_connection(self):
        """Test connection to Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                print(f"Connected to Ollama. Available models: {available_models}")

                if not any(self.model_name in model for model in available_models):
                    print(f"Warning: {self.model_name} not found. Using first available model.")
                    if available_models:
                        self.model_name = available_models[0]
            else:
                print(f"Ollama connection failed: {response.status_code}")
        except Exception as e:
            print(f"Ollama connection error: {e}")

    def _image_to_base64(self, image_path):
        """Convert image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None

    def analyze_image(self, image_path, detections=None):
        """Analyze image with LLM"""
        try:
            image_b64 = self._image_to_base64(image_path)
            if not image_b64:
                return "Unable to process image for LLM analysis."

            detection_context = ""
            if detections:
                animals_found = [d['animal'] for d in detections]
                confidence_scores = [f"{d['animal']}: {d['confidence']:.2f}" for d in detections]
                detection_context = f"\n\nDetected wildlife: {', '.join(set(animals_found))} with confidence scores: {', '.join(confidence_scores)}"

            prompt = f"""Analyze this wildlife image and provide a detailed description of what you see. Focus on:
1. The environment and habitat
2. Animal behavior and positioning
3. Overall scene composition
4. Any interesting wildlife interactions

{detection_context}

Provide a natural, engaging description as if you're a wildlife expert explaining the scene."""

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No analysis available.')
            else:
                return f"LLM analysis failed: HTTP {response.status_code}"

        except Exception as e:
            print(f"LLM analysis error: {e}")
            return "LLM analysis unavailable due to technical issues."

    def analyze_frame(self, frame_image, detections=None):
        """Analyze a single frame (for video processing)"""
        try:
            if hasattr(frame_image, 'shape'):
                from PIL import Image
                import cv2
                frame_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                image_b64 = base64.b64encode(img_byte_arr).decode('utf-8')
            else:
                return "Invalid frame format for analysis."

            detection_context = ""
            if detections:
                animals_found = [d['animal'] for d in detections]
                detection_context = f"\n\nDetected in this frame: {', '.join(set(animals_found))}"

            prompt = f"""Briefly describe what's happening in this wildlife video frame. Focus on animal behavior and environment.{detection_context}

Keep the response concise (2-3 sentences)."""

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No analysis available.')
            else:
                return "Analysis unavailable."

        except Exception as e:
            print(f"Frame analysis error: {e}")
            return "Frame analysis failed."