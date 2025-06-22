"""
Model management and YOLO integration
"""
import threading
import requests
import base64
import subprocess
import platform
import time
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
    def __init__(self, ollama_url="http://localhost:11434", model_name=None):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.available_models = []
        self.installed_models = []
        self.is_connected = False
        self.installation_status = {}
        self._popular_models = [
            # Vision models
            {"name": "llava:7b", "description": "7B vision model - Good balance of speed and quality", "size": "4.7GB", "type": "vision"},
            {"name": "llava:13b", "description": "13B vision model - Higher quality, slower", "size": "8.0GB", "type": "vision"},
            {"name": "llava:34b", "description": "34B vision model - Best quality, requires powerful hardware", "size": "20GB", "type": "vision"},
            {"name": "bakllava", "description": "BakLLaVA - Alternative vision model", "size": "4.4GB", "type": "vision"},
            {"name": "moondream", "description": "Moondream - Lightweight vision model", "size": "1.7GB", "type": "vision"},
            {"name": "llava-phi3", "description": "LLaVA Phi3 - Microsoft's efficient vision model", "size": "2.9GB", "type": "vision"},
            {"name": "gemma:2b", "description": "Gemma 2B - Ultra-lightweight model by Google", "size": "1.4GB", "type": "vision"},
            {"name": "gemma:7b", "description": "Gemma 7B - Efficient model with good performance", "size": "4.8GB", "type": "vision"}
        ]
        # Try to start Ollama if installed but not running
        if not self._check_ollama_running():
            self._try_start_ollama()

        # Test connection after possibly starting Ollama
        self._test_connection()

    def _check_ollama_running(self):
        """Check if Ollama is already running by trying to connect"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _try_start_ollama(self):
        """Attempt to start Ollama using the ollama serve command"""
        print("Trying to start Ollama...")

        try:
            # First, check if ollama command is available
            result = subprocess.run(
                ["ollama", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                print("Ollama command not found or not working")
                return

            print(f"Found Ollama: {result.stdout.strip()}")

            # Start Ollama serve in the background
            print("Starting Ollama serve...")

            # Platform-specific process creation
            system = platform.system()

            if system == "Windows":
                # On Windows, use CREATE_NO_WINDOW to prevent console window
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                # On macOS and Linux
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            print("Ollama serve started in background")

            # Wait a bit for Ollama to start up
            print("Waiting for Ollama to start...")
            for i in range(10):  # Wait up to 10 seconds
                time.sleep(1)
                if self._check_ollama_running():
                    print(f"Ollama is now running after {i+1} seconds")
                    return

            print("Ollama may still be starting up...")

        except subprocess.TimeoutExpired:
            print("Timeout checking for Ollama command")
        except FileNotFoundError:
            print("Ollama command not found. Please install Ollama from https://ollama.com")
        except Exception as e:
            print(f"Error trying to start Ollama: {e}")
            print("You may need to start Ollama manually by running 'ollama serve' in a terminal")
    def _test_connection(self):
        """Test connection to Ollama and get available models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.is_connected = True
                models_data = response.json().get('models', [])

                # Helper function to determine if a model is a vision model
                def is_vision_model(model_name):
                    vision_keywords = ['llava', 'bakllava', 'moondream', 'llava-phi3', 'gemma']
                    return any(keyword in model_name.lower() for keyword in vision_keywords)

                self.installed_models = [
                    {
                        'name': m['name'],
                        'size': m.get('size', 0),
                        'modified_at': m.get('modified_at', ''),
                        'details': m.get('details', {}),
                        'type': 'vision' if is_vision_model(m['name']) else 'text'  # Add this line
                    }
                    for m in models_data
                ]

                # Auto-select first available vision model if none specified
                if not self.model_name:
                    vision_models = [m['name'] for m in self.installed_models
                                     if m['type'] == 'vision']  # Use the type field instead
                    if vision_models:
                        self.model_name = vision_models[0]
                        print(f"Auto-selected model: {self.model_name}")
                    else:
                        print("No vision models found. Please install a vision model.")

                print(f"Connected to Ollama. Installed models: {[m['name'] for m in self.installed_models]}")

            else:
                self.is_connected = False
                print(f"Ollama connection failed: {response.status_code}")
        except Exception as e:
            self.is_connected = False
            print(f"Ollama connection error: {e}")

    def get_connection_status(self):
        """Get current connection status and model info"""
        return {
            'connected': self.is_connected,
            'current_model': self.model_name,
            'installed_models': self.installed_models,
            'popular_models': self._popular_models,
            'installation_status': self.installation_status
        }

    def set_model(self, model_name):
        """Set the current model"""
        # Check if model is installed
        installed_model_names = [m['name'] for m in self.installed_models]
        if model_name in installed_model_names:
            self.model_name = model_name
            return {'success': True, 'message': f'Model set to {model_name}'}
        else:
            return {'success': False, 'message': f'Model {model_name} is not installed'}

    def install_model(self, model_name):
        """Install a model asynchronously"""
        if not self.is_connected:
            return {'success': False, 'message': 'Not connected to Ollama'}

        if model_name in [m['name'] for m in self.installed_models]:
            return {'success': False, 'message': 'Model already installed'}

        # Start installation in background thread
        def _install():
            try:
                self.installation_status[model_name] = {
                    'status': 'downloading',
                    'progress': 0,
                    'message': 'Starting download...'
                }

                # Use Ollama API to pull model
                response = requests.post(
                    f"{self.ollama_url}/api/pull",
                    json={'name': model_name},
                    stream=True,
                    timeout=3600  # 1 hour timeout for large models
                )

                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            try:
                                import json
                                data = json.loads(line.decode('utf-8'))
                                status = data.get('status', '')

                                if 'pulling' in status.lower():
                                    self.installation_status[model_name] = {
                                        'status': 'downloading',
                                        'progress': 50,  # Rough estimate
                                        'message': status
                                    }
                                elif 'verifying' in status.lower():
                                    self.installation_status[model_name] = {
                                        'status': 'verifying',
                                        'progress': 90,
                                        'message': status
                                    }
                                elif 'success' in status.lower() or data.get('status') == 'success':
                                    self.installation_status[model_name] = {
                                        'status': 'completed',
                                        'progress': 100,
                                        'message': 'Installation completed'
                                    }
                                    # Refresh installed models list
                                    self._test_connection()
                                    break
                            except json.JSONDecodeError:
                                continue

                    # Final success check
                    if model_name not in self.installation_status or \
                            self.installation_status[model_name]['status'] != 'completed':
                        self.installation_status[model_name] = {
                            'status': 'completed',
                            'progress': 100,
                            'message': 'Installation completed'
                        }
                        self._test_connection()
                else:
                    self.installation_status[model_name] = {
                        'status': 'error',
                        'progress': 0,
                        'message': f'Installation failed: {response.status_code}'
                    }

            except Exception as e:
                self.installation_status[model_name] = {
                    'status': 'error',
                    'progress': 0,
                    'message': f'Installation failed: {str(e)}'
                }

        thread = threading.Thread(target=_install, daemon=True)
        thread.start()

        return {'success': True, 'message': f'Installation of {model_name} started'}

    def get_installation_status(self, model_name):
        """Get installation status for a specific model"""
        return self.installation_status.get(model_name, {'status': 'not_started'})

    def remove_model(self, model_name):
        """Remove an installed model"""
        if not self.is_connected:
            return {'success': False, 'message': 'Not connected to Ollama'}

        try:
            response = requests.delete(f"{self.ollama_url}/api/delete", json={'name': model_name})
            if response.status_code == 200:
                # Refresh installed models list
                self._test_connection()
                # Clear current model if it was removed
                if self.model_name == model_name:
                    self.model_name = None
                return {'success': True, 'message': f'Model {model_name} removed'}
            else:
                return {'success': False, 'message': f'Failed to remove model: {response.status_code}'}
        except Exception as e:
            return {'success': False, 'message': f'Error removing model: {str(e)}'}

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
        if not self.is_connected:
            return "LLM service not available - please check Ollama connection."

        if not self.model_name:
            return "No LLM model selected - please select a vision model."

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
        if not self.is_connected or not self.model_name:
            return "LLM analysis unavailable."

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