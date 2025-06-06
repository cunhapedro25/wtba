"""
Model management and YOLO integration
"""

from pathlib import Path
from ultralytics import YOLO

class DetectionModel:
    def __init__(self):
        self.model = None
        self.class_names = ['hog', 'rabbit', 'pigeon']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
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

    def predict(self, image_path, conf_threshold=0.5):
        """Run prediction on image"""
        if not self.is_loaded():
            raise Exception("Model not loaded")
        return self.model(image_path, conf=conf_threshold)

    def predict_silent(self, image, conf_threshold=0.5):
        """Run prediction without verbose output"""
        if not self.is_loaded():
            raise Exception("Model not loaded")
        return self.model(image, conf=conf_threshold, verbose=False)