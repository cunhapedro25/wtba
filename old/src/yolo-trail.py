import os
from pathlib import Path
import pandas as pd
import warnings
import yt_dlp

warnings.filterwarnings('ignore')

from ultralytics import YOLO

class TrailCameraEvaluator:
    def __init__(self, model_path=None, project_dir=None):
        """
        Initialize the trail camera model evaluator

        Args:
            model_path: Path to your trained model (.pt file)
            project_dir: Project directory to organize outputs (optional)
        """
        # Set up project directory structure first
        if project_dir:
            self.project_dir = Path(project_dir)
        else:
            # If we're in src directory, go up one level to project root
            current_dir = Path.cwd()
            if current_dir.name == 'src':
                self.project_dir = current_dir.parent
            else:
                self.project_dir = current_dir

        # Set model path relative to project directory
        if model_path is None:
            self.model_path = self.project_dir / "models" / "best.pt"
        else:
            # If it's a relative path, make it relative to project directory
            model_path = Path(model_path)
            if not model_path.is_absolute():
                self.model_path = self.project_dir / model_path
            else:
                self.model_path = model_path

        self.model = None
        self.class_names = ['hog', 'coyote', 'deer', 'rabbit']

        self.runs_dir = self.project_dir / 'runs'
        self.runs_dir.mkdir(exist_ok=True)

        # Change working directory to project root to ensure runs folder is created there
        os.chdir(self.project_dir)

        print(f"Project directory: {self.project_dir}")
        print(f"Model path: {self.model_path}")

    def load_model(self):
        """Load the trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("Model loaded successfully!")

        # Display model info
        self._display_model_info()
        return self.model

    def _display_model_info(self):
        """Display information about the loaded model"""
        print("\n" + "="*50)
        print("MODEL INFORMATION")
        print("="*50)

        model_info = self.model.info()
        print(f"Parameters: {model_info[0] if model_info else 'N/A'}")
        print(f"GFLOPs: {model_info[1] if model_info and len(model_info) > 1 else 'N/A'}")

        print(f"\nDetection Classes:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {i}: {class_name}")

    def validate_model(self, dataset_path):
        """
        Validate the model and display detailed metrics

        Args:
            dataset_path: Path to dataset containing data.yaml
        """
        if not self.model:
            print("Please load model first using load_model()")
            return

        data_yaml_path = Path(dataset_path) / "data.yaml"
        if not data_yaml_path.exists():
            print(f"Data configuration file not found: {data_yaml_path}")
            return

        print("Validating model...")
        val_results = self.model.val(data=str(data_yaml_path))

        print("\n" + "="*50)
        print("VALIDATION METRICS")
        print("="*50)

        print(f"Overall mAP@0.5: {val_results.box.map50:.4f}")
        print(f"Overall mAP@0.5:0.95: {val_results.box.map:.4f}")
        print(f"Overall Precision: {val_results.box.mp:.4f}")
        print(f"Overall Recall: {val_results.box.mr:.4f}")

        if val_results.box.mp > 0 and val_results.box.mr > 0:
            f1_score = 2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr)
            print(f"Overall F1-Score: {f1_score:.4f}")

        print("\nPer-Class Metrics:")
        print("-" * 70)
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<10} {'mAP@0.5':<12} {'F1-Score':<10}")
        print("-" * 70)

        for i, class_name in enumerate(self.class_names):
            if i < len(val_results.box.ap_class_index):
                precision = val_results.box.p[i] if i < len(val_results.box.p) else 0
                recall = val_results.box.r[i] if i < len(val_results.box.r) else 0
                map50 = val_results.box.ap50[i] if i < len(val_results.box.ap50) else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                print(f"{class_name:<10} {precision:<12.4f} {recall:<10.4f} {map50:<12.4f} {f1:<10.4f}")

        return val_results

    def detect_single_image(self, image_path, conf_threshold=0.5):
        """
        Detect animals in a single image

        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold for detections
        """
        if not self.model:
            print("Please load model first using load_model()")
            return None

        if not Path(image_path).exists():
            print(f"Image not found: {image_path}")
            return None

        print(f"Running detection on: {image_path}")
        print(f"Confidence threshold: {conf_threshold}")

        results = self.model(image_path, conf=conf_threshold, save=True)
        return self._process_detection_results(results)

    def detect_all_images(self, images_dir, conf_threshold=0.5):
        """
        Detect animals in all images in a directory

        Args:
            images_dir: Directory containing images
            conf_threshold: Confidence threshold for detections
        """
        if not self.model:
            print("Please load model first using load_model()")
            return None

        images_path = Path(images_dir)
        if not images_path.exists():
            print(f"Directory not found: {images_dir}")
            return None

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in images_path.iterdir()
                       if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in: {images_dir}")
            return None

        print(f"Found {len(image_files)} images in {images_dir}")
        print(f"Confidence threshold: {conf_threshold}")

        results = self.model(str(images_path), conf=conf_threshold, save=True)
        return self._process_detection_results(results)

    def detect_video(self, video_path, conf_threshold=0.5):
        """
        Detect animals in a video file

        Args:
            video_path: Path to video file
            conf_threshold: Confidence threshold for detections
        """
        if not self.model:
            print("Please load model first using load_model()")
            return None

        if not Path(video_path).exists():
            print(f"Video not found: {video_path}")
            return None

        print(f"Running detection on video: {video_path}")
        print(f"Confidence threshold: {conf_threshold}")

        results = self.model(video_path, conf=conf_threshold, save=True)
        return self._process_detection_results(results)

    def detect_youtube_video(self, youtube_url, conf_threshold=0.5):
        """
        Download and detect animals in a YouTube video

        Args:
            youtube_url: YouTube video URL
            conf_threshold: Confidence threshold for detections
        """
        if not self.model:
            print("Please load model first using load_model()")
            return None

        print(f"Downloading YouTube video: {youtube_url}")

        # Create downloads directory
        downloads_dir = self.project_dir / 'downloads'
        downloads_dir.mkdir(exist_ok=True)

        # Configure yt-dlp options
        ydl_opts = {
            'outtmpl': str(downloads_dir / '%(title)s.%(ext)s'),
            'format': 'best[ext=mp4]/best',
            'noplaylist': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info
                info = ydl.extract_info(youtube_url, download=False)
                video_title = info.get('title', 'video')

                # Download video
                ydl.download([youtube_url])

                # Find downloaded file
                video_files = list(downloads_dir.glob(f"{video_title}.*"))
                if not video_files:
                    # Try finding any recently created video file
                    video_files = [f for f in downloads_dir.iterdir()
                                   if f.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}]

                if not video_files:
                    print("Failed to find downloaded video file")
                    return None

                video_path = video_files[0]
                print(f"Video downloaded: {video_path}")

                # Run detection
                return self.detect_video(video_path, conf_threshold)

        except Exception as e:
            print(f"Error downloading YouTube video: {e}")
            return None

    def _process_detection_results(self, results):
        """Process and display detection results"""
        detections = []

        for r in results:
            img_path = r.path
            img_name = Path(img_path).name

            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()

                    if class_id < len(self.class_names):
                        animal = self.class_names[class_id]
                        detections.append({
                            'image': img_name,
                            'animal': animal,
                            'confidence': confidence,
                            'bbox': bbox
                        })

        if detections:
            df_detections = pd.DataFrame(detections)
            print(f"\nDetected {len(detections)} animals:")

            animal_counts = df_detections['animal'].value_counts()
            for animal, count in animal_counts.items():
                avg_conf = df_detections[df_detections['animal'] == animal]['confidence'].mean()
                print(f"  {animal}: {count} detections (avg confidence: {avg_conf:.3f})")

            return df_detections
        else:
            print("No animals detected.")
            return None

    def export_model(self, format='onnx'):
        """
        Export model to different formats

        Args:
            format: Export format ('onnx', 'tflite', 'coreml', 'engine')
        """
        if not self.model:
            print("Please load model first using load_model()")
            return

        print(f"Exporting model to {format.upper()} format...")
        export_path = self.model.export(format=format)
        print(f"Model exported to: {export_path}")
        return export_path


def display_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print("    🦌 TRAIL CAMERA ANIMAL DETECTION SYSTEM 🦌")
    print("="*60)
    print("1. 📸 Detect animals in a single image")
    print("2. 📁 Detect animals in all images (directory)")
    print("3. 🎥 Detect animals in a video file")
    print("4. 🌐 Detect animals in a YouTube video")
    print("5. 📊 Validate model performance")
    print("6. 💾 Export model")
    print("7. ❌ Exit")
    print("="*60)


def get_user_input(prompt, input_type=str):
    """Get user input with type validation"""
    while True:
        try:
            value = input(prompt)
            if input_type == float:
                return float(value)
            elif input_type == int:
                return int(value)
            else:
                return value
        except ValueError:
            print(f"Please enter a valid {input_type.__name__}.")


def main():
    # Initialize evaluator - it will automatically detect the correct paths
    evaluator = TrailCameraEvaluator()

    # Load model
    try:
        evaluator.load_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure your model file exists at: {evaluator.model_path}")
        return

    while True:
        display_menu()

        choice = get_user_input("Enter your choice (1-7): ", int)

        if choice == 1:
            # Single image detection
            image_path = get_user_input("Enter path to image: ")
            conf = get_user_input("Enter confidence threshold (0.0-1.0) [default: 0.5]: ")
            conf = float(conf) if conf else 0.5

            print("\n" + "-"*50)
            evaluator.detect_single_image(image_path, conf)

        elif choice == 2:
            # All images detection
            images_dir = get_user_input("Enter path to images directory: ")
            conf = get_user_input("Enter confidence threshold (0.0-1.0) [default: 0.5]: ")
            conf = float(conf) if conf else 0.5

            print("\n" + "-"*50)
            evaluator.detect_all_images(images_dir, conf)

        elif choice == 3:
            # Video detection
            video_path = get_user_input("Enter path to video file: ")
            conf = get_user_input("Enter confidence threshold (0.0-1.0) [default: 0.5]: ")
            conf = float(conf) if conf else 0.5

            print("\n" + "-"*50)
            evaluator.detect_video(video_path, conf)

        elif choice == 4:
            # YouTube video detection
            youtube_url = get_user_input("Enter YouTube URL: ")
            conf = get_user_input("Enter confidence threshold (0.0-1.0) [default: 0.5]: ")
            conf = float(conf) if conf else 0.5

            print("\n" + "-"*50)
            evaluator.detect_youtube_video(youtube_url, conf)

        elif choice == 5:
            # Model validation
            dataset_path = get_user_input("Enter path to dataset directory: ")

            print("\n" + "-"*50)
            evaluator.validate_model(dataset_path)

        elif choice == 6:
            # Export model
            print("\nAvailable formats: onnx, tflite, coreml, engine")
            format_choice = get_user_input("Enter export format [default: onnx]: ")
            format_choice = format_choice.lower() if format_choice else 'onnx'

            print("\n" + "-"*50)
            evaluator.export_model(format_choice)

        elif choice == 7:
            # Exit
            print("\nGoodbye! 👋")
            break

        else:
            print("Invalid choice. Please enter a number between 1-7.")

        # Wait for user to continue
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()