import os
import cv2
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import yt_dlp
from ultralytics import YOLO

class TrailCameraGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Trail Camera Detection System")
        self.root.geometry("1200x800")

        # Core components
        self.model = None
        self.class_names = ['hog', 'coyote', 'deer', 'rabbit']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        # Video processing
        self.video_cap = None
        self.video_thread = None
        self.is_playing = False
        self.current_frame = None

        # Setup paths
        self.setup_paths()

        # Build interface
        self.create_widgets()
        self.load_model()

    def setup_paths(self):
        current_dir = Path.cwd()
        self.project_dir = current_dir.parent if current_dir.name == 'src' else current_dir
        self.model_path = self.project_dir / "models" / "best.pt"
        self.runs_dir = self.project_dir / 'runs'
        self.runs_dir.mkdir(exist_ok=True)
        os.chdir(self.project_dir)

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Control panel
        self.create_control_panel(main_frame)

        # Display area
        self.create_display_area(main_frame)

        # Results panel
        self.create_results_panel(main_frame)

    def create_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(0, 0), pady=(0, 10))

        # File selection buttons
        ttk.Button(control_frame, text="Select Image", command=self.select_image).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Select Video", command=self.select_video).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="YouTube URL", command=self.youtube_dialog).grid(row=0, column=2, padx=5)

        # Confidence threshold
        ttk.Label(control_frame, text="Confidence:").grid(row=0, column=3, padx=(20, 5))
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, variable=self.conf_var, orient=tk.HORIZONTAL, length=150)
        conf_scale.grid(row=0, column=4, padx=5)
        self.conf_label = ttk.Label(control_frame, text="0.50")
        self.conf_label.grid(row=0, column=5, padx=5)
        conf_scale.configure(command=self.update_conf_label)

        # Video controls
        self.video_controls = ttk.Frame(control_frame)
        self.video_controls.grid(row=1, column=0, columnspan=6, pady=(10, 0))

        self.play_btn = ttk.Button(self.video_controls, text="Play", command=self.toggle_video, state='disabled')
        self.play_btn.grid(row=0, column=0, padx=5)

        ttk.Button(self.video_controls, text="Stop", command=self.stop_video).grid(row=0, column=1, padx=5)

        # Progress bar for video
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.video_controls, variable=self.progress_var, length=300)
        self.progress_bar.grid(row=0, column=2, padx=20)

    def create_display_area(self, parent):
        display_frame = ttk.LabelFrame(parent, text="Detection View", padding="5")
        display_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

        # Canvas for image/video display
        self.canvas = tk.Canvas(display_frame, bg='black', width=640, height=480)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbars
        v_scroll = ttk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scroll = ttk.Scrollbar(display_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

    def create_results_panel(self, parent):
        results_frame = ttk.LabelFrame(parent, text="Detection Results", padding="5")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)

        # Summary
        self.summary_text = tk.Text(results_frame, height=8, width=40)
        self.summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Detailed results
        self.results_tree = ttk.Treeview(results_frame, columns=('Animal', 'Confidence'), show='headings', height=15)
        self.results_tree.heading('Animal', text='Animal')
        self.results_tree.heading('Confidence', text='Confidence')
        self.results_tree.column('Animal', width=100)
        self.results_tree.column('Confidence', width=100)
        self.results_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Results scrollbar
        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        results_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.results_tree.configure(yscrollcommand=results_scroll.set)

    def load_model(self):
        try:
            if not self.model_path.exists():
                messagebox.showerror("Error", f"Model not found: {self.model_path}")
                return

            self.model = YOLO(str(self.model_path))
            self.update_summary(f"Model loaded successfully\nClasses: {', '.join(self.class_names)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def update_conf_label(self, value):
        self.conf_label.config(text=f"{float(value):.2f}")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if file_path:
            self.process_image(file_path)

    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm")]
        )
        if file_path:
            self.setup_video(file_path)

    def youtube_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("YouTube Video")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Enter YouTube URL:").pack(pady=10)
        url_entry = ttk.Entry(dialog, width=50)
        url_entry.pack(pady=5)
        url_entry.focus()

        def download_and_process():
            url = url_entry.get().strip()
            if not url:
                return

            dialog.destroy()
            self.download_youtube_video(url)

        ttk.Button(dialog, text="Download & Process", command=download_and_process).pack(pady=10)

        # Bind Enter key
        url_entry.bind('<Return>', lambda e: download_and_process())

    def process_image(self, image_path):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded")
            return

        try:
            # Run detection
            results = self.model(image_path, conf=self.conf_var.get())

            # Load and process image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            detections = self.draw_detections(image, results[0])
            self.display_image(image)
            self.show_results(detections, Path(image_path).name)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")

    def setup_video(self, video_path):
        self.stop_video()

        try:
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened():
                raise ValueError("Cannot open video file")

            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)

            self.play_btn.config(state='normal')
            self.update_summary(f"Video loaded: {Path(video_path).name}\nFrames: {self.total_frames}\nFPS: {self.fps:.1f}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")

    def toggle_video(self):
        if not self.video_cap or not self.model:
            return

        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="Play")
        else:
            self.is_playing = True
            self.play_btn.config(text="Pause")
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()

    def stop_video(self):
        self.is_playing = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.play_btn.config(text="Play", state='disabled')
        self.progress_var.set(0)

    def process_video(self):
        frame_count = 0
        all_detections = []

        while self.is_playing and self.video_cap:
            ret, frame = self.video_cap.read()
            if not ret:
                self.is_playing = False
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run detection every few frames for performance
            if frame_count % 3 == 0:
                results = self.model(frame_rgb, conf=self.conf_var.get(), verbose=False)
                detections = self.draw_detections(frame_rgb, results[0])
                all_detections.extend(detections)

            self.display_image(frame_rgb)

            # Update progress
            progress = (frame_count / self.total_frames) * 100
            self.progress_var.set(progress)

            frame_count += 1

            # Control playback speed
            self.root.after(int(1000 / self.fps))

        # Show final results
        if all_detections:
            self.show_results(all_detections, "Video")

        self.root.after(0, lambda: self.play_btn.config(text="Play"))

    def download_youtube_video(self, url):
        try:
            downloads_dir = self.project_dir / 'downloads'
            downloads_dir.mkdir(exist_ok=True)

            ydl_opts = {
                'outtmpl': str(downloads_dir / '%(title)s.%(ext)s'),
                'format': 'best[ext=mp4]/best',
                'noplaylist': True,
            }

            self.update_summary("Downloading YouTube video...")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'video')
                ydl.download([url])

                # Find downloaded file
                video_files = list(downloads_dir.glob(f"{video_title}.*"))
                if not video_files:
                    video_files = [f for f in downloads_dir.iterdir()
                                   if f.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}]

                if video_files:
                    self.setup_video(str(video_files[0]))
                else:
                    messagebox.showerror("Error", "Downloaded video file not found")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to download video: {e}")

    def draw_detections(self, image, results):
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                if class_id < len(self.class_names):
                    animal = self.class_names[class_id]
                    color = self.colors[class_id % len(self.colors)]

                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    label = f"{animal}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    detections.append({
                        'animal': animal,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })

        return detections

    def display_image(self, image):
        # Resize image to fit canvas while maintaining aspect ratio
        h, w = image.shape[:2]
        canvas_w, canvas_h = 640, 480

        scale = min(canvas_w/w, canvas_h/h)
        new_w, new_h = int(w*scale), int(h*scale)

        image_resized = cv2.resize(image, (new_w, new_h))

        # Convert to PhotoImage
        image_pil = Image.fromarray(image_resized)
        self.photo = ImageTk.PhotoImage(image_pil)

        # Display on canvas
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.canvas.create_image(new_w//2, new_h//2, image=self.photo)

    def show_results(self, detections, source):
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        if not detections:
            self.update_summary(f"No animals detected in {source}")
            return

        # Count detections by animal
        df = pd.DataFrame(detections)
        animal_counts = df['animal'].value_counts()

        # Update summary
        summary = f"Source: {source}\nTotal detections: {len(detections)}\n\n"
        for animal, count in animal_counts.items():
            avg_conf = df[df['animal'] == animal]['confidence'].mean()
            summary += f"{animal}: {count} ({avg_conf:.3f} avg conf)\n"

        self.update_summary(summary)

        # Update detailed results
        for detection in detections:
            self.results_tree.insert('', 'end', values=(
                detection['animal'],
                f"{detection['confidence']:.3f}"
            ))

    def update_summary(self, text):
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, text)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TrailCameraGUI()
    app.run()