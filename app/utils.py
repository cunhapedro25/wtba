import cv2
import base64
import threading
import yt_dlp
from pathlib import Path
import time

class DetectionProcessor:
    def __init__(self, detection_model):
        self.model = detection_model
        self.current_results = {}
        self.is_processing = False
        self.is_streaming = False
        self.current_stats = {
            'frames_processed': 0,
            'detections_by_class': {},
            'total_detections': 0
        }
        self.current_video_path = None
        self.conf_threshold = 0.5
        # Add locks for thread safety
        self._stats_lock = threading.Lock()
        self._processing_lock = threading.Lock()
        self._stop_event = threading.Event()

    def process_image(self, image_path, conf_threshold=0.5):
        if not self.model.is_loaded():
            raise Exception("Model not loaded")

        save_dir = Path("runs/detect/predict")
        save_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=str(image_path),
            conf=conf_threshold,
            save=True,
            project="runs/detect",
            name="predict",
            exist_ok=True
        )

        detections = []
        res = results[0]
        if res.boxes is not None:
            for box in res.boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                coords = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = (int(v) for v in coords)
                if cls_id < len(self.model.class_names):
                    detections.append({
                        'animal': self.model.class_names[cls_id],
                        'confidence': float(conf),
                        'bbox': [x1, y1, x2, y2]
                    })

        saved_name = Path(image_path).stem
        saved_path = save_dir / f"{saved_name}.jpg"
        if not saved_path.exists():
            raise Exception(f"Annotated image not found: {saved_path}")

        with open(saved_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        summary = self.generate_summary(detections, saved_name)
        return {
            'image': f"data:image/jpeg;base64,{img_b64}",
            'detections': detections,
            'summary': summary
        }

    def process_video(self, video_path, conf_threshold=0.5):
        if not self.model.is_loaded():
            raise Exception("Model not loaded")

        save_dir = Path("runs/detect/predict")
        save_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=video_path,
            conf=conf_threshold,
            save=True,
            project="runs/detect",
            name="predict",
            exist_ok=True
        )

        detections = []
        for res in results:
            frame_idx = getattr(res, "orig_frame_id", None)
            if res.boxes is not None:
                for box in res.boxes:
                    cls_id = int(box.cls.cpu().numpy()[0])
                    conf = float(box.conf.cpu().numpy()[0])
                    raw_frame = getattr(res, "orig_frame_id", None)
                    frame_idx = int(raw_frame) if raw_frame is not None else None
                    if cls_id < len(self.model.class_names):
                        detections.append({
                            'animal': self.model.class_names[cls_id],
                            'confidence': float(conf),
                            'frame': frame_idx
                        })

        saved_video = save_dir / Path(video_path).name
        if not saved_video.exists():
            raise Exception(f"Annotated video not found: {saved_video}")

        summary = self.generate_summary(detections, Path(video_path).name)
        return {
            'video_path': str(saved_video),
            'detections': detections,
            'summary': summary
        }

    def process_video_async(self, video_path, conf_threshold=0.5):
        def _run():
            with self._processing_lock:
                self.is_processing = True
                self.current_results = {'type': 'video_progress', 'progress': 0, 'detections_so_far': 0}
                try:
                    result = self.process_video(video_path, conf_threshold)
                    self.current_results = {
                        'type': 'video_complete',
                        'detections': result['detections'],
                        'summary': result['summary']
                    }
                except Exception as e:
                    self.current_results = {'type': 'error', 'message': str(e)}
                finally:
                    self.is_processing = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def download_youtube_video(self, url, conf_threshold=0.5):
        """Download YouTube video and prepare it for streaming (same end state as upload)"""
        def _run():
            with self._processing_lock:
                self.is_processing = True
                self.current_results = {'type': 'download_progress', 'status': 'Downloading...'}
                current_dir = Path.cwd()
                upload_dir = current_dir / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)

                ydl_opts = {
                    'outtmpl': str(upload_dir / '%(title)s.%(ext)s'),
                    'format': 'best[ext=mp4]/best',
                    'noplaylist': True,
                }

                try:
                    # Check if we should stop
                    if self._stop_event.is_set():
                        self.current_results = {'type': 'error', 'message': 'Download cancelled'}
                        return

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                        title = info.get('title', 'video')
                        # Clean title for file matching
                        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                        self.current_results = {'type': 'download_progress', 'status': f'Downloading: {title}'}

                        # Check if we should stop before downloading
                        if self._stop_event.is_set():
                            self.current_results = {'type': 'error', 'message': 'Download cancelled'}
                            return

                        ydl.download([url])

                    # Find the downloaded file
                    files = list(upload_dir.glob(f"*{safe_title[:20]}*"))  # Use partial title match
                    if not files:
                        # Fallback: find recent video files
                        files = [
                            f for f in upload_dir.iterdir()
                            if f.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
                        ]
                        if files:
                            files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

                    if not files:
                        raise Exception("Downloaded video not found")

                    video_path = str(files[0])
                    filename = Path(video_path).name

                    # Set up for streaming (same final state as upload_video)
                    self.current_video_path = video_path
                    self.conf_threshold = conf_threshold

                    self.current_results = {
                        'type': 'ready_to_stream',
                        'status': 'ready_to_stream',
                        'filename': filename
                    }

                except Exception as e:
                    self.current_results = {'type': 'error', 'message': str(e)}
                finally:
                    self.is_processing = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _reset_stats(self):
        with self._stats_lock:
            self.current_stats = {
                'frames_processed': 0,
                'detections_by_class': {},
                'total_detections': 0,
                'finished': False
            }

    def _annotate_frame(self, frame_bgr, conf_threshold):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict_silent(frame_rgb, conf_threshold)

        detections_this_frame = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)

                if cls_id < len(self.model.class_names):
                    animal = self.model.class_names[cls_id]
                    detections_this_frame.append({
                        'animal': animal,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })

                    # Thread-safe stats update
                    with self._stats_lock:
                        self.current_stats['total_detections'] += 1
                        cnt = self.current_stats['detections_by_class'].get(animal, 0) + 1
                        self.current_stats['detections_by_class'][animal] = cnt

                    color = self.model.colors[cls_id % len(self.model.colors)]
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    label = f"{animal} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
                    cv2.putText(frame_bgr, label, (x1 + 4, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame_bgr, detections_this_frame

    def generate_video_feed(self, video_path, conf_threshold=0.5):
        """Generate video feed with proper cleanup and stop handling"""
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
                return

            self._reset_stats()
            self.is_streaming = True

            while self.is_streaming and not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_bgr, _ = self._annotate_frame(frame, conf_threshold)

                # Thread-safe frame count update
                with self._stats_lock:
                    self.current_stats['frames_processed'] += 1

                ret2, jpeg = cv2.imencode('.jpg', annotated_bgr)
                if not ret2:
                    continue

                jpg_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')

                # Small delay to prevent overwhelming the client
                time.sleep(0.033)  # ~30 FPS

        except Exception as e:
            print(f"Video feed error: {e}")
        finally:
            if cap:
                cap.release()
            self.is_streaming = False
            with self._stats_lock:
                self.current_stats['finished'] = True

    def get_current_stats(self):
        with self._stats_lock:
            # check if video is loaded and stats are finished
            if not self.current_stats.get('finished', False):
                video_path = getattr(self, 'current_video_path', None)
                if video_path and Path(video_path).exists():
                    try:
                        import cv2
                        cap = cv2.VideoCapture(str(video_path))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        if total_frames > 0 and self.current_stats.get('frames_processed', 0) >= total_frames:
                            self.current_stats['finished'] = True
                    except Exception:
                        pass
            return self.current_stats.copy()

    def generate_summary(self, detections, source):
        if not detections:
            return f"No detections in {source}"
        counts, sums = {}, {}
        for d in detections:
            a = d.get('animal')
            counts[a] = counts.get(a, 0) + 1
            sums[a] = sums.get(a, 0) + d.get('confidence', 0)
        summary = f"Source: {source}\nTotal: {len(detections)}\n"
        for a, cnt in counts.items():
            summary += f"{a}: {cnt} (avg {sums[a]/cnt:.3f})\n"
        return summary

    def stop_all(self):
        """
        Completely stop all processing and reset state
        """
        print("Stopping all processing...")

        # Signal all threads to stop
        self._stop_event.set()

        # Stop streaming immediately
        self.is_streaming = False

        # Wait a moment for threads to respond to stop signal
        time.sleep(0.5)

        # Reset processing flags
        with self._processing_lock:
            self.is_processing = False

        # Clear all state
        self.current_video_path = None
        self.conf_threshold = 0.5
        self.current_results = {}
        self._reset_stats()

        # Clear the stop event so new operations can start
        self._stop_event.clear()

        print("All processing stopped and state reset")

    def prepare_for_new_video(self, video_path, conf_threshold=0.5):
        """Prepare for a new video by ensuring clean state"""
        self.stop_all()

        # Wait a bit more to ensure everything is stopped
        time.sleep(0.2)

        # Set new video parameters
        self.current_video_path = video_path
        self.conf_threshold = conf_threshold
        self._reset_stats()

        print(f"Prepared for new video: {video_path}")