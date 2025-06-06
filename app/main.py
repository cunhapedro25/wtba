from flask import Blueprint, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from pathlib import Path
from app.models import DetectionModel
from app.utils import DetectionProcessor

main_bp = Blueprint('main', __name__)

detection_model = DetectionModel()
processor = DetectionProcessor(detection_model)
detection_model.load_model()

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    conf_threshold = float(request.form.get('confidence', 0.5))

    try:
        filename = secure_filename(file.filename)
        uploads_dir = Path.cwd() / 'uploads'
        uploads_dir.mkdir(parents=True, exist_ok=True)
        filepath = uploads_dir / filename
        file.save(str(filepath))

        results = processor.process_image(str(filepath), conf_threshold)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})

@main_bp.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    conf_threshold = float(request.form.get('confidence', 0.5))
    filename = secure_filename(file.filename)
    uploads_dir = Path.cwd() / 'uploads'
    uploads_dir.mkdir(parents=True, exist_ok=True)
    filepath = uploads_dir / filename
    file.save(str(filepath))

    processor.current_video_path = str(filepath)
    processor.conf_threshold = conf_threshold

    return jsonify({'status': 'ready_to_stream', 'filename': filename})

@main_bp.route('/video_feed')
def video_feed():
    video_path = getattr(processor, 'current_video_path', None)
    conf = getattr(processor, 'conf_threshold', 0.5)
    if not video_path:
        return "No video uploaded", 400

    return Response(
        processor.generate_video_feed(video_path, conf),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@main_bp.route('/get_stats')
def get_stats():
    return jsonify(processor.get_current_stats())

@main_bp.route('/youtube_video', methods=['POST'])
def youtube_video():
    data = request.get_json()
    url = data.get('url', '')
    conf_threshold = float(data.get('confidence', 0.5))

    if not url:
        return jsonify({'error': 'No URL provided'})

    try:
        processor.download_and_process_youtube(url, conf_threshold)
        return jsonify({'status': 'download_started'})
    except Exception as e:
        return jsonify({'error': str(e)})

@main_bp.route('/get_results')
def get_results():
    return jsonify(processor.current_results)
