from flask import Blueprint, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from pathlib import Path
from app.models import DetectionModel, LLMAnalyzer
from app.utils import DetectionProcessor

main = Blueprint('main', __name__)

detection_model = DetectionModel()
llm_analyzer = LLMAnalyzer()
processor = DetectionProcessor(detection_model, llm_analyzer)
detection_model.load_model()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload_image', methods=['POST'])
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

@main.route('/upload_video', methods=['POST'])
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

    processor.stop_all()

    processor.current_video_path = str(filepath)
    processor.conf_threshold = conf_threshold

    processor.current_results = {}
    processor._reset_stats()

    return jsonify({'status': 'ready_to_stream', 'filename': filename})

@main.route('/video_feed')
def video_feed():
    video_path = getattr(processor, 'current_video_path', None)
    conf = getattr(processor, 'conf_threshold', 0.5)
    if not video_path:
        return "No video uploaded", 400

    return Response(
        processor.generate_video_feed(video_path, conf),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@main.route('/get_stats')
def get_stats():
    return jsonify(processor.get_current_stats())

@main.route('/youtube_video', methods=['POST'])
def youtube_video():
    data = request.get_json()
    url = data.get('url', '')
    conf_threshold = float(data.get('confidence', 0.5))

    if not url:
        return jsonify({'error': 'No URL provided'})

    try:
        processor.stop_all()

        processor.download_youtube_video(url, conf_threshold)
        return jsonify({'status': 'download_started'})
    except Exception as e:
        return jsonify({'error': str(e)})

@main.route('/get_results')
def get_results():
    return jsonify(processor.current_results)

@main.route('/stop_processing', methods=['POST'])
def stop_processing():
    """
    Interrompe qualquer processamento em andamento e limpa variáveis-chave
    no objeto `processor`.
    """
    try:
        processor.stop_all()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@main.route('/llm/status')
def llm_status():
    """Get LLM connection status and available models"""
    return jsonify(llm_analyzer.get_connection_status())

@main.route('/llm/set_model', methods=['POST'])
def set_llm_model():
    """Set the current LLM model"""
    data = request.get_json()
    model_name = data.get('model_name', '')

    if not model_name:
        return jsonify({'success': False, 'message': 'No model name provided'})

    result = llm_analyzer.set_model(model_name)
    return jsonify(result)

@main.route('/llm/install_model', methods=['POST'])
def install_llm_model():
    """Install a new LLM model"""
    data = request.get_json()
    model_name = data.get('model_name', '')

    if not model_name:
        return jsonify({'success': False, 'message': 'No model name provided'})

    result = llm_analyzer.install_model(model_name)
    return jsonify(result)

@main.route('/llm/installation_status/<model_name>')
def get_installation_status(model_name):
    """Get installation status for a specific model"""
    status = llm_analyzer.get_installation_status(model_name)
    return jsonify(status)

@main.route('/llm/remove_model', methods=['POST'])
def remove_llm_model():
    """Remove an installed LLM model"""
    data = request.get_json()
    model_name = data.get('model_name', '')

    if not model_name:
        return jsonify({'success': False, 'message': 'No model name provided'})

    result = llm_analyzer.remove_model(model_name)
    return jsonify(result)

@main.route('/llm/refresh', methods=['POST'])
def refresh_llm_connection():
    """Refresh LLM connection and model list"""
    llm_analyzer._test_connection()
    return jsonify(llm_analyzer.get_connection_status())
