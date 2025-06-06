# Where The Boars At - (WTBA)

A project developed for the AOOP UC at IPVC, focused on detecting animals—such as boars, pigeons, and rabbits—in the wild using trail cameras or uploaded images via a web interface.

## Project Structure

```
wtba/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── utils.py
│   └── templates/
│       └── index.html
│   └──  static/
│       └── style.css
├── models/
│   └── best.pt
├── uploads/
├── downloads/
├── requirements.txt
├── run.py
└── README.md
```

## Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager

### 2. Installation

1. **Clone/Create the project directory:**
```bash
git clone https://github.com/cunhapedro25/wtba
```

2. **Create virtual environment:**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run model Training**
```bash
yolo detect train data=./wildlife_dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=640
```

5. **Add your YOLO model:**
   - Place your trained model file `best.pt` in the `models/` directory

### 3. Running the Application

```bash
python run.py
```

The application will be available at `http://127.0.0.1:5000`

### 4. Usage

1. **Image Detection:** Upload an image file to detect wildlife
2. **Video Detection:** Upload a video file for frame-by-frame analysis
3. **YouTube Processing:** Enter a YouTube URL to download and process
4. **Confidence Threshold:** Adjust the detection sensitivity (0.1 - 1.0)

## Features

- Real-time image processing with bounding box visualization
- Video processing with progress tracking
- YouTube video download and processing
- Configurable confidence thresholds
- Responsive web interface with Tailwind CSS
- Support for multiple animal classes: hog, coyote, deer, rabbit

## API Endpoints

- `GET /` - Main interface
- `POST /upload_image` - Process uploaded image
- `POST /upload_video` - Process uploaded video
- `POST /youtube_video` - Download and process YouTube video
- `GET /get_results` - Poll processing results

## Dependencies

See `requirements.txt` for complete list of dependencies including:
- Flask for web framework
- OpenCV for image processing
- Ultralytics YOLO for object detection
- yt-dlp for YouTube downloads
- NumPy for numerical operations