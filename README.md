# Face Recognition Pipeline: MediaPipe + LBPH

A complete face recognition system using **MediaPipe for face detection** and **LBPH (Local Binary Pattern Histograms) for face recognition**. This project demonstrates an "AI Without ML" approach where classical computer vision techniques are used instead of deep learning.

## 🎯 Overview

| Stage | Technology | Description |
|-------|------------|-------------|
| **Detection** | MediaPipe | Locates faces in video frames |
| **Preprocessing** | OpenCV | Extracts and normalizes face regions |
| **Recognition** | LBPH | Identifies faces using texture analysis |

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Camera Feed   │────▶│ MediaPipe Face  │────▶│  Extract Face   │
│                 │     │   Detection     │     │      ROI        │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Display Name   │◀────│ LBPH Recognizer │◀────│   Grayscale     │
│  + Confidence   │     │    Predict      │     │   Conversion    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 📋 Requirements

- Python 3.10+
- Webcam

## 🚀 Installation

```bash
# Clone the repository
cd face_detector

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Step 1: Collect Training Data

Capture face images for training:

```bash
python collect_faces.py
```

- Enter your name when prompted
- Position your face in the camera
- Move slightly between captures for variety
- 50 images will be collected automatically
- Press `q` to quit early, `p` to pause

### Step 2: Train the Model

Train the LBPH recognizer on your collected data:

```bash
python train_model.py
```

This creates:

- `models/face_recognizer.yml` - The trained LBPH model
- `models/label_map.json` - Mapping of IDs to names

### Step 3: Run Face Recognition

Start real-time face recognition:

```bash
python recognize_faces.py
```

- Green box = Recognized face with name
- Red box = Unknown face
- Press `q` to quit

## 📁 Project Structure

```
face_detector/
├── dataset/                  # Training images (auto-generated)
│   └── <person_name>/        # One folder per person
│       └── *.jpg             # Face images
├── models/                   # Trained models (auto-generated)
│   ├── face_recognizer.yml   # LBPH model
│   └── label_map.json        # ID to name mapping
├── face_detector.py          # MediaPipe detection module
├── face_recognizer.py        # LBPH recognition module
├── collect_faces.py          # Data collection script
├── train_model.py            # Training script
├── recognize_faces.py        # Real-time recognition
├── main.py                   # Original face mesh demo
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🔬 How It Works

### Face Detection (MediaPipe)

MediaPipe's Face Detection uses a lightweight CNN-based face detector called BlazeFace. It:

1. Takes a video frame as input
2. Returns bounding box coordinates for detected faces
3. Works in real-time on CPU

### Face Recognition (LBPH)

LBPH is a classical computer vision algorithm that works by:

1. **Local Binary Pattern**: For each pixel, compare with 8 neighbors
   - If neighbor ≥ center: 1, else: 0
   - Creates an 8-bit binary number

2. **Histograms**: Divide face into grid of regions
   - Compute LBP histogram for each region
   - Concatenate all histograms

3. **Recognition**: Compare histograms using Chi-Square distance
   - Lower distance = better match
   - Threshold determines known vs unknown

### Why "AI Without ML"?

- MediaPipe uses a **pre-trained model** (no custom training required)
- LBPH uses **statistical texture analysis**, not neural networks
- The recognition is based on **histogram comparison**, not learned features

## 🎛️ Configuration

Adjust parameters in the respective files:

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `NUM_IMAGES` | collect_faces.py | 50 | Images per person |
| `CONFIDENCE_THRESHOLD` | recognize_faces.py | 80.0 | Lower = stricter matching |
| `min_detection_confidence` | face_detector.py | 0.5 | Detection sensitivity |

## 📝 Adding More People

1. Run `python collect_faces.py` for each new person
2. Re-run `python train_model.py` to update the model
3. The recognizer will now detect all trained people

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Run `collect_faces.py` then `train_model.py` first |
| No face detected | Ensure good lighting, face camera directly |
| Wrong recognition | Collect more training images with varied angles |
| Import error | Ensure `opencv-contrib-python` is installed |

## 📚 References

- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection)
- [OpenCV LBPH](https://docs.opencv.org/4.x/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html)
- [Local Binary Patterns (Wikipedia)](https://en.wikipedia.org/wiki/Local_binary_patterns)

## 📄 License

This project is for educational purposes (Week 13 Assignment - AI Without ML).
