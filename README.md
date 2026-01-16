# Hand Gesture Recognition for Edge Devices

A real-time hand gesture recognition system optimized for edge computing devices (Raspberry Pi 4 and Google Coral TPU). This project provides a complete pipeline from model training to deployment with multiple quantization strategies for performance optimization.

## 🎯 Features

- **Real-time hand gesture detection** using MediaPipe-based models
- **Multiple quantization formats** (FP32, FP16, INT8, Edge TPU)
- **Optimized for edge devices** (Raspberry Pi 4, Google Coral TPU)
- **Dual classification system**:
  - Static hand pose recognition (keypoint classifier)
  - Dynamic gesture tracking (point history classifier)
- **Comprehensive benchmarking suite** for performance evaluation
- **System optimization scripts** for consistent benchmarking on Raspberry Pi

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Training](#model-training)
- [Benchmarking](#benchmarking)
- [Model Variants](#model-variants)
- [Gesture Classes](#gesture-classes)
- [Performance Considerations](#performance-considerations)

## 💻 System Requirements

### Raspberry Pi 4
- Raspberry Pi 4 (2GB+ RAM recommended)
- Python 3.9+
- Camera module or USB webcam
- Optional: Google Coral USB Accelerator

### Development Machine
- Python 3.9+
- TensorFlow 2.10.0
- Webcam

## 📦 Installation

### On Raspberry Pi (with TFLite Runtime)

```bash
# Install dependencies
python3.9 -m pip install tflite-runtime==2.9.1 "numpy<2" "opencv-contrib-python>=4.6.0.66" "scikit-learn>=1.1.0" "matplotlib>=3.5.0" psutil
```

### On Development Machine (with TensorFlow)

```bash
# Install dependencies
pip install "numpy<2" "opencv-contrib-python>=4.6.0.66" tensorflow==2.10.0 "scikit-learn>=1.1.0" "matplotlib>=3.5.0"

# Note: protobuf==3.20.3 may show dependency warnings but the code will run
```

## 📂 Project Structure

```
hand-gesture/
├── app.py                          # Main application for real-time gesture recognition
├── benchmark.py                    # Automated benchmarking script
├── rec.py                          # Recording utility
├── keypoint_classification.ipynb   # Training notebook for keypoint classifier
├── point_history_classification.ipynb  # Training notebook for gesture history classifier
├── prepare_benchmark.sh            # System preparation script for benchmarking
├── run_benchmark_suite.sh          # Automated benchmark suite runner
├── restore_system.sh               # System restoration script
├── model/
│   ├── palm_detection/             # Palm detection models
│   │   ├── palm_detection_fp32.tflite
│   │   └── palm_detection_fp16.tflite
│   ├── hand_landmark/              # Hand landmark models
│   │   ├── hand_landmark_fp32.tflite
│   │   ├── hand_landmark_fp16.tflite
│   │   ├── hand_landmark_int8.tflite
│   │   └── hand_landmark_edgetpu.tflite
│   ├── keypoint_classifier/        # Static pose classifier
│   │   ├── keypoint_classifier_fp32.tflite
│   │   ├── keypoint_classifier_fp16.tflite
│   │   ├── keypoint_classifier_int8.tflite
│   │   ├── keypoint_classifier_edgetpu.tflite
│   │   └── keypoint_classifier_label.csv
│   └── point_history_classifier/   # Dynamic gesture classifier
│       ├── point_history_classifier_fp32.tflite
│       ├── point_history_classifier_fp16.tflite
│       ├── point_history_classifier_int8.tflite
│       ├── point_history_classifier_edgetpu.tflite
│       └── point_history_classifier_label.csv
└── utils/
    ├── cvfpscalc.py               # FPS calculation utility
    ├── temperature_monitor.py      # Temperature monitoring for Pi
    └── utils.py                    # General utilities
```

## 🚀 Usage

### Running the Application

The application uses model configuration variables at the top of `app.py`:

```python
KEYPOINT_MODEL_TYPE = "fp32"         # Options: 'fp32', 'fp16', 'int8', 'edgetpu'
POINT_HISTORY_MODEL_TYPE = "fp32"   # Options: 'fp32', 'fp16', 'int8', 'edgetpu'
HAND_LANDMARK_MODEL_TYPE = "fp32"   # Options: 'fp32', 'fp16', 'int8', 'edgetpu'
PALM_DETECTION_MODEL_TYPE = "fp32"  # Options: 'fp32', 'fp16'
```

**Basic usage:**
```bash
python3 app.py
```

**With custom camera and detection threshold:**
```bash
python3 app.py --device 0 --width 960 --height 540 --min_detection_confidence 0.6
```

**Arguments:**
- `--device`: Camera device index (default: 0)
- `--width`: Camera width (default: 960)
- `--height`: Camera height (default: 540)
- `--min_detection_confidence`: Minimum detection confidence (default: 0.6)
- `--disable_image_flip`: Disable horizontal image flip

**Controls:**
- Press `ESC` to exit

## 🎓 Model Training

### Training the Keypoint Classifier

1. Open `keypoint_classification.ipynb`
2. Configure export settings:
   ```python
   EXPORT_INT8_FULL = False
   EXPORT_FP16 = False
   EXPORT_DYNAMIC = False
   EXPORT_INT8_PRUNED = True      # Enable pruned INT8 export
   EXPORT_FP16_PRUNED = True       # Enable pruned FP16 export
   COMPILE_EDGE_TPU = True         # Compile for Coral TPU
   ```
3. Run all cells to train and export models

### Training the Point History Classifier

1. Open `point_history_classification.ipynb`
2. Configure export settings similarly
3. Run all cells to train and export models

## 📊 Benchmarking

### Quick Benchmark

```bash
python3 benchmark.py
```

This runs 3 iterations and outputs results to `benchmark_results.csv`.

### Full Automated Benchmark Suite (Raspberry Pi only)

```bash
# Prepare system for benchmarking
sudo ./prepare_benchmark.sh

# Run benchmark
sudo python3.9 benchmark.py

# Restore normal system settings
sudo ./restore_system.sh
```

Or use the automated suite:

```bash
sudo ./run_benchmark_suite.sh
```

The preparation script:
- Sets CPU governor to performance mode
- Locks CPU frequency to maximum
- Stops unnecessary background services
- Disables swap
- Clears system caches
- Optimizes I/O scheduler

**Benchmark Metrics:**
- Average cycle time
- Preprocessing time
- Inference time
- Memory usage
- Model confidence scores
- Temperature readings (Raspberry Pi)

## 🔧 Model Variants

### FP32 (Full Precision)
- **Size:** Largest (~6-7 KB for classifiers)
- **Speed:** Slowest
- **Accuracy:** Highest
- **Use case:** Development, baseline comparison

### FP16 (Half Precision)
- **Size:** ~50% of FP32
- **Speed:** Moderate
- **Accuracy:** Near FP32
- **Use case:** Balanced performance on CPU

### INT8 (8-bit Integer)
- **Size:** Smallest (~4 KB for classifiers)
- **Speed:** Fast on CPU
- **Accuracy:** Slight reduction
- **Use case:** CPU-optimized deployment

### Edge TPU
- **Size:** Similar to INT8
- **Speed:** Fastest (requires Coral TPU)
- **Accuracy:** Similar to INT8
- **Use case:** Maximum performance with Coral USB Accelerator

## 🤚 Gesture Classes

### Keypoint Classifier (Static Poses)
- **Open**: Open hand
- **Close**: Closed fist
- **Pointer**: Index finger pointing

### Point History Classifier (Dynamic Gestures)
Configured in `point_history_classifier_label.csv` (check file for available gestures)

## ⚡ Performance Considerations

### Raspberry Pi 4 Optimization
- Use Edge TPU models with Google Coral for best performance
- INT8 models provide good CPU performance
- Run `prepare_benchmark.sh` before deployment for optimal system settings
- Monitor temperature to avoid thermal throttling

### Model Selection Guidelines
- **Development:** FP32 for maximum accuracy
- **CPU deployment:** INT8 for best CPU performance
- **Coral TPU:** Edge TPU models for hardware acceleration
- **Balanced:** FP16 for moderate file size and good accuracy

### Temperature Management
The benchmark system monitors temperature on Raspberry Pi:
- Optimal range: < 60°C
- Warning threshold: 60-70°C
- Throttling risk: > 70°C

## 📝 Notes

- The application supports multi-hand tracking with individual gesture recognition
- Gesture history smoothing is applied for stable predictions
- FPS is displayed in real-time on the video feed
- Benchmark results include averaged metrics from multiple runs

## 🙏 Acknowledgments

This project is built upon MediaPipe hand tracking and uses TensorFlow Lite for efficient edge deployment.

## 📄 License

Check repository for license information.
