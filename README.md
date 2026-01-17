# Hand Gesture Recognition for Edge Devices

A real-time hand gesture recognition system optimized for edge computing devices (Raspberry Pi 4 and Google Coral TPU). This project provides a complete pipeline for deployment with multiple quantization strategies for performance optimization.

## 🎯 Features

- **Real-time hand gesture detection** using MediaPipe-based models
- **Multiple quantization formats** (FP32, FP16, INT8, Edge TPU)
- **Optimized for edge devices** (Raspberry Pi 4, Google Coral TPU)
- **Dual classification system**:
  - Static hand pose recognition (keypoint classifier)
  - Dynamic gesture tracking (point history classifier)
- **Comprehensive benchmarking suite** for performance evaluation

## 💻 System Requirements

- Raspberry Pi 4
- Python 3.9
- Camera module or USB webcam
- Optional: Google Coral USB Accelerator

## 📦 Installation

Install all required dependencies using the provided command file:

```bash
# See cmd2.txt for the complete installation command
python3.9 -m pip install tflite-runtime==2.9.1 "numpy<2" "opencv-contrib-python>=4.6.0.66" "scikit-learn>=1.1.0" "matplotlib>=3.5.0" psutil
```

## 📂 Project Structure

```
hand-gesture/
├── app.py                          # Main application for real-time gesture recognition
├── benchmark.py                    # Automated benchmarking script
├── cmd.txt                         # Installation commands
├── prepare_benchmark.sh            # System preparation script for benchmarking
├── run_benchmark_suite.sh          # Automated benchmark suite runner
├── restore_system.sh               # System restoration script
├── model/                          # Pre-trained models (FP32, FP16, INT8, Edge TPU)
└── utils/                          # Utility scripts
```

## 🚀 Usage

### Running the Application

Configure model types at the top of `app.py`:

```python
KEYPOINT_MODEL_TYPE = "fp32"         # Options: 'fp32', 'fp16', 'int8', 'edgetpu'
POINT_HISTORY_MODEL_TYPE = "fp32"   # Options: 'fp32', 'fp16', 'int8', 'edgetpu'
HAND_LANDMARK_MODEL_TYPE = "fp32"   # Options: 'fp32', 'fp16', 'int8', 'edgetpu'
PALM_DETECTION_MODEL_TYPE = "fp32"  # Options: 'fp32', 'fp16'
```

Run the application:
```bash
python3 app.py
```

## 📊 Benchmarking

### Quick Benchmark

```bash
python3 benchmark.py
```

### Full Automated Benchmark Suite (Raspberry Pi)

```bash
sudo ./run_benchmark_suite.sh
```

**Benchmark Metrics:**
- Average cycle time
- Preprocessing time
- Inference time
- Memory usage
- Model confidence scores
- Temperature readings (Raspberry Pi)

## 🔧 Model Variants

### FP32 (Full Precision)
- **Size:** Largest
- **Speed:** Slowest
- **Accuracy:** Highest

### FP16 (Half Precision)
- **Size:** ~50% of FP32
- **Speed:** Moderate
- **Accuracy:** Near FP32

### INT8 (8-bit Integer)
- **Size:** Smallest
- **Speed:** Fast on CPU
- **Accuracy:** Slight reduction

### Edge TPU
- **Size:** Similar to INT8
- **Speed:** Fastest (requires Coral TPU)
- **Accuracy:** Similar to INT8

## 🤚 Gesture Classes

### Keypoint Classifier (Static Poses)
- **Open**: Open hand
- **Close**: Closed fist
- **Pointer**: Index finger pointing

### Point History Classifier (Dynamic Gestures)
- **Stop**: No movement
- **Clockwise**: Circular motion clockwise
- **Counter Clockwise**: Circular motion counter-clockwise
- **Move**: Linear movement

## 🙏 References

This project is based on the original implementation:
- [Hand Gesture Recognition using MediaPipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) by Kazuhito00

