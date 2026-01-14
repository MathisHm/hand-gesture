#!/usr/bin/env python3
"""Test only the hand_landmark Edge TPU model"""
import numpy as np
import tensorflow.lite as tflite

model_path = "model/hand_landmark/hand_landmark_edgetpu.tflite"

print(f"Testing: {model_path}")
print("="*60)

try:
    print("Step 1: Loading Edge TPU delegate...")
    edgetpu_delegate = tflite.experimental.load_delegate('libedgetpu.so.1')
    print("✓ Edge TPU delegate loaded")
    
    print("Step 2: Creating interpreter...")
    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[edgetpu_delegate]
    )
    print("✓ Interpreter created")
    
    print("Step 3: Allocating tensors...")
    interpreter.allocate_tensors()
    print("✓ Tensors allocated")
    
    input_details = interpreter.get_input_details()
    print(f"\nInput shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    
    print("\nStep 4: Running inference...")
    input_data = np.zeros(input_details[0]['shape'], dtype=input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    print("✓ Inference completed")
    
    print("\n" + "="*60)
    print("✅ HAND LANDMARK EDGE TPU MODEL WORKS!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
