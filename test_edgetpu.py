#!/usr/bin/env python3
"""
Simple test script to verify Edge TPU functionality
"""
import numpy as np
import tensorflow.lite as tflite

def test_edgetpu_model(model_path):
    """Test loading and running an Edge TPU model"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print(f"{'='*60}")
    
    try:
        # Load delegate
        print("Step 1: Loading Edge TPU delegate...")
        edgetpu_delegate = tflite.experimental.load_delegate('libedgetpu.so.1')
        print("✓ Edge TPU delegate loaded successfully")
        
        # Create interpreter
        print("Step 2: Creating interpreter...")
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[edgetpu_delegate]
        )
        print("✓ Interpreter created successfully")
        
        # Allocate tensors
        print("Step 3: Allocating tensors...")
        interpreter.allocate_tensors()
        print("✓ Tensors allocated successfully")
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\nInput shape: {input_details[0]['shape']}")
        print(f"Input dtype: {input_details[0]['dtype']}")
        print(f"Number of outputs: {len(output_details)}")
        
        # Create dummy input
        print("\nStep 4: Running inference with dummy data...")
        input_shape = input_details[0]['shape']
        input_data = np.zeros(input_shape, dtype=input_details[0]['dtype'])
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        print("✓ Inference completed successfully")
        
        print(f"\n{'='*60}")
        print("✅ MODEL TEST PASSED")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ MODEL TEST FAILED")
        print(f"Error: {e}")
        print(f"{'='*60}\n")
        return False


if __name__ == "__main__":
    models_to_test = [
        "model/keypoint_classifier/keypoint_classifier_edgetpu.tflite",
        "model/point_history_classifier/point_history_classifier_edgetpu.tflite",
        "model/hand_landmark/hand_landmark_edgetpu.tflite",
    ]
    
    results = {}
    for model in models_to_test:
        results[model] = test_edgetpu_model(model)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {model}")
    print("="*60)
