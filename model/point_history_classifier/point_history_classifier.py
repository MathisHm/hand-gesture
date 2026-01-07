#!/usr/bin/env python

import onnxruntime
import numpy as np
from typing import (
    Optional,
    List,
)
import tensorflow.lite as tflite

class PointHistoryClassifier(object):
    def __init__(
        self,
        model_path='model/point_history_classifier/point_history_classifier.tflite',
        score_th=0.5,
        num_threads=1,
    ):
        self.model_path = model_path
        self.score_th = score_th
        self.use_onnx = model_path.endswith('.onnx')

        if self.use_onnx:
            import onnxruntime
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self.onnx_session = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_option,
                providers=['CPUExecutionProvider'],
            )
            self.input_names = [input.name for input in self.onnx_session.get_inputs()]
            self.output_names = [output.name for output in self.onnx_session.get_outputs()]
        else:
            # Check if this is an Edge TPU model
            is_edgetpu_model = '_edgetpu.tflite' in model_path
            delegates = []
            
            if is_edgetpu_model:
                # Try to load Edge TPU delegate
                try:
                    # Try different possible Edge TPU library names
                    edgetpu_delegate = None
                    for lib_name in ['libedgetpu.so.1', 'libedgetpu.so', 'libedgetpu.1.dylib']:
                        try:
                            edgetpu_delegate = tflite.load_delegate(lib_name)
                            delegates.append(edgetpu_delegate)
                            print(f"✓ Edge TPU delegate loaded: {lib_name}")
                            break
                        except:
                            continue
                    
                    if not edgetpu_delegate:
                        print("⚠ Edge TPU delegate not found, falling back to CPU")
                        print("  Install Edge TPU runtime: https://coral.ai/docs/accelerator/get-started/")
                except Exception as e:
                    print(f"⚠ Could not load Edge TPU delegate: {e}")
                    print("  Running on CPU instead")
            
            # Create interpreter with or without Edge TPU delegate
            self.interpreter = tflite.Interpreter(
                model_path=model_path,
                num_threads=num_threads,
                experimental_delegates=delegates if delegates else None
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        point_history,
    ):
        # Ensure input is float32 and correct shape
        input_data = np.array([point_history], dtype=np.float32)

        if self.use_onnx:
            # ONNX Inference (Requires score_th as 2nd input)
            score_th_input = np.array(self.score_th, dtype=np.float32)
            
            result = self.onnx_session.run(
                self.output_names,
                {
                    self.input_names[0]: input_data,
                    self.input_names[1]: score_th_input,
                },
            )[0]
            
            # The ONNX model typically returns just the ID if it has ArgMax baked in
            result = np.squeeze(result)
            
            # Check if result looks like an ID (integer) or scores (floats)
            if result.size == 1 and (result.dtype == np.int64 or result.dtype == np.int32):
                return int(result), None
            else:
                return int(np.argmax(result)), None
                
        else:
            # TFLite Inference
            input_details_tensor_index = self.input_details[0]['index']
            input_dtype = self.input_details[0]['dtype']
            
            # Handle quantized models (INT8/UINT8)
            if input_dtype == np.uint8 or input_dtype == np.int8:
                # Get quantization parameters
                input_scale, input_zero_point = self.input_details[0]['quantization']
                # Quantize input data
                input_data = (input_data / input_scale + input_zero_point).astype(input_dtype)
            
            self.interpreter.set_tensor(input_details_tensor_index, input_data)
            self.interpreter.invoke()

            output_details_tensor_index = self.output_details[0]['index']
            result = self.interpreter.get_tensor(output_details_tensor_index)
            
            # Handle quantized output
            output_dtype = self.output_details[0]['dtype']
            if output_dtype == np.uint8 or output_dtype == np.int8:
                # Dequantize output
                output_scale, output_zero_point = self.output_details[0]['quantization']
                result = (result.astype(np.float32) - output_zero_point) * output_scale

            result = np.squeeze(result)
            result_index = np.argmax(result)
            confidence = float(np.max(result))

            return result_index, confidence