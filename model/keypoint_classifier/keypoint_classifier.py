#!/usr/bin/env python

import onnxruntime
import numpy as np
from typing import (
    Optional,
    List,
)
import tensorflow.lite as tflite

import numpy as np

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.model_path = model_path
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
            self.input_name = self.onnx_session.get_inputs()[0].name
            self.output_name = self.onnx_session.get_outputs()[0].name
        else:
            # Check if this is an Edge TPU model
            is_edgetpu_model = '_edgetpu.tflite' in model_path
            delegates = []
            
            if is_edgetpu_model:
                edgetpu_delegate = None
                last_error = None
                
                for lib_name in ['libedgetpu.so.1', 'libedgetpu.so', 'libedgetpu.1.dylib']:
                    try:
                        edgetpu_delegate = tflite.experimental.load_delegate(lib_name)
                        delegates.append(edgetpu_delegate)
                        print(f"✓ Edge TPU delegate loaded: {lib_name}")
                        break
                    except Exception as e:
                        last_error = e
                        continue
                
                if not edgetpu_delegate:
                    error_msg = f"Edge TPU model specified but delegate could not be loaded. Last error: {last_error}\n"
                    error_msg += "Make sure:\n"
                    error_msg += "1. Edge TPU is plugged in (check with 'lsusb | grep Google')\n"
                    error_msg += "2. Edge TPU runtime is installed (https://coral.ai/docs/accelerator/get-started/)\n"
                    error_msg += "3. User has permissions to access the device\n"
                    error_msg += "Or use a non-Edge TPU model (int8, fp16, fp32)"
                    raise RuntimeError(error_msg)
            
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
        landmark_list,
    ):
        """
        Returns:
            (class_id, confidence)
            If ONNX is used, confidence will be None.
        """
        input_data = np.array([landmark_list], dtype=np.float32)

        if self.use_onnx:
            # ONNX Inference
            result = self.onnx_session.run(
                [self.output_name],
                {self.input_name: input_data},
            )[0]
            
            # Handle the specific ONNX output (often just an ID)
            result = np.squeeze(result)
            if result.size == 1 and (result.dtype == np.int64 or result.dtype == np.int32):
                return int(result), None
            else:
                # If logits are returned, we pick max but return None for confidence per instructions
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