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
            self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
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
            self.interpreter.set_tensor(input_details_tensor_index, input_data)
            self.interpreter.invoke()

            output_details_tensor_index = self.output_details[0]['index']
            result = self.interpreter.get_tensor(output_details_tensor_index)

            result = np.squeeze(result)
            result_index = np.argmax(result)
            confidence = float(np.max(result))

            return result_index, confidence