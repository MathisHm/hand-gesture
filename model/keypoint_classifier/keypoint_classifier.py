#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import onnxruntime as ort

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.model_path = model_path
        self.model_ext = os.path.splitext(model_path)[-1].lower()

        if self.model_ext == '.tflite':
            self.backend = 'tflite'
            self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                                   num_threads=num_threads)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        elif self.model_ext == '.onnx':
            self.backend = 'onnx'
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
        else:
            raise ValueError(f"Unsupported model type: {self.model_ext}")

    def __call__(self, landmark_list):
        if self.backend == 'tflite':
            input_details_tensor_index = self.input_details[0]['index']
            self.interpreter.set_tensor(
                input_details_tensor_index,
                np.array([landmark_list], dtype=np.float32))
            self.interpreter.invoke()
            output_details_tensor_index = self.output_details[0]['index']
            result = self.interpreter.get_tensor(output_details_tensor_index)
        elif self.backend == 'onnx':
            result = self.session.run(
                None,
                {self.input_name: np.array([landmark_list], dtype=np.float32)}
            )[0]

        result_index = np.argmax(np.squeeze(result))

        return result_index
