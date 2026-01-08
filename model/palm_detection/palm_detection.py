import copy
from typing import (
    Tuple,
    Optional,
    List,
)
from math import (
    sin,
    cos,
    atan2,
    pi,
)
import numpy as np

from utils.utils import (
    normalize_radians,
    keep_aspect_resize_and_pad,
)


class PalmDetection(object):
    def __init__(
        self,
        model_path: Optional[str] = 'model/palm_detection/palm_detection_full_inf_post_192x192.onnx',
        score_threshold: Optional[float] = 0.60,
        num_threads: Optional[int] = 1,
        providers: Optional[List] = [
            # (
            #     'TensorrtExecutionProvider', {
            #         'trt_engine_cache_enable': True,
            #         'trt_engine_cache_path': '.',
            #         'trt_fp16_enable': True,
            #     }
            # ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """PalmDetection

        Parameters
        ----------
        model_path: Optional[str]
            ONNX or TFLite file path for Palm Detection

        score_threshold: Optional[float]
            Detection score threshold. Default: 0.60

        num_threads: Optional[int]
            Number of threads for TFLite inference. Default: 1

        providers: Optional[List]
            Name of onnx execution providers (only used for ONNX models)
            Default:
            [
                (
                    'TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': '.',
                        'trt_fp16_enable': True,
                    }
                ),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        """
        # Threshold
        self.score_threshold = score_threshold
        self.model_path = model_path
        self.use_onnx = model_path.endswith('.onnx')

        if self.use_onnx:
            # ONNX Model loading
            import onnxruntime
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self.onnx_session = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_option,
                providers=providers,
            )
            self.providers = self.onnx_session.get_providers()

            self.input_shapes = [
                input.shape for input in self.onnx_session.get_inputs()
            ]
            self.input_names = [
                input.name for input in self.onnx_session.get_inputs()
            ]
            self.output_names = [
                output.name for output in self.onnx_session.get_outputs()
            ]
        else:
            # TFLite Model loading
            import tensorflow.lite as tflite
            
            # Check if this is an Edge TPU model
            is_edgetpu_model = '_edgetpu.tflite' in model_path
            delegates = []
            
            if is_edgetpu_model:
                # Try to load Edge TPU delegate
                try:
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
            
            # Get input shapes from TFLite
            self.input_shapes = [detail['shape'] for detail in self.input_details]
            
        self.square_standard_size = 0


    def __call__(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """PalmDetection

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        hands: np.ndarray
            float32[N, 4]
            sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        inference_image = self.__preprocess(
            temp_image,
        )

        # Inference
        if self.use_onnx:
            # ONNX Inference
            inferece_image = np.asarray([inference_image], dtype=np.float32)
            boxes = self.onnx_session.run(
                self.output_names,
                {input_name: inferece_image for input_name in self.input_names},
            )
            boxes = boxes[0]
        else:
            # TFLite Inference
            input_details_tensor_index = self.input_details[0]['index']
            input_dtype = self.input_details[0]['dtype']
            
            # Prepare input data (TFLite expects NHWC format)
            inferece_image = np.asarray([inference_image], dtype=np.float32)
            
            # Handle quantized models (INT8/UINT8)
            if input_dtype == np.uint8 or input_dtype == np.int8:
                # Get quantization parameters
                input_scale, input_zero_point = self.input_details[0]['quantization']
                # Quantize input data
                inferece_image = (inferece_image / input_scale + input_zero_point).astype(input_dtype)
            
            self.interpreter.set_tensor(input_details_tensor_index, inferece_image)
            self.interpreter.invoke()

            # TFLite palm_detection_full.tflite has 2 outputs: regressors and classificators
            # We need to process them similar to the ONNX post-processed output
            regressors_index = self.output_details[0]['index']
            classificators_index = self.output_details[1]['index']
            
            regressors = self.interpreter.get_tensor(regressors_index)
            classificators = self.interpreter.get_tensor(classificators_index)
            
            # Process raw outputs into the format expected by postprocess
            boxes = self.__process_tflite_outputs(regressors, classificators)

        # PostProcess
        hands = self.__postprocess(
            image=temp_image,
            boxes=boxes,
        )

        return hands


    def __process_tflite_outputs(
        self,
        regressors: np.ndarray,
        classificators: np.ndarray,
    ) -> np.ndarray:
        """Process raw TFLite outputs into postprocess-compatible format
        
        Parameters
        ----------
        regressors: np.ndarray
            [1, 2016, 18] - bounding box and keypoint regressors
        classificators: np.ndarray
            [1, 2016, 1] - palm detection scores
            
        Returns
        -------
        boxes: np.ndarray
            [N, 8] - pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y
        """
        # Remove batch dimension
        regressors = regressors[0]  # [2016, 18]
        scores = classificators[0]  # [2016, 1]
        
        # Apply sigmoid to scores
        scores = 1.0 / (1.0 + np.exp(-scores))
        scores = scores.squeeze(-1)  # [2016]
        
        # Filter by score threshold
        keep_indices = scores > self.score_threshold
        scores = scores[keep_indices]
        regressors = regressors[keep_indices]
        
        if len(scores) == 0:
            return np.array([]).reshape(0, 8)
        
        # Extract box coordinates and keypoints from regressors
        # MediaPipe palm detection output format:
        # regressors: [box_y, box_x, box_h, box_w, ...18 values total including keypoints]
        # We need: [pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y]
        
        box_y = regressors[:, 0]
        box_x = regressors[:, 1]
        box_h = regressors[:, 2]
        box_w = regressors[:, 3]
        
        # Keypoints (palm detection uses 7 keypoints, we need keypoint 0 and 2)
        # Keypoints start at index 4, each has x,y coordinates
        kp0_x = regressors[:, 4]
        kp0_y = regressors[:, 5]
        kp2_x = regressors[:, 8]  # keypoint 2 x (skip keypoint 1)
        kp2_y = regressors[:, 9]  # keypoint 2 y
        
        # Calculate box_size as average of width and height
        box_size = (box_w + box_h) / 2.0
        
        # Stack into expected format
        boxes = np.stack([
            scores,
            box_x,
            box_y,
            box_size,
            kp0_x,
            kp0_y,
            kp2_x,
            kp2_y
        ], axis=1)
        
        # Apply simple NMS to remove overlapping detections
        boxes = self.__nms(boxes, iou_threshold=0.3)
        
        return boxes


    def __nms(
        self,
        boxes: np.ndarray,
        iou_threshold: float = 0.3,
    ) -> np.ndarray:
        """Simple Non-Maximum Suppression
        
        Parameters
        ----------
        boxes: np.ndarray
            [N, 8] - pd_score, box_x, box_y, box_size, ...
        iou_threshold: float
            IoU threshold for NMS
            
        Returns
        -------
        boxes: np.ndarray
            Filtered boxes after NMS
        """
        if len(boxes) == 0:
            return boxes
            
        # Sort by score (descending)
        scores = boxes[:, 0]
        sorted_indices = np.argsort(-scores)
        boxes = boxes[sorted_indices]
        
        keep = []
        while len(boxes) > 0:
            # Keep the box with highest score
            keep.append(boxes[0])
            if len(boxes) == 1:
                break
                
            # Calculate IoU with remaining boxes
            box1 = boxes[0]
            remaining = boxes[1:]
            
            # Simple distance-based filtering (instead of full IoU)
            # Calculate center distance
            cx1, cy1 = box1[1], box1[2]
            cx_remaining, cy_remaining = remaining[:, 1], remaining[:, 2]
            
            distances = np.sqrt((cx1 - cx_remaining)**2 + (cy1 - cy_remaining)**2)
            size_threshold = box1[3] * iou_threshold * 2  # Use box size for threshold
            
            # Keep boxes that are far enough
            boxes = remaining[distances > size_threshold]
        
        if len(keep) > 0:
            return np.array(keep)
        return np.array([]).reshape(0, 8)


    def __preprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        """__preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        padded_image: np.ndarray
            Resized and Padding and normalized image.
        """
        # Resize + Padding + Normalization + BGR->RGB
        # Handle different input formats: ONNX uses NCHW, TFLite uses NHWC
        if self.use_onnx:
            input_h = self.input_shapes[0][2]
            input_w = self.input_shapes[0][3]
        else:
            input_h = self.input_shapes[0][1]
            input_w = self.input_shapes[0][2]
        image_height , image_width = image.shape[:2]

        self.square_standard_size = max(image_height, image_width)
        self.square_padding_half_size = abs(image_height - image_width) // 2

        padded_image, resized_image = keep_aspect_resize_and_pad(
            image=image,
            resize_width=input_w,
            resize_height=input_h,
        )

        pad_size_half_h = max(0, (input_h - resized_image.shape[0]) // 2)
        pad_size_half_w = max(0, (input_w - resized_image.shape[1]) // 2)

        self.pad_size_scale_h = pad_size_half_h / input_h
        self.pad_size_scale_w = pad_size_half_w / input_w

        padded_image = np.divide(padded_image, 255.0)
        padded_image = padded_image[..., ::-1]
        
        # Only transpose for ONNX (NCHW), TFLite keeps NHWC format
        if self.use_onnx:
            padded_image = padded_image.transpose(swap)
        
        padded_image = np.ascontiguousarray(
            padded_image,
            dtype=np.float32,
        )
        return padded_image


    def __postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:
        """__postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 8]
            pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y

        Returns
        -------
        hands: np.ndarray
            float32[N, 4]
            sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        hands = []
        keep = boxes[:, 0] > self.score_threshold # pd_score > self.score_threshold
        boxes = boxes[keep, :]

        for box in boxes:
            pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = box
            if box_size > 0:
                kp02_x = kp2_x - kp0_x
                kp02_y = kp2_y - kp0_y
                sqn_rr_size = 2.9 * box_size
                rotation = 0.5 * pi - atan2(-kp02_y, kp02_x)
                rotation = normalize_radians(rotation)
                sqn_rr_center_x = box_x + 0.5*box_size*sin(rotation)
                sqn_rr_center_y = box_y - 0.5*box_size*cos(rotation)
                sqn_rr_center_y = (sqn_rr_center_y * self.square_standard_size - self.square_padding_half_size) / image_height
                hands.append(
                    [
                        sqn_rr_size,
                        rotation,
                        sqn_rr_center_x,
                        sqn_rr_center_y,
                    ]
                )

        return np.asarray(hands)