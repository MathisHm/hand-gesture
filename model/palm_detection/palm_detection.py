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
        model_path: Optional[str] = 'model/palm_detection/palm_detection_fp32.tflite',
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

            # Process raw outputs into the format expected by postprocess
            boxes = self.__process_tflite_outputs()

        # PostProcess
        hands = self.__postprocess(
            image=temp_image,
            boxes=boxes,
        )

        return hands

    def __process_tflite_outputs(
        self,
    ) -> np.ndarray:
        """Process raw TFLite outputs into postprocess-compatible format"""
        
        # Identify outputs by shape
        regressors_index = -1
        classificators_index = -1
        for detail in self.output_details:
            if detail['shape'][-1] == 18:
                regressors_index = detail['index']
            elif detail['shape'][-1] == 1:
                classificators_index = detail['index']
        
        regressors = self.interpreter.get_tensor(regressors_index)
        classificators = self.interpreter.get_tensor(classificators_index)
        
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

        # Generate anchors if not already done
        if not hasattr(self, 'anchors'):
             self.anchors = self._generate_anchors(input_size_w=192, input_size_h=192)
        
        anchors = self.anchors[keep_indices]

        # Decode Logic
        # regressors: [x, y, w, h, kp0_x, kp0_y, kp1_x, kp1_y ... kp6_x, kp6_y] (18 values)
        # However, MediaPipe standard might be [y, x, h, w]?
        # Let's check based on 128x128 behavior or experiment.
        # Assuming [x, y, w, h] for now.
        
        # 192x192 model usually follows standard SSD box decoding
        # box_coordinate = anchor_coordinate + model_output * anchor_scale?
        # Or box_coordinate = model_output (if model output is absolute?)
        # Since standard TFLite models output OFFSETS, we decode.
        
        # Constant scale factors (standard MediaPipe constants)
        target_box_width = 192
        target_box_height = 192
        
        # Decode box
        # Output is usually normalized by 192.
        # dx, dy, dw, dh = regressors[:, 0], regressors[:, 1], regressors[:, 2], regressors[:, 3]
        # cx = dx + anchor_x_center
        # cy = dy + anchor_y_center
        # w = dw + anchor_w? No?
        
        # Using standard Mediapie Palm Detection decoding constants:
        # x_scale = y_scale = 192
        # w_scale = h_scale = 192
        
        box_x_center = regressors[:, 0] / 192.0 + anchors[:, 0]
        box_y_center = regressors[:, 1] / 192.0 + anchors[:, 1]
        box_w = regressors[:, 2] / 192.0
        box_h = regressors[:, 3] / 192.0
        
        # Keypoints (7 keypoints * 2 coords = 14 values)
        # Starts at index 4
        # kp_x = reg_kp_x / 192.0 + anchor_x
        # kp_y = reg_kp_y / 192.0 + anchor_y
        
        kp0_x = regressors[:, 4] / 192.0 + anchors[:, 0]
        kp0_y = regressors[:, 5] / 192.0 + anchors[:, 1]
        
        # Keypoint 2 (Index 2 -> 4 + 2*2 = 8)
        kp2_x = regressors[:, 8] / 192.0 + anchors[:, 0]
        kp2_y = regressors[:, 9] / 192.0 + anchors[:, 1]

        # Convert simple [cx, cy, w, h] to [x, y, size] format expected by post-process
        # Code expects: box_x (center), box_y (center), box_size (max(w,h)?)
        
        # The existing postprocess expects:
        # pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y
        
        box_size = np.maximum(box_w, box_h)
        
        boxes = np.stack([
            scores,
            box_x_center,
            box_y_center,
            box_size,
            kp0_x,
            kp0_y,
            kp2_x,
            kp2_y
        ], axis=1)
        
        boxes = self.__nms(boxes, iou_threshold=0.3)
        return boxes

    def _generate_anchors(self, input_size_w=192, input_size_h=192):
        # Anchor config (Deduced for 2016 anchors)
        # Layer 0: Stride 8, 2 anchors (24x24) -> 1152
        # Layer 1: Stride 16, 6 anchors (12x12) -> 864
        # Total: 2016
        
        anchors = []
        
        # Layer 0
        stride = 8
        rows, cols = input_size_h // stride, input_size_w // stride
        num_anchors = 2
        for y in range(rows):
            for x in range(cols):
                x_center = (x + 0.5) * stride / input_size_w
                y_center = (y + 0.5) * stride / input_size_h
                for _ in range(num_anchors):
                    anchors.append([x_center, y_center])
                    
        # Layer 1
        stride = 16
        rows, cols = input_size_h // stride, input_size_w // stride
        num_anchors = 6
        for y in range(rows):
            for x in range(cols):
                x_center = (x + 0.5) * stride / input_size_w
                y_center = (y + 0.5) * stride / input_size_h
                for _ in range(num_anchors):
                    anchors.append([x_center, y_center])
                    
        return np.array(anchors)

    def __nms(
        self,
        boxes: np.ndarray,
        iou_threshold: float = 0.3,
    ) -> np.ndarray:
        """Simple Non-Maximum Suppression"""
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
            
            # Simple distance-based filtering
            cx1, cy1 = box1[1], box1[2]
            cx_remaining, cy_remaining = remaining[:, 1], remaining[:, 2]
            
            distances = np.sqrt((cx1 - cx_remaining)**2 + (cy1 - cy_remaining)**2)
            size_threshold = box1[3] * iou_threshold * 2
            
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