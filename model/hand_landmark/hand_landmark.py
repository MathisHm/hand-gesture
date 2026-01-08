import copy
from typing import (
    Tuple,
    Optional,
    List,
)
import cv2
import numpy as np

from utils.utils import keep_aspect_resize_and_pad


class HandLandmark(object):
    def __init__(
        self,
        model_path: Optional[str] = 'model/hand_landmark/hand_landmark_int8.tflite',
        class_score_th: Optional[float] = 0.50,
        num_threads: Optional[int] = 1,
        providers: Optional[List] = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self.class_score_th = class_score_th
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
                except Exception as e:
                    print(f"⚠ Could not load Edge TPU delegate: {e}")
            
            self.interpreter = tflite.Interpreter(
                model_path=model_path,
                num_threads=num_threads,
                experimental_delegates=delegates if delegates else None
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.input_shapes = [detail['shape'] for detail in self.input_details]
            
            # Identify output indices by name
            self.output_indices = {}
            for detail in self.output_details:
                name = detail['name']
                # Clean name (remove :0 suffix if present)
                clean_name = name.split(':')[0]
                
                if clean_name == 'Identity':
                    self.output_indices['landmarks'] = detail['index']
                elif clean_name == 'Identity_1':
                    self.output_indices['score'] = detail['index']
                elif clean_name == 'Identity_2':
                    self.output_indices['handedness'] = detail['index']


    def __call__(
        self,
        images: List[np.ndarray],
        rects: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:

        temp_images = copy.deepcopy(images)

        inference_images, resized_images, resize_scales_224x224, half_pad_sizes_224x224 = self.__preprocess(
            images=temp_images,
        )

        if self.use_onnx:
            xyz_x21s, hand_scores, left_hand_0_or_right_hand_1s = self.onnx_session.run(
                self.output_names,
                {input_name: inference_images for input_name in self.input_names},
            )
        else:
            # TFLite Inference
            # TFLite models usually have batch size 1, so we process images one by one
            xyz_x21s = []
            hand_scores = []
            left_hand_0_or_right_hand_1s = []

            input_details_tensor_index = self.input_details[0]['index']
            input_dtype = self.input_details[0]['dtype']
            
            for i, inference_image in enumerate(inference_images):
                 # Prepare input data (Add batch dimension [1, 224, 224, 3])
                inference_image = inference_image[np.newaxis, ...]
                
                # Handle quantized models
                if input_dtype == np.uint8 or input_dtype == np.int8:
                    input_scale, input_zero_point = self.input_details[0]['quantization']
                    inference_image = (inference_image / input_scale + input_zero_point).astype(input_dtype)
                
                self.interpreter.set_tensor(input_details_tensor_index, inference_image)
                self.interpreter.invoke()

                # Get outputs using identified indices
                output_landmarks = self.interpreter.get_tensor(self.output_indices['landmarks'])
                output_score = self.interpreter.get_tensor(self.output_indices['score'])
                output_handedness = self.interpreter.get_tensor(self.output_indices['handedness'])
                
                xyz_x21s.append(output_landmarks[0])
                hand_scores.append(output_score[0])
                left_hand_0_or_right_hand_1s.append(output_handedness[0])
            
            xyz_x21s = np.array(xyz_x21s)
            hand_scores = np.array(hand_scores)
            left_hand_0_or_right_hand_1s = np.array(left_hand_0_or_right_hand_1s)


        hand_landmarks, rotated_image_size_leftrights = self.__postprocess(
            resized_images=resized_images,
            resize_scales_224x224=resize_scales_224x224,
            half_pad_sizes_224x224=half_pad_sizes_224x224,
            rects=rects,
            xyz_x21s=xyz_x21s,
            hand_scores=hand_scores,
            left_hand_0_or_right_hand_1s=left_hand_0_or_right_hand_1s,
        )

        return hand_landmarks, rotated_image_size_leftrights


    def __preprocess(
        self,
        images: List[np.ndarray],
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        temp_images = copy.deepcopy(images)

        # Get input size from model
        if self.use_onnx:
            input_h = self.input_shapes[0][2]
            input_w = self.input_shapes[0][3]
        else:
            input_h = self.input_shapes[0][1]
            input_w = self.input_shapes[0][2]

        padded_images = []
        resized_images = []
        resize_scales_224x224 = []
        half_pad_sizes_224x224 = []
        
        for image in temp_images:
            padded_image, resized_image = keep_aspect_resize_and_pad(
                image=image,
                resize_width=input_w,
                resize_height=input_h,
            )

            resize_224x224_scale_h = resized_image.shape[0] / image.shape[0]
            resize_224x224_scale_w = resized_image.shape[1] / image.shape[1]
            resize_scales_224x224.append(
                [resize_224x224_scale_w, resize_224x224_scale_h]
            )

            pad_h = padded_image.shape[0] - resized_image.shape[0]
            pad_w = padded_image.shape[1] - resized_image.shape[1]
            half_pad_h_224x224 = pad_h // 2 if pad_h >= 0 else 0
            half_pad_w_224x224 = pad_w // 2 if pad_w >= 0 else 0
            half_pad_sizes_224x224.append([half_pad_w_224x224, half_pad_h_224x224])

            padded_image = np.divide(padded_image, 255.0)
            padded_image = padded_image[..., ::-1] # BGR to RGB
            
            # ONNX expects NCHW, TFLite expects NHWC
            if self.use_onnx:
                padded_image = padded_image.transpose(swap)
            
            padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)

            padded_images.append(padded_image)
            resized_images.append(resized_image)

        return (
            np.asarray(padded_images, dtype=np.float32),
            resized_images,
            np.asarray(resize_scales_224x224, dtype=np.float32),
            np.asarray(half_pad_sizes_224x224, dtype=np.int32),
        )


    def __postprocess(
        self,
        resized_images: List[np.ndarray],
        resize_scales_224x224: np.ndarray,
        half_pad_sizes_224x224: np.ndarray,
        rects: np.ndarray,
        xyz_x21s: np.ndarray,
        hand_scores: np.ndarray,
        left_hand_0_or_right_hand_1s: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:

        extracted_hands = []
        rotated_image_size_leftrights = []

        keep = hand_scores[:, 0] > self.class_score_th
        xyz_x21s = xyz_x21s[keep, :]
        hand_scores = hand_scores[keep, :]
        left_hand_0_or_right_hand_1s = left_hand_0_or_right_hand_1s[keep, :]
        
        # Filter resized_images based on keep indices
        # Since resized_images is a list, we need to compress it
        resized_images = [img for img, k in zip(resized_images, keep) if k]
        
        # We need to filter the corresponding input metadata
        resize_scales_224x224 = resize_scales_224x224[keep]
        half_pad_sizes_224x224 = half_pad_sizes_224x224[keep]
        rects = rects[keep]

        for resized_image, resize_scale_224x224, half_pad_size_224x224, rect, xyz_x21, left_hand_0_or_right_hand_1 in \
            zip(resized_images, resize_scales_224x224, half_pad_sizes_224x224, rects, xyz_x21s, left_hand_0_or_right_hand_1s):

            rrn_lms = xyz_x21
            
            # Input sizes for normalization
            if self.use_onnx:
                input_h = self.input_shapes[0][2]
                input_w = self.input_shapes[0][3]
            else:
                input_h = self.input_shapes[0][1]
                input_w = self.input_shapes[0][2]
                
            rrn_lms = rrn_lms / input_h

            rcx = rect[0]
            rcy = rect[1]
            angle = rect[4]

            view_image = copy.deepcopy(resized_image)
            view_image = cv2.resize(
                view_image,
                dsize=None,
                fx=1/resize_scale_224x224[0],
                fy=1/resize_scale_224x224[1],
            )

            rescaled_xy = np.asarray([[v[0], v[1]] for v in zip(rrn_lms[0::3], rrn_lms[1::3])], dtype=np.float32)
            rescaled_xy[:, 0] = (rescaled_xy[:, 0] * input_w - half_pad_size_224x224[0]) / resize_scale_224x224[0]
            rescaled_xy[:, 1] = (rescaled_xy[:, 1] * input_h - half_pad_size_224x224[1]) / resize_scale_224x224[1]
            rescaled_xy = rescaled_xy.astype(np.int32)

            height, width = view_image.shape[:2]
            image_center = (width//2, height//2)
            rotation_matrix = cv2.getRotationMatrix2D(image_center, -int(angle), 1)
            abs_cos = abs(rotation_matrix[0,0])
            abs_sin = abs(rotation_matrix[0,1])
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)
            rotation_matrix[0, 2] += bound_w/2 - image_center[0]
            rotation_matrix[1, 2] += bound_h/2 - image_center[1]
            rotated_image = cv2.warpAffine(view_image, rotation_matrix, (bound_w, bound_h))

            keypoints = []
            for x,y in rescaled_xy:
                coord_arr = np.array([[x, y, 1]])
                new_coord = rotation_matrix.dot(coord_arr.T)
                keypoints.append([int(new_coord[0]), int(new_coord[1])])

            rotated_image_width = rotated_image.shape[1]
            rotated_image_height = rotated_image.shape[0]
            roatated_hand_half_width = rotated_image_width // 2
            roatated_hand_half_height = rotated_image_height // 2

            hand_landmarks = np.asarray(keypoints, dtype=np.int32).reshape(-1,2)
            hand_landmarks[..., 0] = hand_landmarks[..., 0] + rcx - roatated_hand_half_width
            hand_landmarks[..., 1] = hand_landmarks[..., 1] + rcy - roatated_hand_half_height

            extracted_hands.append(hand_landmarks)
            rotated_image_size_leftrights.append([
            rotated_image_width,
            rotated_image_height,
            int(left_hand_0_or_right_hand_1)
            ])


        return extracted_hands, np.asarray(rotated_image_size_leftrights, dtype=np.int32)
