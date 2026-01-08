import csv
import time
import copy
import itertools
import subprocess
import os
import psutil
from collections import deque
from math import degrees

import cv2 as cv
import numpy as np

from utils.utils import rotate_and_crop_rectangle
from model import KeyPointClassifier
from model import PointHistoryClassifier
from model import PalmDetection
from model import HandLandmark


# Keypoint Classifier Model Selection
KEYPOINT_MODEL_TYPE = "fp32"  # Options: 'fp32', 'fp16', 'int8', 'edgetpu'

# Point History Classifier Model Selection
POINT_HISTORY_MODEL_TYPE = "fp32"  # Options: 'fp32', 'fp16', 'int8', 'edgetpu'

# Hand Landmark Model Selection
HAND_LANDMARK_MODEL_TYPE = "fp32" # Options: 'fp32', 'fp16', 'int8', 'edgetpu'

# Model path builder
# Model path builder
def get_model_path(base_path, model_type):
    """Build model path based on selected type."""
    return f"{base_path}_{model_type}.tflite"

# Build model paths
KEYPOINT_MODEL_PATH = get_model_path(
    'model/keypoint_classifier/keypoint_classifier',
    KEYPOINT_MODEL_TYPE
)
POINT_HISTORY_MODEL_PATH = get_model_path(
    'model/point_history_classifier/point_history_classifier',
    POINT_HISTORY_MODEL_TYPE
)
 

def get_memory_usage():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Returns MB
    except:
        return -1

def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    if isinstance(temp, np.ndarray):
        temp = temp.tolist()
    bx, by = temp[0]
    for i in range(len(temp)):
        temp[i][0] -= bx
        temp[i][1] -= by
    temp = list(itertools.chain.from_iterable(temp))
    mv = max(map(abs, temp))
    if mv == 0:
        return [0] * len(temp)
    return [v / mv for v in temp]

def pre_process_point_history(image, point_history):
    w, h = image.shape[1], image.shape[0]
    temp = copy.deepcopy(point_history)
    if len(temp) == 0:
        return []
    bx, by = temp[0]
    for i in range(len(temp)):
        temp[i][0] = (temp[i][0] - bx) / w
        temp[i][1] = (temp[i][1] - by) / h
    return list(itertools.chain.from_iterable(temp))

def run_benchmark(video_path, csv_path="benchmark_results.csv"):
    
    cap = cv.VideoCapture(video_path)

    palm_detection = PalmDetection(score_threshold=0.6)
    
    # Resolve Hand Landmark Model Path
    hand_landmark_model_path = get_model_path(
         'model/hand_landmark/hand_landmark',
         HAND_LANDMARK_MODEL_TYPE
    )

    hand_landmark = HandLandmark(model_path=hand_landmark_model_path)
    
    kp_classifier = KeyPointClassifier(model_path=KEYPOINT_MODEL_PATH)
    ph_classifier = PointHistoryClassifier(model_path=POINT_HISTORY_MODEL_PATH, score_th=0.5)

    hist_len = 16
    point_history = deque(maxlen=hist_len)

    cycle_times = []
    preprocess_times = []
    inference_times = []
    memory_usages = []
    
    kp_confidences = [] 
    ph_confidences = []
    combined_confidences = []

    cap_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    if cap_width == 0 or cap_height == 0:
        print("Error: Video not found or size is 0.")
        return

    wh_ratio = cap_width / cap_height

    while True:
        start_cycle_time = time.time()
        
        current_preprocess_time = 0.0
        current_inference_time = 0.0
        
        cur_kp_conf = None
        cur_ph_conf = None
        cur_combined_conf = None

        ret, frame = cap.read()
        if not ret:
            break
        
        image = frame
        debug_image = copy.deepcopy(image)

        t_infer_start = time.time()
        hands = palm_detection(image)
        current_inference_time += (time.time() - t_infer_start)
        
        rects = []
        if len(hands) > 0:
            t_prep_start = time.time()
            for hand in hands:
                sqn_rr_size = hand[0]
                rotation = hand[1]
                sqn_rr_center_x = hand[2]
                sqn_rr_center_y = hand[3]
                cx = int(sqn_rr_center_x * cap_width)
                cy = int(sqn_rr_center_y * cap_height)
                xmin = int((sqn_rr_center_x - (sqn_rr_size / 2)) * cap_width)
                xmax = int((sqn_rr_center_x + (sqn_rr_size / 2)) * cap_width)
                ymin = int((sqn_rr_center_y - (sqn_rr_size * wh_ratio / 2)) * cap_height)
                ymax = int((sqn_rr_center_y + (sqn_rr_size * wh_ratio / 2)) * cap_height)
                xmin, xmax = max(0, xmin), min(cap_width, xmax)
                ymin, ymax = max(0, ymin), min(cap_height, ymax)
                degree = degrees(rotation)
                rects.append([cx, cy, (xmax-xmin), (ymax-ymin), degree])
            
            rects = np.asarray(rects, dtype=np.float32)
            cropped_imgs = rotate_and_crop_rectangle(image, rects, 'padding')
            current_preprocess_time += (time.time() - t_prep_start)

            if len(cropped_imgs) > 0:
                t_infer_start = time.time()
                landmarks_list, _ = hand_landmark(images=cropped_imgs, rects=rects)
                current_inference_time += (time.time() - t_infer_start)
                
                if len(landmarks_list) > 0:
                    lm = landmarks_list[0]
                    lm_list = lm.tolist() 

                    t_prep_start = time.time()
                    lm_proc = pre_process_landmark(lm_list)
                    pt_proc = pre_process_point_history(debug_image, point_history)
                    current_preprocess_time += (time.time() - t_prep_start)
                    
                    t_infer_start = time.time()
                    kp_id, cur_kp_conf = kp_classifier(lm_proc)
                    current_inference_time += (time.time() - t_infer_start)
                    
                    if kp_id == 2:
                        point_history.append(lm_list[8])
                    else:
                        point_history.append([0, 0])

                    if len(pt_proc) == (16 * 2):
                        t_prep_start = time.time()
                        current_preprocess_time += (time.time() - t_prep_start)
                        
                        t_infer_start = time.time()
                        ph_id, cur_ph_conf = ph_classifier(pt_proc)
                        current_inference_time += (time.time() - t_infer_start)
                    else:
                        pass

 
                    valid_scores = []
                    if cur_kp_conf is not None: valid_scores.append(cur_kp_conf)
                    if cur_ph_conf is not None: valid_scores.append(cur_ph_conf)
                    
                    if len(valid_scores) > 0:
                        cur_combined_conf = sum(valid_scores) / len(valid_scores)

                else:
                    point_history.append([0, 0])
            else:
                point_history.append([0, 0])
        else:
            point_history.append([0, 0])

        preprocess_times.append(current_preprocess_time)
        inference_times.append(current_inference_time)
        kp_confidences.append(cur_kp_conf)
        ph_confidences.append(cur_ph_conf)
        combined_confidences.append(cur_combined_conf)
        memory_usages.append(get_memory_usage())
        cycle_times.append(time.time() - start_cycle_time)

    cap.release()

    def safe_avg(lst):
        valid_items = [x for x in lst if x is not None]
        return sum(valid_items) / len(valid_items) if valid_items else 0

    avg_cycle = safe_avg(cycle_times)
    avg_prep = safe_avg(preprocess_times)
    avg_infer = safe_avg(inference_times)
    avg_mem = safe_avg(memory_usages)
    
    avg_kp_conf = safe_avg(kp_confidences)
    avg_ph_conf = safe_avg(ph_confidences)
    avg_combined = safe_avg(combined_confidences)

    header = [
        "kp_model_type",
        "ph_model_type",
        "hl_model_type",
        "avg_cycle_time_s",
        "avg_preprocess_time_s",
        "avg_inference_time_s", 
        "avg_memory_usage_mb",
        "avg_keypoint_confidence",
        "avg_history_confidence",
        "avg_combined_confidence"
    ]

    write_header = False
    try:
        with open(csv_path, "r", newline='', encoding="utf-8") as f:
            pass
    except:
        write_header = True

    # Use the configured model types for CSV logging
    with open(csv_path, "a", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        row = [
            KEYPOINT_MODEL_TYPE,
            POINT_HISTORY_MODEL_TYPE,
            HAND_LANDMARK_MODEL_TYPE,
            avg_cycle,
            avg_prep,
            avg_infer,
            avg_mem,
            avg_kp_conf,
            avg_ph_conf,
            avg_combined
        ]
        w.writerow(row)
    
if __name__ == "__main__":
    run_benchmark("benchmark_video.mp4")