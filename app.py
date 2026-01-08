#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
from math import degrees

import cv2 as cv
import numpy as np

from utils import CvFpsCalc
from utils.utils import rotate_and_crop_rectangle
from model import PalmDetection
from model import HandLandmark
from model import KeyPointClassifier
from model import PointHistoryClassifier



# Keypoint Classifier Model Selection
KEYPOINT_MODEL_TYPE = "fp32"  # Options: 'fp32', 'fp16', 'int8', 'edgetpu'

# Point History Classifier Model Selection

# Point History Classifier Model Selection
POINT_HISTORY_MODEL_TYPE = "fp32"  # Options: 'fp32', 'fp16', 'int8', 'edgetpu'

# Hand Landmark Model Selection
HAND_LANDMARK_MODEL_TYPE = "fp32" # Options: 'fp32', 'fp16', 'int8', 'edgetpu'


# Model path builder
def get_model_path(base_path, model_type):
    """Build model path based on selected type."""
    return f"{base_path}_{model_type}.tflite"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min_detection_confidence", type=float, default=0.6)
    parser.add_argument("--disable_image_flip", action='store_true')

    return parser.parse_args()


def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    min_detection_confidence = args.min_detection_confidence

    lines_hand = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
    ]

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    palm_detection = PalmDetection(score_threshold=min_detection_confidence)
    palm_detection = PalmDetection(score_threshold=min_detection_confidence)
    
    # Hand Landmark
    hand_landmark_model_path = get_model_path(
         'model/hand_landmark/hand_landmark',
         HAND_LANDMARK_MODEL_TYPE
    )
        
    print(f"Loading Hand Landmark Model: {hand_landmark_model_path}")
    hand_landmark = HandLandmark(model_path=hand_landmark_model_path)


    # Load classifiers with selected model variants
    keypoint_model_path = get_model_path(
        'model/keypoint_classifier/keypoint_classifier',
        KEYPOINT_MODEL_TYPE
    )
    point_history_model_path = get_model_path(
        'model/point_history_classifier/point_history_classifier',
        POINT_HISTORY_MODEL_TYPE
    )
    
    print(f"Loading Keypoint Classifier: {keypoint_model_path}")
    print(f"Loading Point History Classifier: {point_history_model_path}")
    
    keypoint_classifier = KeyPointClassifier(model_path=keypoint_model_path)
    point_history_classifier = PointHistoryClassifier(model_path=point_history_model_path)

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    history_length = 16
    point_history = {}
    pre_point_history = {}

    gesture_history_length = 10
    finger_gesture_history = {}

    palm_trackid_cxcy = {}
    wh_ratio = cap_width / cap_height

    while True:
        fps = cvFpsCalc.get()

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        ret, image = cap.read()
        if not ret:
            break
        image = image if args.disable_image_flip else cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        hands = palm_detection(image)

        rects = []
        cropted_rotated_hands_images = []

        if len(hands) == 0:
            palm_trackid_cxcy = {}
        
        palm_trackid_box_x1y1s = {}

        if len(hands) > 0:
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
                xmin = max(0, xmin)
                xmax = min(cap_width, xmax)
                ymin = max(0, ymin)
                ymax = min(cap_height, ymax)
                degree = degrees(rotation)
                rects.append([cx, cy, (xmax - xmin), (ymax - ymin), degree])

            rects = np.asarray(rects, dtype=np.float32)

            cropted_rotated_hands_images = rotate_and_crop_rectangle(
                image=image,
                rects_tmp=rects,
                operation_when_cropping_out_of_range='padding',
            )

            for rect in rects:
                rcx = int(rect[0])
                rcy = int(rect[1])
                half_w = int(rect[2] // 2)
                half_h = int(rect[3] // 2)
                x1 = rcx - half_w
                y1 = rcy - half_h
                x2 = rcx + half_w
                y2 = rcy + half_h

                cv.rectangle(debug_image, (x1, y1), (x2, y2), (0, 128, 255), 2, cv.LINE_AA)
                
                base_point = np.asarray([rcx, rcy], dtype=np.float32)
                points = np.asarray(list(palm_trackid_cxcy.values()), dtype=np.float32)
                if len(points) > 0:
                    diff_val = points - base_point
                    all_points_distance = np.linalg.norm(diff_val, axis=1)
                    nearest_trackid = np.argmin(all_points_distance)
                    nearest_distance = all_points_distance[nearest_trackid]
                    new_trackid = int(nearest_trackid) + 1
                    if nearest_distance > 100:
                        new_trackid = next(iter(reversed(palm_trackid_cxcy))) + 1
                else:
                    new_trackid = 1

                palm_trackid_cxcy[new_trackid] = [rcx, rcy]
                palm_trackid_box_x1y1s[new_trackid] = [x1, y1]

        if len(cropted_rotated_hands_images) > 0:
            hand_landmarks, rotated_image_size_leftrights = hand_landmark(
                images=cropted_rotated_hands_images,
                rects=rects,
            )

            if len(hand_landmarks) > 0:
                pre_processed_landmarks = []
                pre_processed_point_histories = []
                
                for (trackid, x1y1), landmark, rotated_image_size_leftright in \
                        zip(palm_trackid_box_x1y1s.items(), hand_landmarks, rotated_image_size_leftrights):
                    
                    x1, y1 = x1y1
                    rotated_image_width, _, left_hand_0_or_right_hand_1 = rotated_image_size_leftright
                    thick_coef = rotated_image_width / 400
                    lines = np.asarray(
                        [np.array([landmark[point] for point in line]).astype(np.int32) for line in lines_hand]
                    )
                    radius = int(1 + thick_coef * 5)
                    cv.polylines(debug_image, lines, False, (255, 0, 0), int(radius), cv.LINE_AA)
                    for point in landmark:
                        x, y = int(point[0]), int(point[1])
                        cv.circle(debug_image, (x, y), 6, (255, 255, 255), -1)
                        cv.circle(debug_image, (x, y), 3, (0, 0, 0), -1)
                    
                    left_hand_0_or_right_hand_1 = left_hand_0_or_right_hand_1 if args.disable_image_flip else (1 - left_hand_0_or_right_hand_1)
                    handedness = 'Left' if left_hand_0_or_right_hand_1 == 0 else 'Right'

                    pre_processed_landmark = pre_process_landmark(landmark)
                    pre_processed_landmarks.append(pre_processed_landmark)
                    
                    cv.putText(debug_image, f'{handedness}', (x1, y1 - 10), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

                pre_processed_point_histories = pre_process_point_history(
                    image_width=debug_image.shape[1],
                    image_height=debug_image.shape[0],
                    point_history=point_history,
                )

                hand_sign_ids = []
                hand_sign_confs = []
                for landmark_proc in pre_processed_landmarks:
                    kp_id, kp_conf = keypoint_classifier(landmark_proc)
                    hand_sign_ids.append(kp_id)
                    hand_sign_confs.append(kp_conf)
                
                for (trackid, x1y1), landmark, hand_sign_id, hand_sign_conf in \
                    zip(palm_trackid_box_x1y1s.items(), hand_landmarks, hand_sign_ids, hand_sign_confs):
                    
                    point_history.setdefault(trackid, deque(maxlen=history_length))
                    if hand_sign_id == 2:
                        point_history[trackid].append(list(landmark[8]))
                    else:
                        point_history[trackid].append([0, 0])
                    
                    label = keypoint_classifier_labels[hand_sign_id]
                    if hand_sign_conf is not None:
                        label += f" {hand_sign_conf:.0%}"
                        
                    cv.putText(debug_image, f'{label}', (x1y1[0], x1y1[1] + 20), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

                if len(pre_point_history) > 0:
                    temp_point_history = copy.deepcopy(point_history)
                    for track_id, points in temp_point_history.items():
                        if track_id in pre_point_history:
                            pre_points = pre_point_history[track_id]
                            if points == pre_points:
                                _ = point_history.pop(track_id, None)
                pre_point_history = copy.deepcopy(point_history)

                finger_gesture_ids = []
                finger_gesture_confs = []
                temp_trackid_x1y1s = {}
                temp_pre_processed_point_history = []
                
                for (trackid, x1y1), pre_proc_hist in zip(palm_trackid_box_x1y1s.items(), pre_processed_point_histories):
                    hist_len = len(pre_proc_hist)
                    if hist_len > 0 and hist_len % (history_length * 2) == 0:
                        temp_trackid_x1y1s[trackid] = x1y1
                        temp_pre_processed_point_history.append(pre_proc_hist)

                if len(temp_pre_processed_point_history) > 0:
                    for hist_proc in temp_pre_processed_point_history:
                        ph_id, ph_conf = point_history_classifier(hist_proc)
                        finger_gesture_ids.append(ph_id)
                        finger_gesture_confs.append(ph_conf)

                if len(finger_gesture_ids) > 0:
                    for (trackid, x1y1), finger_gesture_id, finger_gesture_conf in \
                        zip(temp_trackid_x1y1s.items(), finger_gesture_ids, finger_gesture_confs):
                        
                        trackid_str = str(trackid)
                        finger_gesture_history.setdefault(trackid_str, deque(maxlen=gesture_history_length))
                        finger_gesture_history[trackid_str].append(int(finger_gesture_id))
                        
                        most_common_fg_id = Counter(finger_gesture_history[trackid_str]).most_common()[0][0]
                        classifier_label = point_history_classifier_labels[most_common_fg_id]
                        
                        if finger_gesture_conf is not None:
                            classifier_label += f" {finger_gesture_conf:.0%}"
                        
                        cv.putText(debug_image, f'Gesture: {classifier_label}', (10, 60), 
                                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

            else:
                point_history = {}
        else:
            point_history = {}

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def pre_process_landmark(landmark_list):
    if len(landmark_list) == 0:
        return []

    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    temp_landmark_list = [[temp_landmark[0] - base_x, temp_landmark[1] - base_y] for temp_landmark in temp_landmark_list]
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def pre_process_point_history(image_width, image_height, point_history):
    if len(point_history) == 0:
        return []

    temp_point_history = copy.deepcopy(point_history)
    relative_coordinate_list_by_trackid = []

    for trackid, points in temp_point_history.items():
        base_x, base_y = points[0][0], points[0][1]
        relative_coordinate_list = [[(point[0] - base_x) / image_width, (point[1] - base_y) / image_height] for point in points]
        relative_coordinate_list_1d = list(itertools.chain.from_iterable(relative_coordinate_list))
        relative_coordinate_list_by_trackid.append(relative_coordinate_list_1d)
    
    return relative_coordinate_list_by_trackid


def draw_point_history(image, point_history):
    for trackid, points in point_history.items():
        for index, point in enumerate(points):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image


def draw_info(image, fps):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()