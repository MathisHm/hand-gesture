import csv
import time
import copy
import itertools
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    arr = np.empty((0, 2), int)
    for lm in landmarks.landmark:
        x = min(int(lm.x * image_width), image_width - 1)
        y = min(int(lm.y * image_height), image_height - 1)
        arr = np.append(arr, [[x, y]], axis=0)
    x, y, w, h = cv.boundingRect(arr)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    w, h = image.shape[1], image.shape[0]
    pts = []
    for lm in landmarks.landmark:
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)
        pts.append([x, y])
    return pts


def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    bx, by = temp[0]
    for i in range(len(temp)):
        temp[i][0] -= bx
        temp[i][1] -= by
    temp = list(itertools.chain.from_iterable(temp))
    mv = max(map(abs, temp))
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

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.5)

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        key_labels = [row[0] for row in csv.reader(f)]

    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        gesture_labels = [row[0] for row in csv.reader(f)]

    fps_calc = CvFpsCalc(buffer_len=10)

    hist_len = 16
    point_history = deque(maxlen=hist_len)
    gesture_history = deque(maxlen=hist_len)

    frame_count = 0
    detect_count = 0
    gesture_counter = Counter()
    processing_times = []

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        df = copy.deepcopy(frame)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            detect_count += 1
            for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(df, lm)
                lm_list = calc_landmark_list(df, lm)
                lm_proc = pre_process_landmark(lm_list)
                pt_proc = pre_process_point_history(df, point_history)
                sign_id = keypoint_classifier(lm_proc)
                if sign_id == 2:
                    point_history.append(lm_list[8])
                else:
                    point_history.append([0, 0])
                gesture_id = 0
                if len(pt_proc) == hist_len * 2:
                    gesture_id = point_history_classifier(pt_proc)
                gesture_history.append(gesture_id)
                mc = Counter(gesture_history).most_common()[0][0]
                gesture_counter[gesture_labels[mc]] += 1
        else:
            point_history.append([0, 0])

        end = time.time()
        processing_times.append(end - start)

    cap.release()

    avg_fps = frame_count / sum(processing_times) if processing_times else 0
    avg_proc = sum(processing_times) / len(processing_times) if processing_times else 0

    header = ["video", "frames", "detections", "detection_rate", "avg_fps", "avg_processing_time"]
    gesture_keys = sorted(list(gesture_counter.keys()))
    header.extend(gesture_keys)

    write_header = False
    try:
        with open(csv_path, "r", newline='', encoding="utf-8") as f:
            pass
    except:
        write_header = True

    with open(csv_path, "a", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        row = [
            video_path,
            frame_count,
            detect_count,
            detect_count / frame_count if frame_count else 0,
            avg_fps,
            avg_proc
        ]
        row.extend([gesture_counter[k] for k in gesture_keys])
        w.writerow(row)


if __name__ == "__main__":
    run_benchmark("benchmark_video.mp4")
