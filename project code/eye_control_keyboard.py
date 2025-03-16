import cv2import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from collections import deque, defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def initialize_csv():
    with open('gaze_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Left Eye X', 'Left Eye Y', 'Right Eye X', 'Right Eye Y',
                         'Smoothed X', 'Smoothed Y', 'Best Match Key', 'Typed Text',
                         'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'WPM', 'Error Rate'])

def save_gaze_data(timestamp, left_eye_x, left_eye_y, right_eye_x, right_eye_y,
                   smoothed_x, smoothed_y, best_match_key, typed_text,
                   accuracy, sensitivity, specificity, precision, wpm, error_rate):
    with open('gaze_data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, left_eye_x, left_eye_y, right_eye_x, right_eye_y,
                         smoothed_x, smoothed_y, best_match_key, typed_text,
                         accuracy, sensitivity, specificity, precision, wpm, error_rate])

keyboard_layout = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', 'Del', 'Spc'],
]

padding = 10
font_scale = 1.0

def scale_gaze(x, y, width, height, calibration_points):
    top_left, top_right, bottom_left, bottom_right = calibration_points
    x_ratio = (x - top_left[0]) / (top_right[0] - top_left[0])
    y_ratio = (y - top_left[1]) / (bottom_left[1] - top_left[1])
    screen_x = int(width * x_ratio)
    screen_y = int(height * y_ratio)
    return screen_x, screen_y


def draw_keyboard(frame, screen_width, screen_height):
    key_width = (screen_width - padding * 11) // 10
    key_height = key_width
    start_x = (screen_width - (key_width * 10 + padding * 9)) // 2
    start_y = screen_height - (key_height * 3 + padding * 2)
    for row_idx, row in enumerate(keyboard_layout):
        for col_idx, key in enumerate(row):
            x = start_x + (key_width + padding) * col_idx
            y = start_y + (key_height + padding) * row_idx
            cv2.rectangle(frame, (x, y), (x + key_width, y + key_height), (255, 255, 255), 2)
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = x + (key_width - text_size[0]) // 2
            text_y = y + (key_height + text_size[1]) // 2
            cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), 2)
    return start_x, start_y, key_width, key_height

def smooth_coordinates(x, y, smoothing_factor=0.8):
    if len(smoothed_coordinates) >= 2:
        prev_x, prev_y = smoothed_coordinates[-1]
        x = prev_x * smoothing_factor + x * (1 - smoothing_factor)
        y = prev_y * smoothing_factor + y * (1 - smoothing_factor)
    smoothed_coordinates.append((x, y))
    return int(x), int(y)


def calculate_metrics(typed_text, target_text, gaze_history, correct_gaze_count, total_gaze_count):
    accuracy = correct_gaze_count / total_gaze_count if total_gaze_count > 0 else 0
    true_positives = sum(1 for gaze in gaze_history if gaze[0] == gaze[1])
    sensitivity = true_positives / len(gaze_history) if gaze_history else 0
    true_negatives = sum(1 for gaze in gaze_history if gaze[0] != gaze[1])
    specificity = true_negatives / len(gaze_history) if gaze_history else 0
    precision = true_positives / (true_positives + (len(gaze_history) - true_positives)) if gaze_history else 0
    wpm = (len(typed_text.split()) / (time.time() - start_time)) * 60 if time.time() - start_time > 0 else 0
    error_rate = (sum(1 for a, b in zip(typed_text, target_text) if a != b) / len(target_text)) * 100 if target_text else 0
    return accuracy, sensitivity, specificity, precision, wpm, error_rate


def create_heatmap(key_gaze_counts):
  
    heatmap_data = np.zeros((len(keyboard_layout), len(keyboard_layout[0])))
    for row_idx, row in enumerate(keyboard_layout):
        for col_idx, key in enumerate(row):
            heatmap_data[row_idx, col_idx] = key_gaze_counts.get(key, 0)

    plt.figure(figsize=(12, 4))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=keyboard_layout[0],
                yticklabels=["Q W E R T Y U I O P", "A S D F G H J K L", "Z X C V B N M"],
                cbar_kws={'label':  'Gaze Duration (seconds)'})
    plt.title("Keyboard Key Gaze Duration (Heatmap)")
    plt.xlabel("Keys")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.show()


def plot_operation_times(operation_times):
    plt.figure(figsize=(10, 4))
    plt.plot(operation_times, label="Operation Times")
    plt.title("Operation Times Graph")
    plt.xlabel("Step")
    plt.ylabel("Time (second)")
    plt.legend()
    plt.show()


def plot_metrics(accuracy_list, sensitivity_list, specificity_list, precision_list, wpm_list, error_rate_list):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.plot(accuracy_list, label='Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Time')
    plt.ylabel('Accuracy')

    plt.subplot(2, 3, 2)
    plt.plot(sensitivity_list, label='Sensitivity')
    plt.title('Sensitivity Over Time')
    plt.xlabel('Time')
    plt.ylabel('Sensitivity')

    plt.subplot(2, 3, 3)
    plt.plot(specificity_list, label='Specificity')
    plt.title('Specificity Over Time')
    plt.xlabel('Time')
    plt.ylabel('Specificity')

    plt.subplot(2, 3, 4)
    plt.plot(precision_list, label='Precision')
    plt.title('Precision Over Time')
    plt.xlabel('Time')
    plt.ylabel('Precision')

    plt.subplot(2, 3, 5)
    plt.plot(wpm_list, label='WPM')
    plt.title('WPM Over Time')
    plt.xlabel('Time')
    plt.ylabel('WPM')

    plt.subplot(2, 3, 6)
    plt.plot(error_rate_list, label='Error Rate')
    plt.title('Error Rate Over Time')
    plt.xlabel('Time')
    plt.ylabel('Error Rate')

    plt.tight_layout()
    plt.show()


calibration_points = []
gaze_reference_points = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
smoothed_coordinates = []
key_gaze_counts = defaultdict(int)
gaze_history = deque(maxlen=100)
typed_text = ""
correct_gaze_count, total_gaze_count = 0, 0
accuracy_list, sensitivity_list, specificity_list, precision_list, wpm_list, error_rate_list = [], [], [], [], [], []
operation_times = []
start_time = time.time()
last_key_time = time.time()


fixation_queue = deque(maxlen=5)


keyboard_entry_time = None


target_text = input("What dou you want to write ? ").upper()
initialize_csv()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    left_eye_x, left_eye_y, right_eye_x, right_eye_y = 0, 0, 0, 0
    smoothed_x, smoothed_y = 0, 0
    best_match_key = ""

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = face_landmarks.landmark[474]
            right_eye = face_landmarks.landmark[473]
            left_eye_x = int(left_eye.x * frame.shape[1])
            left_eye_y = int(left_eye.y * frame.shape[0])
            right_eye_x = int(right_eye.x * frame.shape[1])
            right_eye_y = int(right_eye.y * frame.shape[0])

            if len(calibration_points) < 4:
                cv2.putText(frame, f"Look at the corner {len(calibration_points) + 1}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                for point in gaze_reference_points:
                    px, py = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                    cv2.circle(frame, (px, py), 20, (0, 0, 255), -1)
                if cv2.waitKey(33) == ord(' '):
                    calibration_points.append((left_eye_x, left_eye_y))
            else:
                if keyboard_entry_time is None:
                    keyboard_entry_time = time.time()

                screen_x, screen_y = scale_gaze(left_eye_x, left_eye_y, frame.shape[1], frame.shape[0], calibration_points)
                smoothed_x, smoothed_y = smooth_coordinates(screen_x, screen_y)

                start_x, start_y, key_width, key_height = draw_keyboard(frame, frame.shape[1], frame.shape[0])
                smoothed_x = np.clip(smoothed_x, start_x, start_x + key_width * 9 + padding * 8)
                smoothed_y = np.clip(smoothed_y, start_y, start_y + key_height * 3 + padding * 2)
                cv2.circle(frame, (smoothed_x, smoothed_y), 20, (0, 0, 255), -1)

                min_distance = float('inf')
                best_match_key = ""
                for row_idx, row in enumerate(keyboard_layout):
                    for col_idx, key in enumerate(row):
                        key_x = start_x + (key_width + padding) * col_idx
                        key_y = start_y + (key_height + padding) * row_idx
                        center_x = key_x + key_width // 2
                        center_y = key_y + key_height // 2
                        distance = np.hypot(smoothed_x - center_x, smoothed_y - center_y)
                        if distance < min_distance:
                            min_distance = distance
                            best_match_key = key

                # Fixation kontrolü
                fixation_queue.append(best_match_key)
                common_key, count = Counter(fixation_queue).most_common(1)[0]

                if count >= 5:
                    cv2.putText(frame, f'{common_key}', (smoothed_x - 15, smoothed_y - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if time.time() - keyboard_entry_time >= 2:
                        if time.time() - last_key_time > 3:
                            if common_key == "Del":
                                typed_text = typed_text[:-1]
                            elif common_key == "Spc":
                                typed_text += " "
                            else:
                                typed_text += common_key
                            last_key_time = time.time()
                            total_gaze_count += 1
                            if len(typed_text) > 0 and len(typed_text) <= len(target_text):
                                if common_key == target_text[len(typed_text) - 1]:
                                    correct_gaze_count += 1
                                gaze_history.append((common_key, target_text[len(typed_text) - 1]))

               
                key_gaze_counts[best_match_key] += 1

    cv2.putText(frame, f"Text: {typed_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    timestamp = time.time()

    if len(calibration_points) >= 4:
        accuracy, sensitivity, specificity, precision, wpm, error_rate = calculate_metrics(
            typed_text, target_text, gaze_history, correct_gaze_count, total_gaze_count)
        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        precision_list.append(precision)
        wpm_list.append(wpm)
        error_rate_list.append(error_rate)
        save_gaze_data(timestamp, left_eye_x, left_eye_y, right_eye_x, right_eye_y,
                       smoothed_x, smoothed_y, best_match_key, typed_text,
                       accuracy, sensitivity, specificity, precision, wpm, error_rate)

  
    operation_times.append(time.time() - start_time)

    cv2.imshow('Gaze Keyboard', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


create_heatmap(key_gaze_counts)


plot_operation_times(operation_times)


plot_metrics(accuracy_list, sensitivity_list, specificity_list, precision_list, wpm_list, error_rate_list)
