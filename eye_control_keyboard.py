import cv2
import mediapipe as mp
import numpy as np
import time

# Mediapipe FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Kalibrasyon noktaları ve göz bebeği koordinatları
calibration_points = []
gaze_reference_points = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
calibration_complete = False

# Klavye harfleri
keyboard_layout = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', 'Del', 'Spc'],  # Del ve Space yan yana
]

# Klavye düzeni ve boyutları
key_width = 120  # Tuşları büyütüyoruz
key_height = 120
padding = 20

# Göz bebeği pozisyonunu çerçeveye ölçekleme fonksiyonu
def scale_gaze(x, y, width, height, calibration_points):
    top_left, top_right, bottom_left, bottom_right = calibration_points
    x_ratio = (x - top_left[0]) / (top_right[0] - top_left[0])
    y_ratio = (y - top_left[1]) / (bottom_left[1] - top_left[1])
    screen_x = int(width * x_ratio)
    screen_y = int(height * y_ratio)
    return screen_x, screen_y

# Klavye çizme fonksiyonu
def draw_keyboard(frame, screen_width, screen_height):
    start_x = (screen_width - (key_width * 10 + padding * 9)) // 2
    start_y = screen_height - (key_height * 3 + padding * 2)  # Klavye biraz daha yukarıda

    for row_idx, row in enumerate(keyboard_layout):
        for col_idx, key in enumerate(row):
            x = start_x + (key_width + padding) * col_idx
            y = start_y + (key_height + padding) * row_idx
            cv2.rectangle(frame, (x, y), (x + key_width, y + key_height), (255, 255, 255), 2)
            cv2.putText(frame, key, (x + 35, y + 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    return start_x, start_y

# Başlangıçta zamanlayıcı için
gaze_time = time.time()
current_key = None
typed_text = ""  # Yazılan metin
last_key_time = time.time()

# Göz koordinatlarını yumuşatmak için bir liste
smoothed_coordinates = []

# Göz koordinatlarını yumuşatma (smoothing) fonksiyonu
def smooth_coordinates(x, y, smoothing_factor=0.8):
    if len(smoothed_coordinates) >= 2:
        prev_x, prev_y = smoothed_coordinates[-1]
        x = prev_x * smoothing_factor + x * (1 - smoothing_factor)
        y = prev_y * smoothing_factor + y * (1 - smoothing_factor)
    smoothed_coordinates.append((x, y))
    return int(x), int(y)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Yüz ve göz işaretlerini bul
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Göz bebeği koordinatlarını al
            left_eye = face_landmarks.landmark[474]
            right_eye = face_landmarks.landmark[473]
            left_eye_x, left_eye_y = int(left_eye.x * frame.shape[1]), int(left_eye.y * frame.shape[0])
            right_eye_x, right_eye_y = int(right_eye.x * frame.shape[1]), int(right_eye.y * frame.shape[0])

            # Kalibrasyon tamamlanmadıysa köşelerdeki kırmızı noktalara bakmanız istenir
            if len(calibration_points) < 4:
                cv2.putText(frame, f"Lutfen koseye bakin {len(calibration_points)+1}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                for idx, point in enumerate(gaze_reference_points):
                    px, py = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                    cv2.circle(frame, (px, py), 20, (0, 0, 255), -1)  # Noktaları büyütüyoruz

                # Kalibrasyon noktalarına bakıldığında kayıt et
                if len(calibration_points) < 4 and cv2.waitKey(33) == ord(' '):
                    calibration_points.append((left_eye_x, left_eye_y))

            else:
                # Kalibrasyon tamamlandıktan sonra göz hareketine göre mavi noktayı hareket ettir
                calibration_complete = True
                screen_x, screen_y = scale_gaze(left_eye_x, left_eye_y, frame.shape[1], frame.shape[0], calibration_points)

                # Koordinatları yumuşatma
                smoothed_x, smoothed_y = smooth_coordinates(screen_x, screen_y)

                # Klavye koordinatları içinde sınırlandırma
                start_x, start_y = draw_keyboard(frame, frame.shape[1], frame.shape[0])

                # Göz noktasını ekranın klavye alanı içinde sınırlama
                smoothed_x = np.clip(smoothed_x, start_x, start_x + key_width * 9 + padding * 8)
                smoothed_y = np.clip(smoothed_y, start_y, start_y + key_height * 3 + padding * 2)

                cv2.circle(frame, (smoothed_x, smoothed_y), 20, (0, 0, 255), -1)  # Noktayı ekranda göster

                # Klavye çizme
                draw_keyboard(frame, frame.shape[1], frame.shape[0])

                # Gözün hangi tuşa baktığını belirle
                best_match_key = None
                min_distance = float('inf')

                for row_idx, row in enumerate(keyboard_layout):
                    for col_idx, key in enumerate(row):
                        key_x = start_x + (key_width + padding) * col_idx
                        key_y = start_y + (key_height + padding) * row_idx
                        key_rect = (key_x, key_y, key_x + key_width, key_y + key_height)

                        # Eğer göz noktası tuşun üzerinde ise
                        if key_rect[0] <= smoothed_x <= key_rect[2] and key_rect[1] <= smoothed_y <= key_rect[3]:
                            # Eğer bu tuş ile mesafe daha küçükse güncelle
                            distance = np.sqrt((smoothed_x - key_x) ** 2 + (smoothed_y - key_y) ** 2)
                            if distance < min_distance:
                                min_distance = distance
                                best_match_key = key
                
                # Eğer tahmin edilen tuş varsa, yeşil renkte göster
                if best_match_key:
                    cv2.putText(frame, f"Tahmin edilen: {best_match_key}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    current_key = best_match_key

                # Yazı ekleme ve zamanlama
                if current_key and time.time() - last_key_time > 3:  # 3 saniye bekleme süresi
                    if current_key == "Spc":
                        typed_text += " "
                    elif current_key == "Del":
                        typed_text = typed_text[:-1]
                    else:
                        typed_text += current_key
                    last_key_time = time.time()

                # Yazıyı şeffaf kutu içinde ekranda göster (beyaz siyah font)
                overlay = frame.copy()
                cv2.rectangle(overlay, (50, 50), (frame.shape[1] - 50, 150), (0, 0, 0), -1)  # Kutu oluştur
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                cv2.putText(frame, typed_text, (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Görüntüyü göster
    cv2.imshow("Goz Tabanli Klavye", frame)

    # 'q' tuşuna basılınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
