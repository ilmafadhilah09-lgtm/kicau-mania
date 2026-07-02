import cv2
import mediapipe as mp
import pygame
import numpy as np
from PIL import Image, ImageSequence
import os
import time
import math

MP3_FILE = "kicau_mania.mp3"
GIF_FILE_LEFT = "kucing1.gif"
GIF_FILE_RIGHT = "kucing2.gif"

HAND_UP_THRESHOLD = 0.05  
DETECTION_HOLD_FRAMES = 5
DETECTION_RELEASE_FRAMES = 15
SHAKE_AMPLITUDE = 25
SHAKE_SPEED = 8
GIF_SCALE = 0.45
GIF_FRAME_DURATION = 0.08

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
)

pygame.mixer.init()
music_loaded = False
if os.path.exists(MP3_FILE):
    try:
        pygame.mixer.music.load(MP3_FILE)
        music_loaded = True
        print(f"[OK] Lagu '{MP3_FILE}' berhasil dimuat")
    except Exception as e:
        print(f"[ERROR] Gagal memuat MP3: {e}")
else:
    print(f"[WARNING] File '{MP3_FILE}' tidak ditemukan. Audio tidak akan diputar.")

def load_gif_frames(path, scale_height):
    frames = []
    if not os.path.exists(path):
        print(f"[WARNING] File '{path}' tidak ditemukan.")
        return frames
    try:
        gif = Image.open(path)
        for frame in ImageSequence.Iterator(gif):
            rgba = frame.convert("RGBA")
            w, h = rgba.size
            new_h = scale_height
            new_w = int(w * (new_h / h))
            rgba = rgba.resize((new_w, new_h), Image.LANCZOS)
            arr = np.array(rgba)
            bgra = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            frames.append(bgra)
        print(f"[OK] GIF '{path}' dimuat ({len(frames)} frame)")
    except Exception as e:
        print(f"[ERROR] Gagal memuat GIF '{path}': {e}")
    return frames


def overlay_image_alpha(background, overlay, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, bw), min(y + oh, bh)

    if x1 >= x2 or y1 >= y2:
        return background

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    overlay_crop = overlay[oy1:oy2, ox1:ox2]
    bg_crop = background[y1:y2, x1:x2]

    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0
    overlay_rgb = overlay_crop[:, :, :3].astype(np.float32)
    bg_rgb = bg_crop.astype(np.float32)

    blended = overlay_rgb * alpha + bg_rgb * (1 - alpha)
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background

def detect_hand_up(landmarks):
    if landmarks is None:
        return False

    lm = landmarks.landmark
    left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    min_vis = 0.5
    left_visible = left_wrist.visibility > min_vis and left_shoulder.visibility > min_vis
    right_visible = right_wrist.visibility > min_vis and right_shoulder.visibility > min_vis
    left_hand_up = left_visible and (left_wrist.y < left_shoulder.y - HAND_UP_THRESHOLD)
    right_hand_up = right_visible and (right_wrist.y < right_shoulder.y - HAND_UP_THRESHOLD)

    return left_hand_up or right_hand_up

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat mengakses kamera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Gagal membaca frame dari kamera.")
        return

    frame_h, frame_w = frame.shape[:2]
    gif_target_h = int(frame_h * GIF_SCALE)

    gif_frames_left = load_gif_frames(GIF_FILE_LEFT, gif_target_h)
    gif_frames_right = load_gif_frames(GIF_FILE_RIGHT, gif_target_h)

    if not gif_frames_left and gif_frames_right:
        gif_frames_left = gif_frames_right
    elif not gif_frames_right and gif_frames_left:
        gif_frames_right = gif_frames_left

    detection_counter = 0
    no_detection_counter = 0
    is_active = False

    gif_idx_left = 0
    gif_idx_right = 0
    last_gif_update_left = time.time()
    last_gif_update_right = time.time()
    shake_phase = 0.0

    print("\nKICAU MANIA DETECTOR AKTIF — angkat tangan untuk trigger! (Q untuk keluar)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        detected = detect_hand_up(results.pose_landmarks)

        if detected:
            detection_counter = min(detection_counter + 1, DETECTION_HOLD_FRAMES * 2)
            no_detection_counter = 0
        else:
            no_detection_counter = min(no_detection_counter + 1, DETECTION_RELEASE_FRAMES * 2)
            detection_counter = max(detection_counter - 1, 0)

        if not is_active and detection_counter >= DETECTION_HOLD_FRAMES:
            is_active = True
            if music_loaded:
                pygame.mixer.music.play(loops=-1)
            print(">>> Tangan terangkat! Musik & GIF AKTIF")

        if is_active and no_detection_counter >= DETECTION_RELEASE_FRAMES:
            is_active = False
            if music_loaded:
                pygame.mixer.music.stop()
            print(">>> Tangan turun. Musik & GIF BERHENTI")

        if is_active:
            now = time.time()

            if gif_frames_left and (now - last_gif_update_left) > GIF_FRAME_DURATION:
                gif_idx_left = (gif_idx_left + 1) % len(gif_frames_left)
                last_gif_update_left = now

            if gif_frames_right and (now - last_gif_update_right) > GIF_FRAME_DURATION:
                gif_idx_right = (gif_idx_right + 1) % len(gif_frames_right)
                last_gif_update_right = now

            shake_phase += SHAKE_SPEED * 0.1
            shake_y_left = int(math.sin(shake_phase) * SHAKE_AMPLITUDE)
            shake_y_right = int(math.sin(shake_phase + math.pi) * SHAKE_AMPLITUDE)
            shake_x = int(math.cos(shake_phase * 0.7) * (SHAKE_AMPLITUDE // 3))

            if gif_frames_left:
                current_left = gif_frames_left[gif_idx_left]
                lh, lw = current_left.shape[:2]
                left_x = 10 + shake_x
                left_y = (frame_h - lh) // 2 + shake_y_left
                frame = overlay_image_alpha(frame, current_left, left_x, left_y)

            if gif_frames_right:
                current_right = gif_frames_right[gif_idx_right]
                rh, rw = current_right.shape[:2]
                right_x = frame_w - rw - 10 - shake_x
                right_y = (frame_h - rh) // 2 + shake_y_right
                frame = overlay_image_alpha(frame, current_right, right_x, right_y)

        cv2.imshow("Kicau Mania Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    if music_loaded:
        pygame.mixer.music.stop()
    pygame.mixer.quit()
    pose.close()
    print("\nProgram selesai. Sampai jumpa!")


if __name__ == "__main__":
    main()