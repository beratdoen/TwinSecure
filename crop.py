import cv2 as cv
import numpy as np
import os

# --- Pfade ---
MODEL_PATH_YUNET = "/home/berat/twinsecure/Yunet/face_detection_yunet_2023mar.onnx"
OUTPUT_DIR = "/home/berat/twinsecure/dataset_faces"

# --- Parameter ---
PADDING_RATIO = 1.3
CROP_SIZE = 224

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def align_and_crop(frame, x, y, w, h, landmarks):
    (rx, ry), (lx, ly) = landmarks[0], landmarks[1]
    dx, dy = lx - rx, ly - ry
    angle = np.degrees(np.arctan2(dy, dx))
    cx = float(x + w // 2)
    cy = float(y + h // 2)
    M = cv.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    s = int(max(w, h) * PADDING_RATIO)
    x0 = max(int(cx - s // 2), 0)
    y0 = max(int(cy - s // 2), 0)
    x1 = min(int(cx + s // 2), rotated.shape[1])
    y1 = min(int(cy + s // 2), rotated.shape[0])
    crop = rotated[y0:y1, x0:x1]
    return cv.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv.INTER_LINEAR)

def main():
    person_name = input("Name der Person: ").strip()
    save_dir = os.path.join(OUTPUT_DIR, person_name)
    ensure_dir(save_dir)

    # vorhandene Bilder zählen, damit wir weitermachen können
    existing = [f for f in os.listdir(save_dir) if f.endswith(".png")]
    counter = len(existing) + 1

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden")
        return

    if not os.path.exists(MODEL_PATH_YUNET):
        print("YuNet-Modell nicht gefunden!")
        return

    detector = cv.FaceDetectorYN_create(
        model=MODEL_PATH_YUNET,
        config="",
        input_size=(320, 320),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=3,
        backend_id=cv.dnn.DNN_BACKEND_OPENCV,
        target_id=cv.dnn.DNN_TARGET_CPU,
    )

    print("Leertaste = Foto speichern,  q = Ende")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        output = frame.copy()

        # nur das erste (beste) Gesicht verwenden
        if faces is not None and len(faces) > 0:
            face = faces[0]
            x, y, fw, fh = face[:4].astype(int)
            landmarks = face[4:14].reshape((5, 2)).astype(int)
            cv.rectangle(output, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

        cv.imshow("Cropper - Leertaste = Foto", output)
        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):   # Leertaste
            if faces is not None and len(faces) > 0:
                crop = align_and_crop(frame, x, y, fw, fh, landmarks)
                if crop.size > 0:
                    filename = os.path.join(save_dir, f"{person_name}_{counter:04d}.png")
                    cv.imwrite(filename, crop)
                    print(f"Gespeichert: {filename}")
                    counter += 1
                    # Vorschau des Crops
                    cv.imshow("Letzter Crop", crop)
            else:
                print("Kein Gesicht erkannt – nichts gespeichert.")

    cap.release()
    cv.destroyAllWindows()
    print(f"Fertig! Insgesamt {counter - 1} Bilder in {save_dir}")

if __name__ == "__main__":
    main()
