import cv2 as cv
import numpy as np
import os
from collections import deque, defaultdict
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, classify

# --- Pfade ---
MODEL_PATH_YUNET = "/home/berat/twinsecure/Yunet/face_detection_yunet_2023mar.onnx"
MODEL_PATH_CORAL = "/home/berat/twinsecure/models/model_float_edgetpu.tflite"
LABELS_PATH = "/home/berat/twinsecure/models/labels.txt"

# --- Parameter ---
CONF_THRESHOLD = 0.50       # gelockert: ab 0.5 gilt als erkannt
MARGIN_THRESHOLD = 0.05     # gelockert: kleiner Unterschied reicht
HISTORY_LEN = 10
STABLE_MIN = 3              # schneller stabilisieren (3 statt 5)
PADDING_RATIO = 1.2

# --- Coral vorbereiten ---
interpreter = make_interpreter(MODEL_PATH_CORAL)
interpreter.allocate_tensors()

with open(LABELS_PATH) as f:
    labels = f.read().splitlines()

def align_and_crop(frame, x, y, w, h, landmarks):
    (rx, ry), (lx, ly) = landmarks[0], landmarks[1]
    dx, dy = lx - rx, ly - ry
    angle = np.degrees(np.arctan2(dy, dx))

    cx = float(x + w // 2)
    cy = float(y + h // 2)
    center = (cx, cy)

    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv.INTER_LINEAR)

    s = int(max(w, h) * PADDING_RATIO)
    x0 = max(int(cx - s // 2), 0)
    y0 = max(int(cy - s // 2), 0)
    x1 = min(int(cx + s // 2), rotated.shape[1])
    y1 = min(int(cy + s // 2), rotated.shape[0])
    crop = rotated[y0:y1, x0:x1]

    return crop

def classify_face_uint8(crop):
    crop = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
    crop = cv.resize(crop, (224, 224), interpolation=cv.INTER_LINEAR)
    inp = np.expand_dims(crop, axis=0).astype(np.uint8)

    common.set_input(interpreter, inp)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=2)
    return classes

def decide_label(classes):
    if not classes:
        return "Unbekannt", 0.0

    best = classes[0]
    best_name = labels[best.id] if best.id < len(labels) else f"id_{best.id}"
    margin = best.score - classes[1].score if len(classes) > 1 else best.score

    # gelockerte Bedingungen
    if best.score < CONF_THRESHOLD or margin < MARGIN_THRESHOLD:
        return "Unbekannt", best.score

    return best_name, best.score

def box_key(x, y, w, h):
    cx, cy = x + w // 2, y + h // 2
    return (cx // 20, cy // 20)

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden")
        return

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if frame_width == 0 or frame_height == 0:
        frame_width, frame_height = 640, 480

    print(f"Kamera-Auflösung: {frame_width}x{frame_height}")
    print(f"YuNet-Modellpfad: {MODEL_PATH_YUNET}")

    if not os.path.exists(MODEL_PATH_YUNET):
        print("YuNet-Modell wurde nicht gefunden.")
        return

    try:
        detector = cv.FaceDetectorYN.create(
            model=MODEL_PATH_YUNET,
            config="",
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv.dnn.DNN_BACKEND_OPENCV,
            target_id=cv.dnn.DNN_TARGET_CPU,
        )
    except AttributeError as e:
        print("OpenCV-Version zu alt für FaceDetectorYN.")
        print("Fehler:", e)
        return

    detector.setInputSize((frame_width, frame_height))
    histories = defaultdict(lambda: deque(maxlen=HISTORY_LEN))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kein Frame gelesen – Stream beendet?")
            break

        _, faces = detector.detect(frame)
        output = frame.copy()

        if faces is not None and len(faces) > 0:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                landmarks = face[4:14].reshape((5, 2)).astype(int)
                det_score = float(face[14])

                cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(output, f"{det_score:.2f}", (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue

                crop = align_and_crop(frame, x, y, w, h, landmarks)
                if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                    continue

                classes = classify_face_uint8(crop)
                name, cls_score = decide_label(classes)

                key = box_key(x, y, w, h)
                histories[key].append(name)

                if len(histories[key]) >= STABLE_MIN:
                    counts = {}
                    for n in histories[key]:
                        counts[n] = counts.get(n, 0) + 1
                    stable_name = max(counts.items(), key=lambda t: t[1])[0]
                    show_name = stable_name
                else:
                    show_name = name

                # Anzeige mit Score
                cv.putText(output, f"{show_name} ({cls_score:.2f})", (x, y - 40),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Landmark-Punkte
                colors = [
                    (255, 0, 0), (0, 0, 255), (0, 255, 0),
                    (255, 0, 255), (0, 255, 255)
                ]
                for idx, (lx, ly) in enumerate(landmarks):
                    cv.circle(output, (lx, ly), 2, colors[idx], 2)

        cv.putText(output, "Recognition - 'q' zum Beenden", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow("TwinSecure Face-ID", output)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
