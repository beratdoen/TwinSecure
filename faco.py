import cv2 as cv
import numpy as np
import os
import time
from collections import deque, defaultdict
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, classify

# --- Pfade ---
MODEL_PATH_YUNET = "/home/berat/twinsecure/Yunet/face_detection_yunet_2023mar.onnx"
MODEL_PATH_CORAL = "/home/berat/twinsecure/models/model_float_edgetpu.tflite"
LABELS_PATH = "/home/berat/twinsecure/models/labels.txt"

# --- Parameter ---
CONF_THRESHOLD = 0.85       # <-- harter Schwellwert
MARGIN_THRESHOLD = 0.05
HISTORY_LEN = 10
STABLE_MIN = 3
PADDING_RATIO = 1.2

# --- Coral laden ---
interpreter = make_interpreter(MODEL_PATH_CORAL)
interpreter.allocate_tensors()

with open(LABELS_PATH) as f:
    labels = [line.strip() for line in f]

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
    return cv.resize(crop, (224, 224))

def classify_face_uint8(crop):
    crop = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
    inp = np.expand_dims(crop, axis=0).astype(np.uint8)
    common.set_input(interpreter, inp)
    interpreter.invoke()
    return classify.get_classes(interpreter, top_k=1)[0]

def decide_label(cls):
    if cls.score < CONF_THRESHOLD:
        return "Unbekannt", cls.score
    name = labels[cls.id] if cls.id < len(labels) else f"id_{cls.id}"
    return name, cls.score

def box_key(x, y, w, h):
    return ((x + w // 2) // 20, (y + h // 2) // 20)

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera nicht erreichbar")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

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
    detector.setInputSize((640, 480))
    histories = defaultdict(lambda: deque(maxlen=HISTORY_LEN))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, faces = detector.detect(frame)
        output = frame.copy()

        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                landmarks = face[4:14].reshape((5, 2)).astype(int)

                cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                crop = align_and_crop(frame, x, y, w, h, landmarks)
                if crop.size == 0:
                    continue

                cls = classify_face_uint8(crop)
                name, score = decide_label(cls)

                key = box_key(x, y, w, h)
                histories[key].append(name)
                if len(histories[key]) >= STABLE_MIN:
                    name = max(set(histories[key]), key=histories[key].count)

                cv.putText(output, f"{name} ({score:.2f})", (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv.putText(output, "Face-ID - 'q' zum Beenden", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv.imshow("TwinSecure Face-ID", output)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.1)  # 10 FPS

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
