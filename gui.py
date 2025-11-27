import tkinter as tk
import cv2
from PIL import Image, ImageTk
import datetime
import numpy as np
import tflite_runtime.interpreter as tflite
import os

class TwinSecure:
    def __init__(self, root):
        self.root = root
        self.root.title("TwinSecure")
        self.root.config(bg="#0F0F0F")
        self.root.attributes("-fullscreen", True)
        self.root.resizable(False, False)

        # Modell laden
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_edgetpu.tflite")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Titel
        self.title_label = tk.Label(self.root, text="TwinSecure", font=("Segoe UI", 36, "bold"), fg="#00BFFF", bg="#0F0F0F")
        self.title_label.place(x=30, y=20)

        # Türstatus
        self.lock_icon = tk.Label(self.root, text="🔒", font=("Segoe UI", 40), bg="#0F0F0F", fg="#FF3D00")
        self.lock_icon.place(x=30, y=80)
        self.lock_status_label = tk.Label(self.root, text="Türschloss\nVerriegelt", font=("Segoe UI", 14, "bold"), bg="#1A1A1A", fg="#FFFFFF", relief="solid", width=18, height=3)
        self.lock_status_label.place(x=100, y=90)

        self.toggle_button = tk.Button(self.root, text="Umschalten", font=("Segoe UI", 12, "bold"), command=self.toggle_lock, bg="#00BFFF", fg="#FFFFFF", relief="flat", width=16, height=2)
        self.toggle_button.place(x=100, y=160)

        # Authentifizierungs-Auswahl
        self.auth_label = tk.Label(self.root, text="Authentifizierungs-Auswahl", font=("Segoe UI", 14, "bold"), bg="#0F0F0F", fg="#FFFFFF")
        self.auth_label.place(x=30, y=240)

        self.venen_button = tk.Button(self.root, text="Venenerkennung", font=("Segoe UI", 12, "bold"), command=self.set_venen_mode, bg="#1A1A1A", fg="#00BFFF", relief="flat", width=20, height=2)
        self.venen_button.place(x=30, y=280)

        self.gesicht_button = tk.Button(self.root, text="Gesichtserkennung", font=("Segoe UI", 12, "bold"), command=self.set_gesicht_mode, bg="#1A1A1A", fg="#00BFFF", relief="flat", width=20, height=2)
        self.gesicht_button.place(x=220, y=280)

        # Live-Venenanzeige
        self.camera_label = tk.Label(self.root, text="Live-Venenansicht", font=("Segoe UI", 14, "bold"), bg="#0F0F0F", fg="#00BFFF")
        self.camera_label.place(x=600, y=60)

        self.live_display = tk.Label(self.root, bg="#000000", relief="flat")
        self.live_display.place(x=600, y=100, width=480, height=320)

        # Ergebnisanzeige oben rechts
        self.result_label = tk.Label(self.root, text="Erkannt: —", font=("Segoe UI", 16, "bold"), bg="#0F0F0F", fg="#00BFFF")
        self.result_label.place(x=600, y=30)

        # Modusanzeige unterhalb der Kamera
        self.mode_label = tk.Label(self.root, text="Modus: Venenerkennung aktiv", font=("Segoe UI", 12, "bold"), bg="#0F0F0F", fg="#FFFFFF", wraplength=460, justify="left")
        self.mode_label.place(x=600, y=440)

        # Uhrzeit & Datum unten links
        self.time_label = tk.Label(self.root, font=("Segoe UI", 14, "bold"), fg="#00BFFF", bg="#0F0F0F", justify="left")
        self.time_label.place(x=30, y=680)
        self.update_time()

        # Kamera starten
        self.cap = cv2.VideoCapture(0)
        self.venen_mode_active = False
        self.update_frame()

        # Wartungsanzeige
        self.maintenance_label = None

    def update_time(self):
        now = datetime.datetime.now()
        weekday = now.strftime('%a').upper()[:2]  # z. B. "FR"
        time_string = f"{weekday} {now.strftime('%H:%M:%S')}"
        self.time_label.config(text=time_string)
        self.root.after(1000, self.update_time)

    def enhance_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        veins = clahe.apply(blur)
        veins = cv2.convertScaleAbs(veins, alpha=1.2, beta=10)
        veins = cv2.bitwise_not(veins)
        return veins

    def predict_with_tflite(self, frame):
        img_resized = cv2.resize(frame, (128, 128))
        img_normalized = img_resized / 255.0
        input_data = np.expand_dims(img_normalized, axis=0).astype(np.float32)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0]
        return prediction

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            if self.venen_mode_active:
                enhanced = self.enhance_frame(frame)
                prediction = self.predict_with_tflite(enhanced)
                confidence = np.random.randint(85, 100)  # Simulierte Prozentanzeige

                display_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                cv2.putText(display_frame, f"{prediction} ({confidence}%)", (25, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 191, 255), 2)

                self.result_label.config(text=f"Erkannt: {prediction} ({confidence}%)")
            else:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(display_frame)
            image = ImageTk.PhotoImage(image)
            self.live_display.config(image=image)
            self.live_display.image = image

        self.root.after(10, self.update_frame)

    def toggle_lock(self):
        current = self.lock_status_label.cget("text")
        if "Verriegelt" in current:
            self.lock_status_label.config(text="🔓 Türschloss\nEntsperrt")
            self.lock_icon.config(text="🔓", fg="#00BFFF")
        else:
            self.lock_status_label.config(text="🔒 Türschloss\nVerriegelt")
            self.lock_icon.config(text="🔒", fg="#FF3D00")

    def set_venen_mode(self):
        self.mode_label.config(text="Modus: Venenerkennung aktiv")
        self.venen_mode_active = True
        if self.maintenance_label:
            self.maintenance_label.place_forget()
        self.live_display.place(x=600, y=100, width=480, height=320)
        self.mode_label.place(x=600, y=440)

    def set_gesicht_mode(self):
        self.mode_label.config(text="Modus: Gesichtserkennung aktiv")
        self.venen_mode_active = False
        self.live_display.place_forget()
        if self.maintenance_label:
            self.maintenance_label.place_forget()
        self.maintenance_label = tk.Label(self.root, text="Gesichtserkennung in Wartung", font=("Segoe UI", 14, "bold"), bg="#0F0F0F", fg="#FF3D00", relief="flat", width=30, height=2)
        self.maintenance_label.place(x=600, y=300)

if __name__ == "__main__":
    root = tk.Tk()
    app = TwinSecure(root)
    root.mainloop()
