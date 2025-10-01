import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import datetime

DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Attendance System")

        self.video = cv2.VideoCapture(0)
        self.panel = tk.Label(root)
        self.panel.pack()

        self.status_label = tk.Label(root, text="System Ready", font=("Arial", 12))
        self.status_label.pack(pady=10)

        self.captured_faces = []

        self.capture_btn = tk.Button(root, text="Capture Attendance", command=self.capture_attendance)
        self.capture_btn.pack(pady=5)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            self.frame = frame
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk
            self.panel.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def capture_attendance(self):
        if not hasattr(self, "frame"):
            messagebox.showwarning("Error", "No frame captured from camera.")
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            messagebox.showwarning("No Face", "No face detected.")
            return

        # Take the first detected face
        (x, y, w, h) = faces[0]
        face_img = self.frame[y:y+h, x:x+w]

        # Auto-assign name based on existing dataset folders
        existing_people = os.listdir(DATASET_DIR)
        new_person_id = len(existing_people) + 1
        person_name = f"person_{new_person_id}"
        person_dir = os.path.join(DATASET_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)

        # Save captured face image
        file_name = f"{person_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(person_dir, file_name)
        cv2.imwrite(path, face_img)

        # Mark attendance in CSV
        with open(ATTENDANCE_FILE, "a") as f:
            f.write(f"{person_name},{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        self.status_label.config(text=f"Attendance marked for {person_name}")
        messagebox.showinfo("Attendance", f"Attendance marked for {person_name}")

        # Release camera after capture
        self.video.release()
        self.root.after(500, self.root.destroy)  # Close GUI after 0.5 sec

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop()
