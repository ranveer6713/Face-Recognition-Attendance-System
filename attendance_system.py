import cv2
import numpy as np
import pandas as pd
import os
import threading
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import pickle

# TensorFlow/Keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class AttendanceSystem:
    def __init__(self):
        # Paths
        self.dataset_path = "dataset"
        self.model_path = "model"
        self.attendance_path = "attendance"
        
        # Create directories
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.attendance_path, exist_ok=True)
        
        # Model variables
        self.model = None
        self.label_encoder = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Attendance tracking
        self.today_attendance = set()
        self.confidence_threshold = 0.7
        
        # Camera
        self.camera = None
        self.is_capturing = False
        
    def capture_faces(self, student_id, student_name, num_samples=50):
        """Capture multiple face samples for a student"""
        student_folder = os.path.join(self.dataset_path, f"{student_id}_{student_name}")
        os.makedirs(student_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        count = 0
        
        print(f"Capturing faces for {student_name} (ID: {student_id})")
        print("Press 'q' to quit early")
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Extract and save face
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (128, 128))
                
                # Save image
                img_path = os.path.join(student_folder, f"{count}.jpg")
                cv2.imwrite(img_path, face_resized)
                count += 1
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Captured: {count}/{num_samples}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Capture Faces - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        return count
    
    def load_dataset(self):
        """Load all face images from dataset folder"""
        X = []
        y = []
        
        if not os.path.exists(self.dataset_path):
            return None, None
            
        for person_folder in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_folder)
            
            if not os.path.isdir(person_path):
                continue
                
            # Extract student ID and name from folder name
            label = person_folder.split('_')[0]  # Use ID as label
            
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    X.append(img)
                    y.append(label)
        
        if len(X) == 0:
            return None, None
            
        X = np.array(X)
        y = np.array(y)
        
        # Normalize
        X = X.astype('float32') / 255.0
        X = X.reshape(-1, 128, 128, 1)
        
        return X, y
    
    def build_cnn_model(self, num_classes):
        """Build CNN model for face recognition"""
        model = keras.Sequential([
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, epochs=20):
        """Train the CNN model"""
        print("Loading dataset...")
        X, y = self.load_dataset()
        
        if X is None or len(X) == 0:
            print("No data found in dataset folder!")
            return False
            
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        # Build model
        self.model = self.build_cnn_model(len(self.label_encoder.classes_))
        
        # Train
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        # Save model and encoder
        self.model.save(os.path.join(self.model_path, 'face_recognition_model.h5'))
        with open(os.path.join(self.model_path, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        print(f"Model trained! Final accuracy: {history.history['accuracy'][-1]:.4f}")
        return True
    
    def load_trained_model(self):
        """Load pre-trained model"""
        model_file = os.path.join(self.model_path, 'face_recognition_model.h5')
        encoder_file = os.path.join(self.model_path, 'label_encoder.pkl')
        
        if not os.path.exists(model_file) or not os.path.exists(encoder_file):
            return False
            
        self.model = keras.models.load_model(model_file)
        with open(encoder_file, 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        return True
    
    def recognize_face(self, face_img):
        """Recognize a face using trained model"""
        if self.model is None or self.label_encoder is None:
            return None, 0.0
            
        # Preprocess
        face_resized = cv2.resize(face_img, (128, 128))
        face_normalized = face_resized.astype('float32') / 255.0
        face_input = face_normalized.reshape(1, 128, 128, 1)
        
        # Predict
        predictions = self.model.predict(face_input, verbose=0)
        confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        if confidence >= self.confidence_threshold:
            student_id = self.label_encoder.inverse_transform([predicted_class])[0]
            return student_id, confidence
            
        return None, confidence
    
    def get_student_name(self, student_id):
        """Get student name from folder structure"""
        for folder in os.listdir(self.dataset_path):
            if folder.startswith(f"{student_id}_"):
                return folder.split('_', 1)[1]
        return "Unknown"
    
    def mark_attendance(self, student_id, student_name):
        """Mark attendance in CSV"""
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Check if already marked today
        if student_id in self.today_attendance:
            return False
            
        # Create CSV file for today
        csv_file = os.path.join(self.attendance_path, f"attendance_{today}.csv")
        
        # Append to CSV
        data = {
            'ID': [student_id],
            'Name': [student_name],
            'Date': [today],
            'Time': [current_time]
        }
        df = pd.DataFrame(data)
        
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, mode='w', header=True, index=False)
            
        self.today_attendance.add(student_id)
        print(f"Attendance marked: {student_name} (ID: {student_id}) at {current_time}")
        return True
    
    def start_attendance(self, callback=None):
        """Start real-time attendance marking"""
        if not self.load_trained_model():
            print("Model not found! Please train the model first.")
            return
            
        self.today_attendance.clear()
        cap = cv2.VideoCapture(0)
        
        print("Starting attendance system... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                
                # Recognize
                student_id, confidence = self.recognize_face(face)
                
                if student_id:
                    student_name = self.get_student_name(student_id)
                    label = f"{student_name} ({confidence:.2f})"
                    color = (0, 255, 0)
                    
                    # Mark attendance
                    if self.mark_attendance(student_id, student_name):
                        if callback:
                            callback(f"✓ {student_name} marked present")
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.imshow('Attendance System - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()


class AttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        self.system = AttendanceSystem()
        
        # Title
        title = tk.Label(root, text="Face Recognition Attendance System", 
                        font=("Arial", 18, "bold"), bg="#2c3e50", fg="white", pady=10)
        title.pack(fill="x")
        
        # Main frame
        main_frame = tk.Frame(root, bg="#ecf0f1", padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        # Buttons
        btn_style = {
            "font": ("Arial", 12),
            "bg": "#3498db",
            "fg": "white",
            "activebackground": "#2980b9",
            "activeforeground": "white",
            "relief": "flat",
            "cursor": "hand2",
            "width": 25,
            "height": 2
        }
        
        tk.Button(main_frame, text="📸 Register New Student", 
                 command=self.register_student, **btn_style).pack(pady=10)
        
        tk.Button(main_frame, text="🧠 Train Model", 
                 command=self.train_model_thread, **btn_style).pack(pady=10)
        
        tk.Button(main_frame, text="✅ Start Attendance", 
                 command=self.start_attendance_thread, **btn_style).pack(pady=10)
        
        tk.Button(main_frame, text="📊 View Attendance Records", 
                 command=self.view_attendance, **btn_style).pack(pady=10)
        
        tk.Button(main_frame, text="⚙️ Settings", 
                 command=self.show_settings, **btn_style).pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Ready", 
                                    font=("Arial", 10), bg="#ecf0f1", fg="#27ae60")
        self.status_label.pack(pady=20)
        
    def register_student(self):
        """Register a new student"""
        student_id = simpledialog.askstring("Input", "Enter Student ID:")
        if not student_id:
            return
            
        student_name = simpledialog.askstring("Input", "Enter Student Name:")
        if not student_name:
            return
            
        num_samples = simpledialog.askinteger("Input", "Number of samples to capture:", 
                                             initialvalue=50, minvalue=10, maxvalue=200)
        if not num_samples:
            return
            
        self.status_label.config(text="Capturing faces... Look at camera!", fg="#e74c3c")
        self.root.update()
        
        count = self.system.capture_faces(student_id, student_name, num_samples)
        
        messagebox.showinfo("Success", 
                          f"Captured {count} images for {student_name}!\n"
                          "Please train the model to recognize this student.")
        self.status_label.config(text="Registration complete", fg="#27ae60")
    
    def train_model_thread(self):
        """Train model in separate thread"""
        def train():
            self.status_label.config(text="Training model... Please wait", fg="#e67e22")
            self.root.update()
            
            success = self.system.train_model(epochs=20)
            
            if success:
                messagebox.showinfo("Success", "Model trained successfully!")
                self.status_label.config(text="Model ready", fg="#27ae60")
            else:
                messagebox.showerror("Error", "Training failed! Check dataset.")
                self.status_label.config(text="Training failed", fg="#e74c3c")
        
        threading.Thread(target=train, daemon=True).start()
    
    def start_attendance_thread(self):
        """Start attendance in separate thread"""
        def mark():
            self.status_label.config(text="Attendance system running...", fg="#27ae60")
            self.system.start_attendance(callback=self.update_status)
            self.status_label.config(text="Attendance system stopped", fg="#7f8c8d")
        
        threading.Thread(target=mark, daemon=True).start()
    
    def update_status(self, message):
        """Update status label from thread"""
        self.status_label.config(text=message)
        self.root.update()
    
    def view_attendance(self):
        """View attendance records"""
        today = datetime.now().strftime("%Y-%m-%d")
        csv_file = os.path.join(self.system.attendance_path, f"attendance_{today}.csv")
        
        if not os.path.exists(csv_file):
            messagebox.showinfo("Info", "No attendance records for today!")
            return
            
        # Create new window
        view_window = tk.Toplevel(self.root)
        view_window.title(f"Attendance - {today}")
        view_window.geometry("600x400")
        
        # Treeview
        tree = ttk.Treeview(view_window, columns=("ID", "Name", "Date", "Time"), 
                           show="headings")
        tree.heading("ID", text="ID")
        tree.heading("Name", text="Name")
        tree.heading("Date", text="Date")
        tree.heading("Time", text="Time")
        
        # Load data
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            tree.insert("", "end", values=(row['ID'], row['Name'], row['Date'], row['Time']))
        
        tree.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Export button
        tk.Button(view_window, text="Export to Excel", 
                 command=lambda: self.export_to_excel(df)).pack(pady=10)
    
    def export_to_excel(self, df):
        """Export attendance to Excel"""
        filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(filename, index=False)
        messagebox.showinfo("Success", f"Exported to {filename}")
    
    def show_settings(self):
        """Show settings window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        
        tk.Label(settings_window, text="Confidence Threshold:", 
                font=("Arial", 12)).pack(pady=10)
        
        threshold_var = tk.DoubleVar(value=self.system.confidence_threshold)
        threshold_scale = tk.Scale(settings_window, from_=0.5, to=1.0, 
                                  resolution=0.05, orient="horizontal",
                                  variable=threshold_var, length=300)
        threshold_scale.pack(pady=10)
        
        def save_settings():
            self.system.confidence_threshold = threshold_var.get()
            messagebox.showinfo("Success", "Settings saved!")
            settings_window.destroy()
        
        tk.Button(settings_window, text="Save", command=save_settings).pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceGUI(root)
    root.mainloop()