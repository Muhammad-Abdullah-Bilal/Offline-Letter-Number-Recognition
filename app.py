import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import tensorflow as tf
import pyttsx3  # Purely offline Text-to-Speech

# Mapping for EMNIST Balanced (47 classes)
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnrt"

class ProfessionalRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Handwriting Recognition Pro")
        self.root.geometry("450x650")
        self.root.configure(bg="#2c3e50")

        # 1. Load Model
        try:
            self.model = tf.keras.models.load_model('char_recognition_model.h5')
        except Exception as e:
            print(f"Model Error: {e}")

        # 2. Initialize Offline Voice Engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150) # Set speaking speed

        # 3. UI Styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Helvetica", 10, "bold"), padding=10)

        # Header
        tk.Label(root, text="Handwriting Recognition", font=("Helvetica", 18, "bold"), 
                 bg="#2c3e50", fg="#ecf0f1").pack(pady=20)

        # Canvas Setup (Black background as per EMNIST training)
        self.canvas_frame = tk.Frame(root, bg="#bdc3c7", padx=3, pady=3)
        self.canvas_frame.pack()

        self.canvas = tk.Canvas(self.canvas_frame, width=300, height=300, bg='black', 
                               highlightthickness=0, cursor="pencil")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        # Internal Image for Processing
        self.image = Image.new("L", (300, 300), 0)
        self.draw_obj = ImageDraw.Draw(self.image)
        
        self.brush_color = "white"
        self.brush_size = 12

        # 4. Result Display
        self.result_frame = tk.Frame(root, bg="#2c3e50")
        self.result_frame.pack(pady=15)
        
        self.res_val = tk.Label(self.result_frame, text="READY", font=("Helvetica", 32, "bold"), 
                               bg="#2c3e50", fg="#2ecc71")
        self.res_val.pack()
        
        self.confidence_label = tk.Label(self.result_frame, text="Confidence: 0%", font=("Helvetica", 11), 
                                        bg="#2c3e50", fg="#bdc3c7")
        self.confidence_label.pack()

        # 5. Controls
        self.btn_frame = tk.Frame(root, bg="#2c3e50")
        self.btn_frame.pack(pady=10)

        # Primary Actions
        ttk.Button(self.btn_frame, text="RECOGNIZE", command=self.smart_predict).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.btn_frame, text="CLEAR", command=self.clear).grid(row=0, column=1, padx=5, pady=5)

        # Secondary Actions (Eraser/Pen - Extra Credit)
        tool_frame = tk.Frame(root, bg="#2c3e50")
        tool_frame.pack()
        ttk.Button(tool_frame, text="Pen", command=self.use_pen).grid(row=0, column=0, padx=5)
        ttk.Button(tool_frame, text="Eraser", command=self.use_eraser).grid(row=0, column=1, padx=5)

    def use_pen(self):
        self.brush_color = "white"
        self.brush_size = 12
        self.canvas.config(cursor="pencil")

    def use_eraser(self):
        self.brush_color = "black"
        self.brush_size = 25
        self.canvas.config(cursor="dot")

    def paint(self, event):
        x, y = event.x, event.y
        r = self.brush_size
        # Draw on UI
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=self.brush_color, outline=self.brush_color)
        # Draw on Internal PIL Image
        val = 255 if self.brush_color == "white" else 0
        self.draw_obj.ellipse([x-r, y-r, x+r, y+r], fill=val)

    def smart_preprocess(self, pil_image):
        """Finds the character, centers it, and resizes to 28x28."""
        # Find drawing boundaries
        diff = ImageChops.difference(pil_image, Image.new("L", (300, 300), 0))
        bbox = diff.getbbox()
        
        if not bbox: return None
            
        # Crop and pad to make it a square
        char_crop = pil_image.crop(bbox)
        w, h = char_crop.size
        max_dim = max(w, h) + 40 
        new_img = Image.new("L", (max_dim, max_dim), 0)
        new_img.paste(char_crop, ((max_dim - w) // 2, (max_dim - h) // 2))
        
        # Resize to model input size
        return new_img.resize((28, 28), Image.LANCZOS)

    def smart_predict(self):
        processed_img = self.smart_preprocess(self.image)
        
        if processed_img is None:
            self.res_val.config(text="EMPTY", fg="#e74c3c")
            return

        # Prepare for model (Normalize)
        img_array = np.array(processed_img).reshape(1, 28, 28, 1) / 255.0
        
        # Inference
        predictions = self.model.predict(img_array)
        best_match_idx = np.argmax(predictions)
        confidence = predictions[0][best_match_idx] * 100
        result = CLASSES[best_match_idx]
        
        # Update UI
        self.res_val.config(text=result, fg="#2ecc71")
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Offline Voice Output
        self.speak(result)

    def speak(self, text):
        try:
            self.engine.say(f"The character is {text}")
            self.engine.runAndWait()
        except: pass

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (300, 300), 0)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.res_val.config(text="READY", fg="#2ecc71")
        self.confidence_label.config(text="Confidence: 0%")

if __name__ == "__main__":
    root = tk.Tk()
    app = ProfessionalRecognitionApp(root)
    root.mainloop()