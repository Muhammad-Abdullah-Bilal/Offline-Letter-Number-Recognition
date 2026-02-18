# Offline-Letter-Number-Recognition

An end-to-end **Machine Learning desktop application** that recognizes **handwritten letters and numbers** using a **Convolutional Neural Network (CNN)**.  
The system works **completely offline** and provides **real-time predictions** through an interactive GUI.

---

## 📌 Project Overview

This project allows users to draw **uppercase letters, lowercase letters, or digits** using a mouse or touchscreen.  
The drawn character is preprocessed and passed to a trained CNN model, which predicts the character and displays the result instantly.

---

## ✨ Features

- 🖊️ Draw handwritten characters on a canvas
- 🔠 Recognizes **A–Z, a–z, and 0–9**
- 🧠 CNN-based deep learning model
- ⚡ Real-time prediction with confidence score
- 💻 Fully **offline desktop application**
- 🎤 Offline voice output announcing the recognized character
- ✏️ Pen and Eraser support (extra feature)

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** TensorFlow / Keras  
- **Model:** Convolutional Neural Network (CNN)  
- **Dataset:** EMNIST Balanced  
- **GUI:** Tkinter  
- **Image Processing:** NumPy, Pillow (PIL)  
- **Text-to-Speech:** pyttsx3  

---

## 🧠 How It Works

1. User draws a character on the canvas  
2. Image is preprocessed (cropping, centering, resizing, normalization)  
3. Preprocessed image is passed to the trained CNN model  
4. Model predicts the character  
5. Result is displayed on the screen and spoken aloud  

---

## 📂 Project Structure

offline-handwriting-recognition/
│
├── train.py # Model training script
├── app.py # Desktop application
├── char_recognition_model.h5 # Trained CNN model
├── requirements.txt # Dependencies
├── README.md # Project documentation

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/offline-handwriting-recognition.git
cd offline-handwriting-recognition
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Application
python app.py
🏋️ Model Training (Optional)
To retrain the model from scratch:

python train.py
Note: Training may take time depending on system performance.

🎯 Learning Outcomes
Built an end-to-end ML pipeline

Hands-on experience with CNNs for image recognition

Practical deployment of ML models in desktop applications

Improved understanding of image preprocessing techniques

Experience with offline AI systems

📌 Future Improvements
Support for more handwriting styles

Accuracy improvements with data augmentation

Export model to lighter formats for faster inference

Cross-platform executable packaging

📜 License
This project is for educational and learning purposes.
