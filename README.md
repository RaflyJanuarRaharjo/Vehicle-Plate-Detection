# Real-Time Vehicle License Plate Detection and Character Recognition

This project is a deep learning implementation for detecting and recognizing vehicle license plates. It utilizes **YOLOv8** for object detection and **EasyOCR** for Optical Character Recognition, optimized with custom preprocessing techniques to handle Indonesian license plate formats.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-yellow)

## 📋 Key Features

* **High-Accuracy Detection:** Uses a custom-trained YOLOv8 Nano model specifically tuned for license plates.
* **Advanced Preprocessing:** Implements Grayscale conversion and Otsu's Thresholding to enhance text visibility before OCR processing.
* **Smart Padding:** Automatically expands the cropped area to ensure edge characters (like region codes) are not cut off.
* **Intelligent Filtering:** logic to filter out non-license plate text (e.g., removing expiry dates like "11.28") and keeping only the registration number.
* **Visual Output:** Generates images with bounding boxes, confidence scores, and recognized text.

## 🛠️ Tech Stack

* **Language:** Python 3
* **Object Detection:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* **OCR Engine:** [EasyOCR](https://github.com/JaidedAI/EasyOCR)
* **Image Processing:** OpenCV (cv2) & NumPy
* **Training Environment:** Google Colab (T4 GPU)

## 📂 Project Structure

```text
Vehicle-Plate-Detection/
│
├── models/
│   └── best.pt             # Custom trained YOLOv8 model weights
├── images/                 # Folder for test images
├── main.py                 # Main application script (Detection + OCR pipeline)
├── requirements.txt        # Python dependencies
├── debug_potongan_plat.jpg # (Output) Cropped plate for debugging
├── hasil_akhir_deteksi.jpg # (Output) Final result with bounding boxes
└── README.md               # Project documentation
