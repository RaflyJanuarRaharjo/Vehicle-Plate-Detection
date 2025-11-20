import cv2
import easyocr
from ultralytics import YOLO

# KITA AKAN PAKAI MODEL HASIL TRAINING NANTI
# Untuk sementara pakai model standar dulu biar tidak error saat upload
model = YOLO('yolov8n.pt') 
reader = easyocr.Reader(['en'])

def detect_plate(img_path):
    img = cv2.imread(img_path)
    results = model(img)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = img[y1:y2, x1:x2]
            text = reader.readtext(plate_img, detail=0)
            print(f"Plat Terbaca: {text}")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Hasil", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    # Pastikan ada file gambar test.jpg di folder yang sama nanti
    detect_plate("test.jpg")