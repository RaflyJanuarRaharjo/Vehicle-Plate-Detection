import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

def detect_and_read_plate(image_path):
    print(f"🚀 Memulai proses untuk: {image_path}")
    
    # 1. LOAD MODEL & OCR
    # Pastikan file best.pt ada di folder models/
    try:
        model = YOLO('models/best.pt')
        reader = easyocr.Reader(['en']) # Bahasa Inggris cukup untuk plat nomor
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. BACA GAMBAR
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Gambar tidak ditemukan. Periksa nama filenya!")
        return
    
    height, width, _ = img.shape

    # 3. DETEKSI PLAT DENGAN YOLO
    results = model(img)
    
    found_plate = False

    for result in results:
        for box in result.boxes:
            found_plate = True
            
            # --- A. KOORDINAT DASAR ---
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            print(f"\n🔍 Plat Terdeteksi (Yakin: {confidence*100:.1f}%)")

            # --- B. TAMBAH PADDING (PENTING!) ---
            # Memperluas kotak sedikit agar huruf pinggir (misal 'H') tidak terpotong
            padding_x = int((x2 - x1) * 0.05) # Tambah lebar 5%
            padding_y = int((y2 - y1) * 0.10) # Tambah tinggi 10%

            # Pastikan tidak keluar batas gambar
            crop_x1 = max(0, x1 - padding_x)
            crop_y1 = max(0, y1 - padding_y)
            crop_x2 = min(width, x2 + padding_x)
            crop_y2 = min(height, y2 + padding_y)

            # Potong Gambar (Cropping)
            plate_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

            # Simpan potongan untuk Debugging (Cek apakah huruf H terpotong?)
            cv2.imwrite("debug_potongan_plat.jpg", plate_img)

            # --- C. PRE-PROCESSING (Bikin Hitam Putih) ---
            # Mengubah ke abu-abu agar OCR lebih fokus
            gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            
            # Opsional: Meningkatkan kontras (Thresholding)
            # _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # --- D. BACA TEKS (OCR) ---
            raw_results = reader.readtext(gray_plate, detail=0)
            print(f"   📝 Mentah dari OCR: {raw_results}")

            # --- E. FILTER HASIL (Membuang 11.28) ---
            final_text_parts = []
            for text in raw_results:
                # Hapus spasi aneh
                text_clean = text.strip().upper()
                
                # Logika Filter:
                # 1. Abaikan jika mengandung titik '.' (biasanya tanggal masa berlaku 11.28)
                # 2. Abaikan jika terlalu pendek (< 1 karakter)
                if "." not in text_clean and len(text_clean) > 0:
                    final_text_parts.append(text_clean)
            
            # Gabungkan sisa teks
            final_plate_text = " ".join(final_text_parts)
            print(f"   ✅ Hasil Akhir: {final_plate_text}")

            # --- F. GAMBAR DI LAYAR ---
            # Gambar kotak hijau di foto asli
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Tulis teks di atas kotak
            label = f"{final_plate_text} ({confidence*100:.0f}%)"
            cv2.putText(img, label, (x1, y1 - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 4. SIMPAN HASIL
    if found_plate:
        output_filename = "hasil_akhir_deteksi.jpg"
        cv2.imwrite(output_filename, img)
        print(f"\n🎉 SUKSES! Gambar hasil disimpan sebagai: {output_filename}")
        print("👉 Cek juga file 'debug_potongan_plat.jpg' untuk melihat potongan platnya.")
        
        # Tampilkan Window (Tekan spasi untuk tutup)
        # cv2.imshow("Hasil Deteksi", img) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("⚠️ Tidak ada plat nomor yang terdeteksi.")

if __name__ == "__main__":
    # GANTI nama file ini sesuai gambar Anda
    image_file = "test.jpg" 
    detect_and_read_plate(image_file)