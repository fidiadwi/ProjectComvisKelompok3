# ProjectComvisKelompok3
Deteksi Huruf Kaganga Aksara Ulu Rejang menggunakan YOLOv8

Anggota:
1. G1A023040 - Fidia Dewi Wulandari Batu Bara
2. G1A023052 - Anisa Julianti
3. G1A023080 - Khalisa Rizgita Amanda+

# Deteksi Huruf Kaganga Aksara Ulu Rejang Menggunakan YOLOv8

## Cell 1 - Install Library

```python
!pip install ultralytics roboflow
```

Menginstal library yang diperlukan untuk proses pengolahan dataset dan pelatihan model YOLOv8.

- **ultralytics** : Library resmi YOLOv8.
- **roboflow** : Digunakan untuk mengakses dan mengunduh dataset dari Roboflow.

---

## Cell 2 - Import Library

```python
from roboflow import Roboflow
from ultralytics import YOLO

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
```

Mengimpor library yang digunakan selama proses pelatihan, prediksi, dan visualisasi hasil.

---

## Cell 3 - Download Dataset

```python
rf = Roboflow(api_key="YOUR_API_KEY")

project = rf.workspace(
    "novalrizkiansyah-ymail-com"
).project(
    "aksara-ulu-rejang"
)

version = project.version(4)

dataset = version.download("yolov8")
```

Menghubungkan notebook dengan Roboflow dan mengunduh dataset Aksara Ulu Rejang versi 4 dalam format YOLOv8.

---

## Cell 4 - Hapus Hasil Training Sebelumnya

```python
!rm -rf /content/runs/detect/train
!rm -rf /content/runs/detect/predict
```

Menghapus folder hasil training dan prediksi sebelumnya agar tidak tercampur dengan hasil eksperimen yang baru.

---

## Cell 5 - Cek Struktur Dataset

```python
print(os.listdir("/content/Aksara-Ulu-Rejang-4"))
```

Menampilkan isi folder dataset untuk memastikan dataset berhasil diunduh dan memiliki struktur yang benar.

---

## Cell 6 - Cek File data.yaml

```python
yaml_path = "/content/Aksara-Ulu-Rejang-4/data.yaml"

with open(yaml_path, "r") as f:
    print(f.read())
```

Menampilkan isi file `data.yaml` yang berisi konfigurasi dataset seperti lokasi data train, validation, test, jumlah kelas, dan nama kelas.

---

## Cell 7 - Training YOLOv8

```python
model = YOLO("yolov8m.pt")

results = model.train(
    data="/content/Aksara-Ulu-Rejang-4/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    plots=True,
    save=True
)
```

Melatih model YOLOv8 menggunakan dataset Aksara Ulu Rejang.

### Parameter Training

| Parameter | Nilai | Keterangan |
|------------|--------|------------|
| epochs | 50 | Jumlah iterasi pelatihan |
| imgsz | 640 | Ukuran gambar input |
| batch | 16 | Jumlah data per batch |
| plots | True | Menyimpan grafik hasil training |
| save | True | Menyimpan model hasil training |

Model dasar yang digunakan adalah **YOLOv8m (Medium)**.

---

## Cell 8 - Cek Hasil Training

```python
train_path = "/content/runs/detect/train"

print(os.listdir(train_path))
```

Menampilkan seluruh file yang dihasilkan selama proses training seperti grafik evaluasi, confusion matrix, dan folder weights.

---

## Cell 9 - Menampilkan Grafik Training

```python
png_files = [
    f for f in os.listdir(train_path)
    if f.endswith(".png")
]
```

Mengambil seluruh file gambar hasil training dan menampilkannya menggunakan Matplotlib.

Grafik yang biasanya dihasilkan meliputi:

- Results Curve
- F1 Curve
- Precision Curve
- Recall Curve
- Precision-Recall Curve
- Confusion Matrix

---

## Cell 10 - Load Model Terbaik

```python
best_model = YOLO(
    "/content/runs/detect/train/weights/best.pt"
)
```

Memuat model terbaik yang dihasilkan selama proses training untuk digunakan pada tahap pengujian atau prediksi.

---

## Cell 11 - Prediksi Gambar

```python
results = best_model.predict(
    source="/content/dataset/train/images",
    save=True,
    conf=0.01,
    line_width=2,
    show_labels=True,
    show_conf=True
)
```

Melakukan deteksi huruf Kaganga pada gambar menggunakan model yang telah dilatih.

### Parameter Prediksi

| Parameter | Fungsi |
|------------|---------|
| source | Lokasi gambar yang akan diuji |
| save | Menyimpan hasil prediksi |
| conf | Confidence threshold |
| line_width | Ketebalan bounding box |
| show_labels | Menampilkan label kelas |
| show_conf | Menampilkan nilai confidence |

Hasil prediksi disimpan pada folder `runs/detect/predict`.

---

## Cell 12 - Cek Folder Predict

```python
predict_folder = "/content/runs/detect/predict"

print(os.listdir(predict_folder))
```

Menampilkan daftar file hasil prediksi yang telah disimpan.

---

## Cell 13 - Menampilkan Hasil Deteksi

```python
index = 57
```

Memilih salah satu gambar hasil prediksi berdasarkan indeks tertentu, kemudian menampilkannya menggunakan Matplotlib.

Output yang ditampilkan meliputi:

- Bounding box objek
- Label huruf Kaganga
- Nilai confidence deteksi

---

## Cell 14 - Export Model

```python
best_model.export(format="onnx")
```

Mengonversi model YOLOv8 ke format ONNX agar dapat digunakan pada berbagai platform deployment seperti desktop, web, mobile, maupun edge device.

---

## Menampilkan Data Uji

```python
test_path = "/content/Aksara-Ulu-Rejang-4/test/images"

files = os.listdir(test_path)

for i, file in enumerate(files[:30]):
    print(i, file)
```

Menampilkan 30 file pertama pada folder test untuk memastikan data pengujian tersedia dan dapat digunakan pada tahap evaluasi model.

---

## Output Program

Program menghasilkan beberapa output utama:

- Model terbaik (`best.pt`)
- Model ONNX (`best.onnx`)
- Grafik hasil training
- Confusion Matrix
- Precision Curve
- Recall Curve
- F1 Curve
- Hasil deteksi huruf Kaganga pada gambar uji

Model yang dihasilkan dapat digunakan untuk mendeteksi huruf-huruf Kaganga Aksara Ulu Rejang secara otomatis menggunakan pendekatan Object Detection berbasis YOLOv8.
