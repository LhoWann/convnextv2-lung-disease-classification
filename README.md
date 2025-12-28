
# LungScan AI: Deteksi Penyakit Paru Menggunakan ConvNeXt V2

Proyek ini adalah sistem *Computer-Aided Diagnosis* (CAD) berbasis *Deep Learning* untuk mengklasifikasikan citra X-Ray dada ke dalam 4 kategori:  **Normal** ,  **Pneumonia** ,  **Tuberculosis (TBC)** , dan  **Unknown** .

Sistem dibangun menggunakan arsitektur **ConvNeXt V2 (Tiny)** dengan framework  **PyTorch Lightning** , dilengkapi dengan strategi penanganan data tidak seimbang ( *imbalanced data* ) dan visualisasi *Explainable AI* (Grad-CAM).

## Fitur Utama

* **State-of-the-Art Model** : Menggunakan ConvNeXt V2 Tiny pre-trained.
* **Robust Training Strategy** : Menggunakan *Class-Weighted Loss* dan *Label Smoothing* untuk menangani dataset yang tidak seimbang.
* **Explainable AI (XAI)** : Integrasi **Grad-CAM** untuk memvisualisasikan area fokus model (heatmap) pada citra paru.
* **Interaktif Web App** : Antarmuka pengguna berbasis **Streamlit** dengan tema  *Dark Mode* , dukungan upload gambar/URL, dan visualisasi  *confidence score* .
* **Scalable** : Mendukung pelatihan *Single GPU* maupun *Multi-GPU* (DDP) secara otomatis.

## Struktur Proyek

```
.
├── data               # Dataset: train, val, test
├── model.py           # Arsitektur LungDiseaseModel (LightningModule)
├── utils.py           # DataModule, Augmentasi, dan fungsi Utilitas
├── preprocess.py      # Preprocess dari pretrained
├── train.py           # Skrip utama pelatihan (Training loop)
├── gradcam.py         # Skrip untuk generate visualisasi Grad-CAM manual
├── app.py             # Aplikasi Web Streamlit (Deployment)
├── requirements.txt   # Daftar dependensi library
└── README.md          # Dokumentasi proyek

```

## Instalasi

1. Pastikan Python 3.8+ terinstal.
2. Disarankan menggunakan  *Virtual Environment* .
3. Install dependensi yang diperlukan:

```
pip install -r requirements.txt
pip install streamlit grad-cam opencv-python altair

```

## Panduan Penggunaan

### 1. Pelatihan Model (Training)

Jalankan `train.py` untuk melatih model dari awal. Parameter default sudah dioptimalkan untuk dataset medis (~13k gambar).

**Perintah Standar (Single GPU - RTX 3050/T4):**

```
python train.py --data_dir "path/ke/dataset" --epochs 15 --batch_size 16 --lr 2e-5

```

**Argumen Penting:**

* `--data_dir`: Path ke folder dataset (harus berisi subfolder `train`, `val`, `test`).
* `--output_dir`: Lokasi simpan checkpoint model (Default: `./output`).
* `--epochs`: Jumlah epoch (Rekomendasi: 15-20).
* `--batch_size`: Ukuran batch (Sesuaikan VRAM, rekomendasi 16 untuk 4GB VRAM).
* `--lr`: Learning Rate (Rekomendasi: `2e-5` untuk stabilitas).

Hasil Output:

File checkpoint terbaik (best-checkpoint.ckpt) akan tersimpan di folder output/.

### 2. Visualisasi Grad-CAM (Manual)

Untuk melihat heatmap fokus model pada satu gambar tertentu melalui terminal:

```
python gradcam.py --image_path "data/test/TBC/pasien_01.jpg" --checkpoint_path "output/best-checkpoint.ckpt"

```

Hasil visualisasi akan disimpan sebagai `gradcam_result.jpg`.

### 3. Menjalankan Aplikasi Web (Streamlit)

Untuk menjalankan antarmuka grafis (GUI) di browser:

1. Pastikan file checkpoint ada di `output/best-checkpoint.ckpt` (atau sesuaikan path di `app.py`).
2. Jalankan perintah:

```
streamlit run app.py

```

3. Buka browser di alamat yang muncul (biasanya `http://localhost:8501`).

**Fitur Aplikasi:**

* **Dual Input:** Upload file lokal atau tempel link URL gambar.
* **AI Analysis:** Prediksi penyakit beserta tingkat keyakinan ( *confidence score* ).
* **Explainability:** Toggle "Explain AI Decision" untuk melihat Grad-CAM overlay.

## Konfigurasi Hyperparameter

Penentuan *Learning Rate* (LR) mengikuti **Linear Scaling Rule** sesuai paper ConvNeXt V2 untuk menjaga stabilitas pelatihan pada *batch size* kecil.

Rumus yang digunakan:


$$
lr_{new} = lr_{base} \times \frac{batch_size_{new}}{batch_size_{base}}
$$

Dimana:

- $lr_{base} = 8 \times 10^{-4}$ (Base LR dari paper)
- $batch\_size_{base} = 1024$ (Base Batch dari paper)
- $batch_size_{new} = 16$ (Disesuaikan dengan limitasi VRAM)

Perhitungan:

$$
lr_{new} = 0.0008 \times \frac{16}{1024} = 1.25 \times 10^{-5}
$$

Dalam implementasi ini, nilai dibulatkan menjadi $2 \times 10^{-5}$ **(`2e-5`)** untuk mempercepat konvergensi pada dataset berukuran 13.000 citra tanpa merusak bobot  *pre-trained* .

## Performa Model

Pada eksperimen menggunakan dataset X-Ray paru (~13.000 citra), model ini mencapai performa:

* **Akurasi Keseluruhan:** ~98.5% - 99.0%
* **Sensitivitas TBC:** Sangat tinggi (>98%) berkat strategi augmentasi spasial yang presisi.
* **Deteksi Unknown:** 100% (Mampu membedakan X-Ray valid vs input non-medis).

## Catatan Penting

* **Augmentasi:** Menggunakan *Simple Augmentation* (Resize, Flip, Rotate 10°) untuk menjaga integritas fitur patologis (seperti kavitas TBC di apeks paru).
* **Disclaimer:** Sistem ini adalah alat bantu keputusan ( *Clinical Decision Support* ) dan bukan pengganti diagnosis dokter ahli.

*Dikembangkan untuk Tugas Akhir/Penelitian Infrastruktur Sains Data.*

