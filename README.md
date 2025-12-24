# convnextv2-lung-disease-classification

```
# Klasifikasi Penyakit Paru-paru Menggunakan ConvNeXt V2

Proyek ini berisi implementasi pipeline Deep Learning untuk klasifikasi penyakit paru-paru (seperti Pneumonia, Tuberculosis, dan Normal) menggunakan arsitektur ConvNeXt V2. Kode dibangun di atas kerangka kerja PyTorch Lightning untuk strukturisasi pelatihan yang efisien dan skalabel.

## Fitur Utama

* **Model Backbone**: ConvNeXt V2 (Tiny) pre-trained dari HuggingFace Transformers.
* **Training Framework**: PyTorch Lightning.
* **Logging**: Mendukung logging metrik (Loss, Accuracy, F1-Score) dan penyimpanan checkpoint otomatis.
* **Fleksibilitas**: Mendukung konfigurasi direktori data eksternal dan pemilihan GPU spesifik via command line.
* **Evaluasi**: Menyediakan Classification Report dan Confusion Matrix setelah pelatihan selesai.

## Struktur Direktori

Pastikan struktur proyek Anda terlihat seperti berikut:

```text
.
├── model.py           # Definisi arsitektur model (LightningModule)
├── utils.py           # DataModule dan fungsi utilitas (plotting, metrics)
├── train.py           # Skrip utama untuk training dan evaluasi
├── requirements.txt   # Daftar dependensi
├── .gitignore         # Konfigurasi git ignore
└── README.md          # Dokumentasi proyek
```

## Persiapan Dataset

Skrip ini menggunakan `ImageFolder` standar dari Torchvision. Dataset harus diatur dengan struktur berikut:

**Plaintext**

```
/path/ke/dataset/
├── train/
│   ├── class_A/
│   ├── class_B/
│   └── ...
├── val/
│   ├── class_A/
│   ├── class_B/
│   └── ...
└── test/
    ├── class_A/
    ├── class_B/
    └── ...
```

## Instalasi

1. Disarankan menggunakan virtual environment (conda/venv).
2. Install dependensi yang diperlukan:

**Bash**

```
pip install -r requirements.txt
```

## Penggunaan

Skrip `train.py` dijalankan melalui terminal. Argumen `--data_dir` wajib diisi.

### Argumen Command Line

| **Argumen** | **Tipe** | **Default** | **Deskripsi**                          |
| ----------------- | -------------- | ----------------- | -------------------------------------------- |
| `--data_dir`    | str            | (Wajib)           | Path lengkap menuju root folder dataset.     |
| `--output_dir`  | str            | `./output`      | Lokasi penyimpanan model checkpoint dan log. |
| `--gpu_id`      | int            | `0`             | ID GPU yang akan digunakan.                  |
| `--epochs`      | int            | `10`            | Jumlah epoch pelatihan.                      |
| `--batch_size`  | int            | `32`            | Ukuran batch per iterasi.                    |
| `--lr`          | float          | `5e-5`          | Learning rate awal.                          |
| `--num_workers` | int            | `4`             | Jumlah worker untuk DataLoader.              |

### Contoh Eksekusi

#### 1. Penggunaan Dasar (Single GPU Default)

Gunakan perintah ini jika dataset berada di dalam folder proyek dan Anda menggunakan GPU utama (ID 0).

**Bash**

```
python train.py --data_dir ./data/lung_dataset
```

#### 2. Menggunakan Direktori Data Berbeda (Eksternal)

Kasus ini digunakan jika dataset tersimpan di drive lain (misalnya hard disk eksternal atau partisi dataset terpisah) untuk menghemat ruang disk proyek.

**Format Linux/WSL:**

**Bash**

```
python train.py --data_dir /mnt/d/Datasets/Medical/Lung_Xray --output_dir ./experiment_1
```

**Format Windows:**

**Bash**

```
python train.py --data_dir "D:\Datasets\Medical\Lung_Xray" --output_dir ".\experiment_1"
```

#### 3. Memilih GPU Spesifik (Single GPU pada Server Multi-GPU)

Jika Anda bekerja pada server yang memiliki banyak GPU (misal: GPU 0, 1, 2, 3) dan ingin menjalankan pelatihan hanya pada **GPU ke-2** (ID 1), gunakan argumen `--gpu_id`.

**Bash**

```
python train.py --data_dir ./data --gpu_id 1
```

#### 4. Konfigurasi Hiperparameter Lengkap

Contoh untuk mengubah ukuran batch, learning rate, dan jumlah epoch secara bersamaan.

**Bash**

```
python train.py --data_dir ./data --batch_size 64 --lr 1e-4 --epochs 20 --gpu_id 0
```

## Hasil Keluaran

Setelah eksekusi selesai, skrip akan menghasilkan:

1. **Checkpoint Model** : File `.ckpt` dengan nilai validasi F1-Score terbaik akan disimpan di folder yang ditentukan oleh `--output_dir`.
2. **Metrik Evaluasi** : Laporan klasifikasi (Precision, Recall, F1-Score) akan dicetak di terminal.
3. **Visualisasi** : Confusion Matrix akan ditampilkan sebagai plot gambar.

## Catatan Teknis

* Model menggunakan `CosineAnnealingLR` untuk penjadwalan learning rate.
* Training menggunakan *Mixed Precision* (16-mixed) secara default untuk efisiensi memori GPU.
* Class weights dihitung secara otomatis untuk menangani ketidakseimbangan data (imbalanced dataset).
