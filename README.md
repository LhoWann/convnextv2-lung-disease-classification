# convnextv2-lung-disease-classification

Klasifikasi penyakit paru-paru menggunakan arsitektur ConvNeXt V2 dan PyTorch Lightning.

## Ringkasan

- Model backbone: ConvNeXt V2 (Tiny) pre-trained dari HuggingFace.
- Framework: PyTorch Lightning.
- Fitur: logging metrik, checkpoint otomatis, evaluasi (classification report & confusion matrix), dukungan GPU dan mixed precision.

## Struktur proyek

```
.
├── model.py         # LightningModule (model & training/validation step)
├── utils.py         # DataModule, utilitas (plot, metrics)
├── train.py         # Skrip utama untuk training dan evaluasi
├── requirements.txt # Dependensi
└── README.md        # Dokumentasi
```

## Persiapan dataset

Gunakan struktur standar `ImageFolder`:

```
/path/ke/dataset/
├── train/
│   ├── class_A/
│   └── class_B/
├── val/
│   ├── class_A/
│   └── class_B/
└── test/
    ├── class_A/
    └── class_B/
```

## Instalasi

Disarankan menggunakan virtual environment. Install dependensi:

```bash
pip install -r requirements.txt
```

## Penggunaan

Jalankan `train.py`. Argumen penting:

- `--data_dir` (str, wajib): path ke root dataset.
- `--output_dir` (str, default `./output`): tempat menyimpan checkpoint dan log.
- `--devices` (Int, default `1`): Jumlah GPU yang ingin digunakan.
- `--epochs` (int, default `10`): jumlah epoch.
- `--batch_size` (int, default `32`): ukuran batch.
- `--lr` (float, default `5e-5`): learning rate awal.
- `--num_workers` (int, default `4`): jumlah worker DataLoader.

Contoh dasar:

```bash
python train.py --data_dir ./data/lung_dataset
```

Contoh (Windows, direktori eksternal):

```bash
python train.py --data_dir "D:\Datasets\Medical\Lung_Xray" --output_dir .\experiment_1
```

Memilih GPU tertentu:

```bash
python train.py --data_dir --data_dir --devices 2 --batch_size 64 --num_workers 4
```
Menyesuaikan Parameter:
```bash
!python /kaggle/working/convnextv2-lung-disease-classification/train.py \
  --data_dir /kaggle/input/combined-unknown-pneumonia-and-tuberculosis/data \
  --devices 2 \
  --batch_size 128 \
  --epochs 10 \
  --num_workers 4
```

## Output

- Checkpoint terbaik (`.ckpt`) disimpan di `--output_dir`.
- Laporan klasifikasi (Precision, Recall, F1) dicetak di terminal.
- Confusion matrix ditampilkan / disimpan sesuai implementasi di `utils.py`.

## Catatan teknis

- Scheduler: `CosineAnnealingLR`.
- Mixed precision (16-bit) digunakan untuk efisiensi memori.
- Class weights dihitung otomatis untuk menangani ketidakseimbangan kelas.

