"""
train.py — Uçtan Uca Eğitim Scripti
=====================================
Proje  : Eğitim Teknolojileri için PyTorch Eğitimi
Yazar  : Ugur Sirvermez — Bursa Uludag Universitesi
Lisans : CC BY-NC-SA 4.0

Bu script, Module_Files içindeki tüm bileşenleri bir araya getirerek
FoodVision Mini modelini eğitir ve kaydeder.

Kullanım:
    python Module_Files/train.py

Çıktı:
    models/tinyvgg_modular.pth — eğitilmiş modelin ağırlıkları

Hiperparametreler (aşağıda düzenlenebilir):
    NUM_EPOCHS    : Epoch sayısı
    BATCH_SIZE    : Mini-batch boyutu
    HIDDEN_UNITS  : Konvolüsyon filtre sayısı
    LEARNING_RATE : Adam optimizer öğrenme hızı
"""

import torch
from torchvision import transforms

import data_setup, engine, model_builder, utils


# ─────────────────────────────────────────────
# Hiperparametreler
# ─────────────────────────────────────────────
NUM_EPOCHS    = 5
BATCH_SIZE    = 32
HIDDEN_UNITS  = 10
LEARNING_RATE = 0.001


# ─────────────────────────────────────────────
# Veri Yolları
# ─────────────────────────────────────────────
train_dir = "data/pizza_steak_sushi/train"
test_dir  = "data/pizza_steak_sushi/test"


# ─────────────────────────────────────────────
# Cihaz Seçimi
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[train] Kullanılan cihaz: {device}")


# ─────────────────────────────────────────────
# Görüntü Dönüşümü
# ─────────────────────────────────────────────
# Görüntüleri 64×64 piksel boyutuna getirip tensöre çeviriyoruz.
veri_donusumu = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# ─────────────────────────────────────────────
# DataLoader'lar (data_setup.py)
# ─────────────────────────────────────────────
# create_dataloaders → (train_dl, test_dl, sinif_adlari) döndürür
train_dataloader, test_dataloader, sinif_adlari = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=veri_donusumu,
    batch_size=BATCH_SIZE
)
print(f"[train] Sınıflar: {sinif_adlari}")


# ─────────────────────────────────────────────
# Model (model_builder.py)
# ─────────────────────────────────────────────
# TinyVGG: 2 konvolüsyon bloğu + lineer sınıflandırıcı
model = model_builder.TinyVGG(
    girdi_kanali=3,
    gizli_birim=HIDDEN_UNITS,
    cikti_sinif=len(sinif_adlari)
).to(device)


# ─────────────────────────────────────────────
# Kayıp Fonksiyonu ve Optimizer
# ─────────────────────────────────────────────
kayip_fn  = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ─────────────────────────────────────────────
# Eğitim (engine.py)
# ─────────────────────────────────────────────
# egit → her epoch için (kayıp, doğruluk) değerlerini sözlükte toplar
sonuclar = engine.egit(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    kayip_fn=kayip_fn,
    epochs=NUM_EPOCHS,
    device=device
)


# ─────────────────────────────────────────────
# Model Kaydetme (utils.py)
# ─────────────────────────────────────────────
# Yalnızca state_dict kaydedilir — taşınabilir ve kompakt
utils.modeli_kaydet(
    model=model,
    hedef_klasor="models",
    model_adi="tinyvgg_modular.pth"
)
