"""
data_setup.py — Veri Hazırlama Modülü
======================================
Proje  : Eğitim Teknolojileri için PyTorch Eğitimi
Yazar  : Ugur Sirvermez — Bursa Uludag Universitesi
Lisans : CC BY-NC-SA 4.0

Bu modül, görüntü sınıflandırma projeleri için veri setini
indirir, ayıklar ve DataLoader nesnelerine dönüştürür.

Kullanım:
    from Module_Files import data_setup
    train_dl, test_dl, siniflar = data_setup.create_dataloaders(
        train_dir="data/pizza_steak_sushi/train",
        test_dir="data/pizza_steak_sushi/test",
        transform=transforms.ToTensor(),
        batch_size=32
    )
"""

import os
import zipfile
from pathlib import Path
from typing import Tuple, List

import requests
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# Veri İndirme
# ─────────────────────────────────────────────

def veri_indir(
    kaynak_url: str,
    hedef_klasor: Path
) -> Path:
    """Zip arşivini indirir, ayıklar ve yerel klasöre kaydeder.

    Args:
        kaynak_url   : İndirilecek zip dosyasının URL'si.
        hedef_klasor : Dosyaların ayıklanacağı klasör yolu (Path nesnesi).

    Returns:
        Görüntülerin bulunduğu hedef klasörün Path nesnesi.
    """
    if hedef_klasor.is_dir():
        print(f"[data_setup] Veri zaten mevcut → {hedef_klasor}")
        return hedef_klasor

    print(f"[data_setup] İndiriliyor: {kaynak_url}")
    hedef_klasor.mkdir(parents=True, exist_ok=True)

    zip_yolu = hedef_klasor.parent / "gecici_veri.zip"
    with open(zip_yolu, "wb") as f:
        f.write(requests.get(kaynak_url).content)

    with zipfile.ZipFile(zip_yolu, "r") as zf:
        print("[data_setup] Ayıklanıyor...")
        zf.extractall(hedef_klasor)

    os.remove(zip_yolu)
    print(f"[data_setup] Tamamlandı → {hedef_klasor}")
    return hedef_klasor


# ─────────────────────────────────────────────
# DataLoader Oluşturma
# ─────────────────────────────────────────────

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Eğitim ve test DataLoader'larını oluşturur.

    torchvision.datasets.ImageFolder kullanır; klasör yapısı şu şekilde
    olmalıdır:
        train_dir/
            sinif_adi_1/  resim1.jpg  resim2.jpg ...
            sinif_adi_2/  ...
        test_dir/
            sinif_adi_1/  ...

    Args:
        train_dir   : Eğitim görüntülerinin bulunduğu klasör yolu (str).
        test_dir    : Test görüntülerinin bulunduğu klasör yolu (str).
        transform   : Görüntülere uygulanacak torchvision dönüşüm zinciri.
        batch_size  : Her mini-batch'teki örnek sayısı. Varsayılan: 32.
        num_workers : Veri yükleme için kullanılacak işçi sayısı. Varsayılan: 0.

    Returns:
        (train_dataloader, test_dataloader, sinif_adlari) demeti:
            - train_dataloader : Karıştırılmış eğitim verisi yükleyici.
            - test_dataloader  : Sıralı test verisi yükleyici.
            - sinif_adlari     : Sınıf isimlerinin alfabetik listesi.
    """
    egitim_verisi = datasets.ImageFolder(train_dir, transform=transform)
    test_verisi   = datasets.ImageFolder(test_dir,  transform=transform)

    sinif_adlari = egitim_verisi.classes

    train_dataloader = DataLoader(
        egitim_verisi,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_dataloader = DataLoader(
        test_verisi,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_dataloader, test_dataloader, sinif_adlari
