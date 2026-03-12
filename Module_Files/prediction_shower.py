"""
prediction_shower.py — Tahmin Görselleştirici
===============================================
Proje  : Eğitim Teknolojileri için PyTorch Eğitimi
Yazar  : Ugur Sirvermez — Bursa Uludag Universitesi
Lisans : CC BY-NC-SA 4.0

Bu modül, eğitilmiş bir PyTorch modeliyle tek bir görüntü üzerinde
sınıf tahmini yapar ve sonucu matplotlib ile görselleştirir.

İçerik:
    - tahmin_ve_goster : Görüntüyü tahmin eder, sınıf adı + olasılıkla gösterir.

Kullanım:
    from Module_Files.prediction_shower import tahmin_ve_goster
    tahmin_ve_goster(model, "test_images/pizza.jpg", ["pizza", "steak", "sushi"])
"""

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image


# ─────────────────────────────────────────────
# Varsayılan cihaz
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"


def tahmin_ve_goster(
    model: torch.nn.Module,
    goruntu_yolu: str,
    sinif_adlari: List[str],
    goruntu_boyutu: Tuple[int, int] = (224, 224),
    donusum: torchvision.transforms = None,
    device: torch.device = device
) -> None:
    """Tek bir görüntü üzerinde tahmin yapar ve sonucu görselleştirir.

    Adımlar:
        1. Görüntüyü PIL ile aç.
        2. Dönüşüm yoksa varsayılan ImageNet normalize dönüşümünü uygula.
        3. Batch boyutu ekle → (1, C, H, W).
        4. model.eval() + torch.inference_mode() ile ileri geçiş yap.
        5. Softmax → argmax ile tahmin sınıfını belirle.
        6. matplotlib ile görüntüyü ve başlığı (sınıf + olasılık) göster.

    Args:
        model          : Tahmin yapacak eğitilmiş PyTorch modeli.
        goruntu_yolu   : Görüntü dosyasının yolu (str).
        sinif_adlari   : Sınıf isimlerinin listesi (indeks sırasıyla).
        goruntu_boyutu : Yeniden boyutlandırma hedefi. Varsayılan: (224, 224).
        donusum        : Özel torchvision dönüşüm zinciri. None ise varsayılan
                         ImageNet normalizasyonu kullanılır.
        device         : Hesaplama cihazı. Varsayılan: mevcut GPU veya CPU.

    Returns:
        None — sonucu matplotlib penceresiyle gösterir.

    Örnek:
        >>> tahmin_ve_goster(
        ...     model=model,
        ...     goruntu_yolu="test_images/pizza_01.jpg",
        ...     sinif_adlari=["pizza", "steak", "sushi"]
        ... )
    """
    # ── 1. Görüntüyü aç ──────────────────────────────────────────────
    img = Image.open(goruntu_yolu)

    # ── 2. Dönüşüm hazırla ───────────────────────────────────────────
    # Özel dönüşüm verilmemişse ImageNet istatistikleriyle normalize et.
    if donusum is not None:
        goruntu_donusumu = donusum
    else:
        goruntu_donusumu = transforms.Compose([
            transforms.Resize(goruntu_boyutu),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    # ── 3. Modeli cihaza taşı, eval moduna al ────────────────────────
    model.to(device)
    model.eval()

    with torch.inference_mode():
        # Batch boyutu ekle: (C, H, W) → (1, C, H, W)
        donusturulmus = goruntu_donusumu(img).unsqueeze(dim=0)

        # İleri geçiş — ham logitler
        logitler = model(donusturulmus.to(device))

    # ── 4. Olasılık ve tahmin sınıfı ─────────────────────────────────
    # softmax: logitleri [0, 1] aralığına normalleştirir
    olasiliklar  = torch.softmax(logitler, dim=1)
    tahmin_sinif = torch.argmax(olasiliklar, dim=1)

    # ── 5. Görselleştir ──────────────────────────────────────────────
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Tahmin: {sinif_adlari[tahmin_sinif]} | "
        f"Olasılık: {olasiliklar.max():.3f}"
    )
    plt.axis(False)
