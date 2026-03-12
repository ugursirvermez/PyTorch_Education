"""
model_builder.py — Model Tanımlama Modülü
==========================================
Proje  : Eğitim Teknolojileri için PyTorch Eğitimi
Yazar  : Ugur Sirvermez — Bursa Uludag Universitesi
Lisans : CC BY-NC-SA 4.0

Bu modül, projede kullanılan CNN mimarilerini barındırır.
Her sınıf nn.Module'den türetilmiş ve yeniden kullanılabilir şekilde tasarlanmıştır.

İçerik:
    - TinyVGG : 2 konvolüsyon bloğu + sınıflandırıcıdan oluşan küçük ölçekli CNN.

Kullanım:
    from Module_Files.model_builder import TinyVGG
    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=3)
"""

import torch
from torch import nn


class TinyVGG(nn.Module):
    """Küçük ölçekli Evrişimsel Sinir Ağı (CNN) — TinyVGG mimarisi.

    VGG ailesinden ilham alan bu model iki konvolüsyon bloğu ve
    bir tam bağlantılı sınıflandırıcıdan oluşur. Görüntü işleme
    konularını öğrenmek için tasarlanmıştır.

    Mimari:
        [Girdi]
           ↓
        KonvBlok-1: Conv2d → ReLU → Conv2d → ReLU → MaxPool2d
           ↓
        KonvBlok-2: Conv2d → ReLU → Conv2d → ReLU → MaxPool2d
           ↓
        Sınıflandırıcı: Flatten → Linear
           ↓
        [Çıktı — sınıf olasılıkları]

    Args:
        girdi_kanali  : Giriş görüntüsündeki renk kanalı sayısı
                        (gri tonlamalı = 1, RGB = 3).
        gizli_birim   : Her konvolüsyon katmanındaki filtre (kanal) sayısı.
        cikti_sinif   : Sınıflandırılacak sınıf sayısı.

    Örnek:
        >>> model = TinyVGG(girdi_kanali=3, gizli_birim=10, cikti_sinif=3)
        >>> x = torch.randn(1, 3, 64, 64)
        >>> model(x).shape
        torch.Size([1, 3])
    """

    def __init__(
        self,
        girdi_kanali: int,
        gizli_birim: int,
        cikti_sinif: int
    ) -> None:
        super().__init__()

        # ── Konvolüsyon Bloğu 1 ──────────────────────────────────────
        # Her blok: 2× (Conv2d → ReLU) + MaxPool2d
        # MaxPool2d(2) boyutu yarıya indirir: 64×64 → 32×32
        self.konv_blok_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=girdi_kanali,
                out_channels=gizli_birim,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=gizli_birim,
                out_channels=gizli_birim,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ── Konvolüsyon Bloğu 2 ──────────────────────────────────────
        self.konv_blok_2 = nn.Sequential(
            nn.Conv2d(gizli_birim, gizli_birim, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(gizli_birim, gizli_birim, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # ── Sınıflandırıcı ───────────────────────────────────────────
        # Flatten: 2B özellik haritasını 1B vektöre çevirir
        # Linear : özellik → sınıf skoru (logit)
        # Not: in_features, giriş boyutuna göre hesaplanmalıdır.
        #      64×64 girdi için her iki blok sonrası 13×13 boyutuna düşer.
        self.siniflandirici = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=gizli_birim * 13 * 13,
                out_features=cikti_sinif
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş (forward pass).

        Args:
            x : (batch, kanal, yükseklik, genişlik) boyutunda tensör.

        Returns:
            (batch, cikti_sinif) boyutunda ham logit tensörü.
        """
        x = self.konv_blok_1(x)
        x = self.konv_blok_2(x)
        return self.siniflandirici(x)
