"""
engine.py — Eğitim ve Değerlendirme Motoru
============================================
Proje  : Eğitim Teknolojileri için PyTorch Eğitimi
Yazar  : Ugur Sirvermez — Bursa Uludag Universitesi
Lisans : CC BY-NC-SA 4.0

Bu modül, PyTorch modellerinin eğitim ve değerlendirme döngülerini
yeniden kullanılabilir fonksiyonlar olarak sağlar.

İçerik:
    - egitim_adimi  : Tek bir epoch için eğitim döngüsü.
    - test_adimi    : Tek bir epoch için değerlendirme döngüsü.
    - egit          : Belirtilen epoch sayısı kadar eğitim + değerlendirme.

Kullanım:
    from Module_Files import engine
    sonuclar = engine.egit(
        model, train_dl, test_dl,
        optimizer, loss_fn,
        epochs=5, device=device
    )
"""

from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm


# ─────────────────────────────────────────────
# Tek Epoch Eğitim Adımı
# ─────────────────────────────────────────────

def egitim_adimi(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    kayip_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Modeli bir epoch boyunca eğitir.

    Eğitim döngüsü adımları:
        1. model.train() — eğitim modunu aç
        2. İleri geçiş (forward pass)
        3. Kaybı hesapla
        4. optimizer.zero_grad() — önceki gradyanları sıfırla
        5. loss.backward() — geri yayılım
        6. optimizer.step() — ağırlıkları güncelle

    Args:
        model      : Eğitilecek PyTorch modeli.
        dataloader : Eğitim mini-batch'lerini sağlayan DataLoader.
        kayip_fn   : Kayıp fonksiyonu (örn. nn.CrossEntropyLoss).
        optimizer  : Ağırlık güncelleyici (örn. torch.optim.Adam).
        device     : Hesaplamaların yapılacağı cihaz ("cpu" veya "cuda").

    Returns:
        (ortalama_egitim_kaybi, ortalama_egitim_dogrulugu) demeti.
    """
    model.train()
    toplam_kayip = 0.0
    toplam_dogru = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # İleri geçiş
        y_tahmin = model(X)

        # Kayıp hesapla ve biriktir
        kayip = kayip_fn(y_tahmin, y)
        toplam_kayip += kayip.item()

        # Geri yayılım + ağırlık güncelleme
        optimizer.zero_grad()
        kayip.backward()
        optimizer.step()

        # Doğruluğu hesapla (sınıflandırma için)
        tahmin_sinif = torch.argmax(torch.softmax(y_tahmin, dim=1), dim=1)
        toplam_dogru += (tahmin_sinif == y).sum().item() / len(y_tahmin)

    ort_kayip  = toplam_kayip  / len(dataloader)
    ort_dogru  = toplam_dogru  / len(dataloader)
    return ort_kayip, ort_dogru


# ─────────────────────────────────────────────
# Tek Epoch Değerlendirme Adımı
# ─────────────────────────────────────────────

def test_adimi(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    kayip_fn: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Modeli değerlendirme modunda çalıştırır; gradyan hesaplanmaz.

    Args:
        model      : Değerlendirilecek PyTorch modeli.
        dataloader : Test mini-batch'lerini sağlayan DataLoader.
        kayip_fn   : Kayıp fonksiyonu.
        device     : Hesaplama cihazı.

    Returns:
        (ortalama_test_kaybi, ortalama_test_dogrulugu) demeti.
    """
    model.eval()
    toplam_kayip = 0.0
    toplam_dogru = 0.0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            tahmin_logit = model(X)
            kayip = kayip_fn(tahmin_logit, y)
            toplam_kayip += kayip.item()

            tahmin_sinif = tahmin_logit.argmax(dim=1)
            toplam_dogru += (tahmin_sinif == y).sum().item() / len(tahmin_sinif)

    ort_kayip = toplam_kayip / len(dataloader)
    ort_dogru = toplam_dogru / len(dataloader)
    return ort_kayip, ort_dogru


# ─────────────────────────────────────────────
# Ana Eğitim Fonksiyonu
# ─────────────────────────────────────────────

def egit(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    kayip_fn: torch.nn.Module,
    epochs: int,
    device: torch.device
) -> Dict[str, List[float]]:
    """Modeli belirtilen epoch sayısı kadar eğitir ve sonuçları döndürür.

    Her epoch'ta eğitim adımı ve test adımı çalışır; sonuçlar sözlükte biriktirilir.

    Args:
        model             : Eğitilecek PyTorch modeli.
        train_dataloader  : Eğitim DataLoader'ı.
        test_dataloader   : Test DataLoader'ı.
        optimizer         : Optimizer nesnesi.
        kayip_fn          : Kayıp fonksiyonu.
        epochs            : Kaç epoch eğitim yapılacağı.
        device            : Hesaplama cihazı.

    Returns:
        Şu anahtarları içeren sözlük:
            "egitim_kayip", "egitim_dogru", "test_kayip", "test_dogru"
        Her anahtar altında epoch başına ölçüm listesi bulunur.

    Örnek:
        >>> sonuclar = egit(model, train_dl, test_dl, opt, loss_fn, 5, device)
        >>> print(sonuclar["test_dogru"])
    """
    sonuclar: Dict[str, List[float]] = {
        "egitim_kayip": [],
        "egitim_dogru": [],
        "test_kayip":   [],
        "test_dogru":   []
    }

    for epoch in tqdm(range(epochs)):
        eg_kayip, eg_dogru = egitim_adimi(
            model, train_dataloader, kayip_fn, optimizer, device
        )
        ts_kayip, ts_dogru = test_adimi(
            model, test_dataloader, kayip_fn, device
        )

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Eğitim → kayıp: {eg_kayip:.4f}  doğruluk: {eg_dogru:.4f} | "
            f"Test   → kayıp: {ts_kayip:.4f}  doğruluk: {ts_dogru:.4f}"
        )

        sonuclar["egitim_kayip"].append(eg_kayip)
        sonuclar["egitim_dogru"].append(eg_dogru)
        sonuclar["test_kayip"].append(ts_kayip)
        sonuclar["test_dogru"].append(ts_dogru)

    return sonuclar
