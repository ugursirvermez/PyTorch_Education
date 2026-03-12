"""
utils.py — Yardımcı Araçlar Modülü
=====================================
Proje  : Eğitim Teknolojileri için PyTorch Eğitimi
Yazar  : Ugur Sirvermez — Bursa Uludag Universitesi
Lisans : CC BY-NC-SA 4.0

Bu modül, eğitim sürecinde sık kullanılan yardımcı fonksiyonları sağlar.

İçerik:
    - modeli_kaydet : Modelin state_dict'ini diske kaydeder.
"""

from pathlib import Path
import torch


def modeli_kaydet(
    model: torch.nn.Module,
    hedef_klasor: str,
    model_adi: str
) -> None:
    """Modelin ağırlıklarını (state_dict) belirtilen konuma kaydeder.

    Neden state_dict?
        Tüm modeli kaydetmek yerine yalnızca ağırlıkları kaydetmek
        hem daha az yer kaplar hem de farklı PyTorch sürümleri arasında
        daha taşınabilirdir. Modeli yüklerken aynı mimariyi yeniden
        oluşturup ağırlıkları yüklemek yeterlidir.

    Args:
        model        : Kaydedilecek PyTorch modeli.
        hedef_klasor : Dosyanın kaydedileceği klasör adı (str).
        model_adi    : Dosya adı; ".pt" veya ".pth" uzantısıyla bitmelidir.

    Raises:
        AssertionError: model_adi ".pt" veya ".pth" ile bitmiyorsa.

    Örnek:
        >>> modeli_kaydet(model, "models", "tinyVGG_v1.pth")
        [utils] Model kaydedildi → models/tinyVGG_v1.pth
    """
    assert model_adi.endswith(".pth") or model_adi.endswith(".pt"), (
        f"model_adi '.pt' veya '.pth' ile bitmelidir. Alınan: '{model_adi}'"
    )

    klasor = Path(hedef_klasor)
    klasor.mkdir(parents=True, exist_ok=True)

    kayit_yolu = klasor / model_adi
    torch.save(obj=model.state_dict(), f=kayit_yolu)
    print(f"[utils] Model kaydedildi → {kayit_yolu}")
