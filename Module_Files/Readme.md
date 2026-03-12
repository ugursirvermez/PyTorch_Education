# Module_Files — Modüler PyTorch Bileşenleri

> Bu klasör, FoodVision Mini projesini (ve benzer görüntü sınıflandırma projelerini)
> destekleyen yeniden kullanılabilir Python modüllerini barındırır.
> Her `.py` dosyası tek bir sorumluluğa sahiptir ve birbirinden bağımsız olarak test edilebilir.

```
Module_Files/
├── data_setup.py        ← Veri indirme + DataLoader oluşturma
├── model_builder.py     ← TinyVGG CNN mimarisi
├── engine.py            ← Eğitim ve değerlendirme döngüleri
├── utils.py             ← Yardımcı araçlar (model kaydetme)
├── train.py             ← Tüm bileşenleri bir araya getiren çalıştırma scripti
└── prediction_shower.py ← Tek görüntü üzerinde tahmin + görselleştirme
```

---

## PyTorch İş Akışı

```
[Ham Veri]
    ↓  data_setup.py
[DataLoader]
    ↓  model_builder.py
[Model (TinyVGG)]
    ↓  engine.py
[Eğitilmiş Model]
    ↓  utils.py
[Kaydedilmiş .pth]
    ↓  prediction_shower.py
[Tahmin + Görselleştirme]
```

![pytorch_workflow_funcs](https://github.com/user-attachments/assets/bcdd97af-28ba-42ed-9b29-ab4489b98eb3)

---

## Dosyalar

### `data_setup.py` — Veri Hazırlama

Görüntü veri setini indirir, ZIP'ten çıkarır ve PyTorch `DataLoader` nesnelerine dönüştürür.

| Fonksiyon | Açıklama |
|---|---|
| `veri_indir(kaynak_url, hedef_klasor)` | ZIP dosyasını indirir ve hedef klasöre çıkarır |
| `create_dataloaders(train_dir, test_dir, transform, batch_size)` | Eğitim ve test DataLoader'larını + sınıf listesini döndürür |

```python
from Module_Files import data_setup
from torchvision import transforms

train_dl, test_dl, siniflar = data_setup.create_dataloaders(
    train_dir="data/pizza_steak_sushi/train",
    test_dir="data/pizza_steak_sushi/test",
    transform=transforms.ToTensor(),
    batch_size=32
)
```

---

### `model_builder.py` — Model Tanımlama

TinyVGG mimarisini `nn.Module` alt sınıfı olarak tanımlar.

| Sınıf | Açıklama |
|---|---|
| `TinyVGG(girdi_kanali, gizli_birim, cikti_sinif)` | 2 konvolüsyon bloğu + lineer sınıflandırıcı |

**Mimari özeti:**
```
[Girdi: (B, C, H, W)]
    ↓  KonvBlok-1: Conv2d → ReLU → Conv2d → ReLU → MaxPool2d
    ↓  KonvBlok-2: Conv2d → ReLU → Conv2d → ReLU → MaxPool2d
    ↓  Sınıflandırıcı: Flatten → Linear
[Çıktı: (B, sınıf_sayısı) — ham logitler]
```

```python
from Module_Files.model_builder import TinyVGG

model = TinyVGG(girdi_kanali=3, gizli_birim=10, cikti_sinif=3)
```

> **Not:** `64×64` girdi boyutu için `in_features = gizli_birim × 13 × 13` olmalıdır.
> Farklı girdi boyutlarında bu değeri güncellemeniz gerekir.

---

### `engine.py` — Eğitim Motoru

Tek epoch eğitim/test döngülerini ve tam eğitim orkestrasını sağlar.

| Fonksiyon | Açıklama |
|---|---|
| `egitim_adimi(model, dataloader, kayip_fn, optimizer, device)` | Bir epoch eğitim; `(kayıp, doğruluk)` döndürür |
| `test_adimi(model, dataloader, kayip_fn, device)` | Bir epoch değerlendirme; `(kayıp, doğruluk)` döndürür |
| `egit(model, train_dl, test_dl, optimizer, kayip_fn, epochs, device)` | N epoch eğitim; sonuç sözlüğü döndürür |

```python
from Module_Files import engine

sonuclar = engine.egit(
    model=model,
    train_dataloader=train_dl,
    test_dataloader=test_dl,
    optimizer=optimizer,
    kayip_fn=loss_fn,
    epochs=5,
    device=device
)
# sonuclar: {"egitim_kayip": [...], "egitim_dogru": [...], ...}
```

---

### `utils.py` — Yardımcı Araçlar

Model ağırlıklarını diske kaydetmek için yardımcı fonksiyon içerir.

| Fonksiyon | Açıklama |
|---|---|
| `modeli_kaydet(model, hedef_klasor, model_adi)` | `state_dict`'i `.pt` / `.pth` olarak kaydeder |

```python
from Module_Files import utils

utils.modeli_kaydet(
    model=model,
    hedef_klasor="models",
    model_adi="tinyvgg_v1.pth"
)
# [utils] Model kaydedildi → models/tinyvgg_v1.pth
```

> **Neden `state_dict`?** Tüm modeli kaydetmek yerine yalnızca ağırlıkları kaydetmek
> hem daha az yer kaplar hem de PyTorch sürümleri arasında taşınabilirlik sağlar.

---

### `train.py` — Çalıştırma Scripti

Tüm modülleri bir araya getiren uçtan uca eğitim scripti. Hiperparametreleri değiştirip
doğrudan çalıştırabilirsiniz.

**Hiperparametreler:**

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `NUM_EPOCHS` | 5 | Eğitim epoch sayısı |
| `BATCH_SIZE` | 32 | Mini-batch boyutu |
| `HIDDEN_UNITS` | 10 | Conv filtre sayısı |
| `LEARNING_RATE` | 0.001 | Adam optimizer öğrenme hızı |

```bash
# Scripti doğrudan çalıştırmak için:
python Module_Files/train.py
```

---

### `prediction_shower.py` — Tahmin Görselleştirici

Eğitilmiş bir modeli kullanarak tek bir görüntü üzerinde tahmin yapar ve sonucu
matplotlib ile görselleştirir.

| Fonksiyon | Açıklama |
|---|---|
| `tahmin_ve_goster(model, goruntu_yolu, sinif_adlari, goruntu_boyutu, donusum, device)` | Görüntüyü tahmin eder ve başlığında sınıf + olasılık gösterir |

```python
from Module_Files.prediction_shower import tahmin_ve_goster

tahmin_ve_goster(
    model=model,
    goruntu_yolu="test_images/pizza.jpg",
    sinif_adlari=["pizza", "steak", "sushi"]
)
```

---

## İlgili Notebook'lar

Bu modüllerin adım adım nasıl geliştirildiğini görmek için ilgili notebook'lara bakın:

| Modül | İlgili Notebook |
|---|---|
| `data_setup.py` | `04_pytorch_custom_datasets.ipynb` |
| `model_builder.py` | `05_pytorch_going_modular.ipynb` |
| `engine.py` | `05_pytorch_going_modular.ipynb` |
| `utils.py` | `05_pytorch_going_modular.ipynb` |
| `train.py` | `05_pytorch_going_modular.ipynb` |
| `prediction_shower.py` | `06_pytorch_transfer_learning.ipynb` |

---

*Proje: Eğitim Teknolojileri için PyTorch Eğitimi — Ugur Sirvermez, Bursa Uludağ Üniversitesi*
*Lisans: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)*
