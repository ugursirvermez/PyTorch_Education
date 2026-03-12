# Eğitim Teknolojileri için PyTorch Eğitimi 🔥

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/Lisans-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--7266--6408-green)](https://orcid.org/0000-0001-7266-6408)

> **Atıf / Attribution**
> Bu proje, [Daniel Bourke](https://github.com/mrdbourke)'ün
> **[Zero to Mastery Learn PyTorch for Deep Learning](https://github.com/mrdbourke/pytorch-deep-learning)**
> ([learnpytorch.io](https://www.learnpytorch.io)) adlı açık kaynak çalışmasından ilham alınarak hazırlanmıştır.
> Orijinal içerik Türkçeye çevrilmiş, eğitim teknolojistleri için yeniden yapılandırılmış ve pedagojik modül formatına dönüştürülmüştür.
> Orijinal çalışma MIT lisansı ile yayınlanmıştır; bu türev çalışma CC BY-NC-SA 4.0 lisansı kapsamındadır.
>
> *This project is inspired by Daniel Bourke's "Zero to Mastery Learn PyTorch for Deep Learning" (MIT License).
> Content has been translated into Turkish, restructured, and adapted for educational technology audiences under CC BY-NC-SA 4.0.*

---

## İçindekiler

- [Bu Kurs Kimin İçin?](#bu-kurs-kimin-için)
- [Ön Gereksinimler](#ön-gereksinimler)
- [Modül Haritası](#modül-haritası)
- [Nasıl Kullanılır?](#nasıl-kullanılır)
- [Proje Yapısı](#proje-yapısı)
- [Kavramsal Arka Plan](#kavramsal-arka-plan)
- [Kurulum](#kurulum)
- [Kaynaklar ve Atıflar](#kaynaklar-ve-atıflar)
- [Lisans](#lisans)

---

## Bu Kurs Kimin İçin?

Bu açık eğitim kaynağı (OER) özellikle şu kitleler için tasarlanmıştır:

- **Eğitim teknolojistleri** — yapay zekâyı eğitim süreçlerine entegre etmek isteyenler
- **Öğretmenler ve akademisyenler** — PyTorch'u sıfırdan öğrenmek isteyenler
- **Türkçe kaynak arayan yazılım geliştiriciler** — derin öğrenmeye giriş yapacaklar
- **Lisans ve yüksek lisans öğrencileri** — uygulamalı makine öğrenmesi çalışmak isteyenler

Her modül pedagojik olarak yapılandırılmıştır: öğrenme hedefleri, ön koşullar, tahmini süre ve modül özeti içerir.

---

## Ön Gereksinimler

| Konu | Düzey |
|------|-------|
| Python programlama | Temel (değişken, döngü, fonksiyon, sınıf) |
| NumPy / dizi işlemleri | Temel aşinalık (zorunlu değil) |
| Matematik | Lise düzeyi türev bilgisi (yalnızca BackPropagation modülü için) |
| Donanım | GPU önerilir; tüm modüller Google Colab'de GPU olmadan da çalışır |

> 💡 Derin öğrenme veya PyTorch hakkında **sıfır bilgi** ile başlayabilirsiniz.

---

## Modül Haritası

Kurs, temel tensör işlemlerinden model dağıtımına uzanan **14 ana modül** ve **1 ek modül**den oluşur.

| No | Modül Adı | Konu | Süre | Colab |
|----|-----------|------|------|-------|
| 00 | PyTorch'a Giriş | Tensörler, veri tipleri, GPU/CPU | ~2 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/00_pyTorch.ipynb) |
| 01 | İlk PyTorch Modeli | Doğrusal regresyon, eğitim döngüsü | ~2.5 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/01_pytorch_first_model.ipynb) |
| 02 | PyTorch İş Akışı | Uçtan uca pipeline, model kaydetme | ~3 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/02_pytorch_all_one_place.ipynb) |
| 03 | Sinir Ağı Sınıflandırması | Binary, Multiclass, karar sınırları | ~3 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/03_pytorch_nn_model_classification.ipynb) |
| 04 | Bilgisayarlı Görü — Giriş | TorchVision, FashionMNIST, DataLoader | ~3 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/04_pytorch_computer_vision.ipynb) |
| 05 | Bilgisayarlı Görü — CNN | TinyVGG, Conv2d, MaxPool2d | ~3 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/05_pytorch_cv_cnn_model_2.ipynb) |
| 06 | Özel Veri Setleri | Dataset sınıfı, augmentation, FoodVisionMini | ~4 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/06_pytorch_custom_datasets.ipynb) |
| 07 | Python Betik Modülü | Modüler kod, .py dosyaları, %%writefile | ~2 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/07_python_script_module.ipynb) |
| 08 | Transfer Öğrenme | EfficientNet, freeze, fine-tuning | ~3 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/08_pytorch_transfer_learning.ipynb) |
| 09 | Deney Takibi | TensorBoard, deney karşılaştırma | ~3 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/09_pytorch_experiment_tracking.ipynb) |
| 10 | Makale Kopyalama | Vision Transformer (ViT), patch embedding | ~4.5 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/10_pytorch_paper_replicating.ipynb) |
| 11 | Model Dağıtımı | Gradio, Hugging Face Spaces | ~3.5 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/11_pytorch_model_deployment.ipynb) |
| 12 | FoodVision Big Projesi | EfficientNet-B2, Food101, uçtan uca | ~4.5 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/12_pytorch_foodvision.ipynb) |
| 13 | Takviyeli Öğrenme | DQN, Gymnasium, CartPole | ~3.5 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/13_pytorch_reinforcement_learning.ipynb) |
| Ek | Geri Yayılım Örneği | Backpropagation matematiği, NumPy | ~2 saat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ugursirvermez/PyTorch_Education/blob/main/BackPropagation_Example.ipynb) |

**Toplam tahmini süre:** ~45–50 saat

---

## Nasıl Kullanılır?

### Seçenek 1 — Google Colab (Önerilen)

Herhangi bir kurulum gerektirmez. Yukarıdaki tablodaki Colab rozetlerine tıklayın ve sırasıyla modülleri takip edin.

```
Runtime → Change Runtime Type → T4 GPU
```

### Seçenek 2 — Yerel Kurulum

```bash
# 1. Repoyu klonlayın
git clone https://github.com/ugursirvermez/PyTorch_Education.git
cd PyTorch_Education

# 2. Sanal ortam oluşturun (önerilir)
python -m venv pytorch_env
source pytorch_env/bin/activate        # Linux / macOS
# pytorch_env\Scripts\activate         # Windows

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Jupyter başlatın
jupyter notebook
```

### Modülleri Hangi Sırayla Takip Etmeli?

Kurs doğrusal olarak tasarlanmıştır; **00 → 13** sırasını takip etmeniz önerilir.
Ancak her modülün "Ön Koşullar" bölümüne bakarak isteğe bağlı başlangıç noktası seçebilirsiniz.

---

## Proje Yapısı

```
PyTorch_Education/
│
├── 00_pyTorch.ipynb                      ← Modül 00: Tensörler
├── 01_pytorch_first_model.ipynb          ← Modül 01: İlk Model
├── 02_pytorch_all_one_place.ipynb        ← Modül 02: İş Akışı
├── 03_pytorch_nn_model_classification.ipynb
├── 04_pytorch_computer_vision.ipynb
├── 05_pytorch_cv_cnn_model_2.ipynb
├── 06_pytorch_custom_datasets.ipynb
├── 07_python_script_module.ipynb
├── 08_pytorch_transfer_learning.ipynb
├── 09_pytorch_experiment_tracking.ipynb
├── 10_pytorch_paper_replicating.ipynb
├── 11_pytorch_model_deployment.ipynb
├── 12_pytorch_foodvision.ipynb
├── 13_pytorch_reinforcement_learning.ipynb
├── BackPropagation_Example.ipynb         ← Ek Modül
│
├── Module_Files/                         ← Yeniden kullanılabilir Python modülleri
│   ├── data_setup.py                     ← Veri indirme & DataLoader
│   ├── engine.py                         ← Eğitim & değerlendirme döngüsü
│   ├── model_builder.py                  ← TinyVGG CNN mimarisi
│   ├── utils.py                          ← Model kaydetme yardımcıları
│   ├── train.py                          ← Komut satırı eğitim betiği
│   ├── prediction_shower.py              ← Tahmin görselleştirme
│   └── Readme.md                         ← Modül açıklamaları
│
├── CITATION.cff                          ← Zenodo / akademik atıf dosyası
├── LICENSE                               ← CC BY-NC-SA 4.0
└── README.md                             ← Bu dosya
```

---

## Kavramsal Arka Plan

### Makine Öğrenmesi ve Derin Öğrenme Nedir?

Makine öğrenmesi, verilerdeki (resim, metin, sayı vb.) örüntüleri otomatik olarak bulmayı hedefler.
Geleneksel programlamada kuralları biz yazarız; makine öğrenmesinde ise model bu kuralları veriden öğrenir.

```
Yapay Zekâ (AI)
  └── Makine Öğrenmesi (ML)
        └── Derin Öğrenme (DL)  ← Bu kursun odağı
              └── PyTorch        ← Kullandığımız araç
```

PyTorch iş akışı şu adımları izler:

1. Veriyi hazırla ve yükle
2. Modeli tanımla (`nn.Module`)
3. Modeli eğit (kayıp fonksiyonu + optimizer)
4. Modeli değerlendir
5. Modeli kaydet ve kullan

![Pytorch workflow](https://github.com/user-attachments/assets/cb0c8cfb-e21f-4440-8acb-5b6fc2bc56a6)

### Temel TorchVision Kütüphaneleri

| Kütüphane | Açıklama |
|-----------|----------|
| `torchvision.datasets` | Hazır veri setleri (MNIST, FashionMNIST, Food101…) |
| `torchvision.models` | Önceden eğitilmiş modeller (ResNet, EfficientNet, ViT…) |
| `torchvision.transforms` | Görüntü ön işleme ve artırma dönüşümleri |
| `torch.utils.data.Dataset` | Özel veri seti tanımlamak için temel sınıf |
| `torch.utils.data.DataLoader` | Mini-batch veri yükleme ve karıştırma |

### Nöral Ağ Sınıflandırma Türleri

| Tür | Tanım | Örnek |
|-----|-------|-------|
| Binary Classification | İki sınıftan biri (0 / 1) | E-posta: spam mı, değil mi? |
| Multiclass Classification | Birden fazla sınıftan biri | Görüntü: kedi mi, köpek mi, kuş mu? |
| Multilabel Classification | Bir örnekte birden fazla etiket | Makale: hem "AI" hem "eğitim" etiketi |

![IOClass_Data](https://github.com/user-attachments/assets/19f8c440-a0aa-4c11-a788-d7bbdcee1ad1)

### Binary vs. Multiclass — Hiperparametreler

| Hiperparametre | Binary | Multiclass |
|----------------|--------|------------|
| Çıktı katmanı nöronu | 1 | Sınıf sayısı kadar |
| Çıktı aktivasyonu | Sigmoid | Softmax |
| Kayıp fonksiyonu | BCELoss | CrossEntropyLoss |
| Optimizer | SGD veya Adam | SGD veya Adam |

### Evrişimsel Sinir Ağları (CNN)

CNN'ler görüntüdeki yerel örüntüleri (kenar, doku, şekil) öğrenir.
Üç temel katman türünden oluşur:

| Katman | PyTorch | Görevi |
|--------|---------|--------|
| Evrişimsel Katman | `nn.Conv2d()` | Görüntüden özellik çıkarma |
| Havuzlama Katmanı | `nn.MaxPool2d()` | Boyut küçültme, önemli bilgiyi koruma |
| Tam Bağlantılı Katman | `nn.Linear()` | Öğrenilen özellikleri sınıfa eşleme |

![CNN_Figure](https://github.com/user-attachments/assets/3484b0c5-441b-48a9-b39c-6201b9359a80)

### Regresyon Eğrileri

![linear_non_linear](https://github.com/user-attachments/assets/a6baafbf-4507-4338-8e66-5fbbf7afb956)

### Transfer Öğrenme

Milyonlarca görüntüyle eğitilmiş bir modelin ağırlıklarını alıp
kendi küçük veri setinize uyarlamanıza **transfer öğrenme** denir.
Önceki katmanlar dondurulur (`requires_grad = False`), yalnızca son katman yeniden eğitilir.

### Model Dağıtımı (Deployment)

Eğitim sonrası model bir ürüne dönüştürülür. Bu kurs **Gradio** kütüphanesi ile
interaktif web arayüzü oluşturmayı ve **Hugging Face Spaces**'e yüklemeyi öğretir.

![model-deployment](https://github.com/user-attachments/assets/5cb50653-4102-4b52-a858-c9c0cdbac7d2)

### Takviyeli Öğrenme (Reinforcement Learning)

RL'de bir **ajan**, **çevre** ile etkileşerek ödül/ceza mekanizmasıyla öğrenir.
Bu kursun son modülünde **Deep Q-Network (DQN)** ile CartPole oyunu öğretilir.

![reinforcementlearning](https://github.com/user-attachments/assets/643090dc-b951-48c2-95dd-f6f3fc15079b)

---

## Kurulum

### Sistem Gereksinimleri

- Python 3.8+
- PyTorch 2.0+ (önerilir)
- CUDA 11.8+ (GPU kullanımı için, isteğe bağlı)

### Kütüphane Kurulumu

**Google Colab** — kurulum gerekmez, tüm kütüphaneler önceden yüklüdür.

**Mac (CPU):**
```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib torchinfo tqdm gradio
```

**Windows / Linux (GPU — CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib torchinfo tqdm gradio
```

> **Not:** Windows'ta GPU kullanmak için [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)'i ayrıca yüklemeniz gerekir.
> `!nvidia-smi` komutuyla GPU bilgilerinizi kontrol edebilirsiniz.

---

## Terimler Sözlüğü

| İngilizce | Türkçe |
|-----------|--------|
| Deep Learning (DL) | Derin Öğrenme |
| Machine Learning (ML) | Makine Öğrenmesi |
| Artificial Intelligence (AI) | Yapay Zekâ |
| Supervised Learning | Denetimli Öğrenme |
| Unsupervised Learning | Denetimsiz Öğrenme |
| Transfer Learning (TL) | Transfer / Aktararak Öğrenme |
| Reinforcement Learning (RL) | Takviyeli Öğrenme |
| Convolutional Neural Network (CNN) | Evrişimsel Sinir Ağı |
| Vision Transformer (ViT) | Görüntü Transformatörü |
| Mean Absolute Error (MAE) | Ortalama Mutlak Hata |
| GPU (Graphics Processing Unit) | Grafik İşlemci Birimi |
| Backpropagation | Geri Yayılım |
| Overfitting | Aşırı Öğrenme |
| Epoch | Devir (tüm eğitim verisinin bir geçişi) |
| Batch | Mini-Yığın |
| Loss Function | Kayıp Fonksiyonu |
| Optimizer | Optimize Edici |

---

## Kaynaklar ve Atıflar

Bu proje aşağıdaki açık kaynak çalışmalardan yararlanılarak hazırlanmıştır:

### Ana Kaynak

| Başlık | Yazar | Lisans | Bağlantı |
|--------|-------|--------|----------|
| Zero to Mastery Learn PyTorch for Deep Learning | Daniel Bourke | MIT | [GitHub](https://github.com/mrdbourke/pytorch-deep-learning) · [learnpytorch.io](https://www.learnpytorch.io) |

### Ek Kaynaklar

| Başlık | Kaynak |
|--------|--------|
| Reinforcement Learning Course — Full Machine Learning Tutorial | [freeCodeCamp.org](https://www.freecodecamp.org) |
| Convolutional Neural Networks (CNNs) | [IBM](https://www.ibm.com/topics/convolutional-neural-networks) |
| PyTorch Resmi Dokümantasyonu | [pytorch.org](https://pytorch.org) |
| Gymnasium (RL ortamları) | [gymnasium.farama.org](https://gymnasium.farama.org) |

### Bu Projeyi Nasıl Atıfta Bulunursunuz?

```
Sirvermez, U. (2025). Eğitim Teknolojileri için PyTorch Eğitimi
[Açık Eğitim Kaynağı]. GitHub & Zenodo.
https://github.com/ugursirvermez/PyTorch_Education
doi: 10.5281/zenodo.xxxxxxx
```

Bu proje Daniel Bourke'un çalışması üzerine inşa edilmiştir.
Orijinal çalışmaya da atıf vermeniz beklenmektedir:

```
Bourke, D. (2022). Zero to Mastery Learn PyTorch for Deep Learning.
GitHub. https://github.com/mrdbourke/pytorch-deep-learning
```

---

## Lisans

Bu proje **Creative Commons Atıf-GayriTicari-AynıLisanslaPaylaş 4.0 Uluslararası
(CC BY-NC-SA 4.0)** lisansı ile yayınlanmıştır.

- ✅ Kaynak göstererek özgürce paylaşabilirsiniz.
- ✅ Uyarlayabilir ve yeniden yapılandırabilirsiniz.
- ❌ Ticari amaçla kullanamazsınız.
- ❌ Türev çalışmaları farklı bir lisansla yayınlayamazsınız.

Lisans metni: [creativecommons.org/licenses/by-nc-sa/4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

*Hazırlayan: [Uğur Sırvermez](https://orcid.org/0000-0001-7266-6408) — Bursa Uludağ Üniversitesi*
*Son güncelleme: 2025*
