# PyTorch Eğitimi 🔥

Bu dersi Youtube’taki 25 saatlik bir içerikten alıyorum. İçerik, FreeCodeCamp.org  adlı kuruma aittir. İçeriği Daniel Bourke (@mrdbourke), tarafından hazırlanan eğitim içeriğinden faydalanarak kendi öğrendiklerimi yazdığım bir repo oluşturdum. Oradaki içeriği özümseyip kendi çapımda yorumlayıp bir ders notu haline getirdim. Ayrıca aşağıda eğitim boyunca not aldığım ve test ettiğim kodların yer aldığı Github repo’su yer almaktadır.
**NOT:** Oluşturduğum notların küçük bir kısmı burada yer almaktadır.
## Sözlük
- Deep Learning → DL → Derin Öğrenme
- Machine Learning → ML → Makine Öğrenmesi
- Artificial Intelligence → AI → Yapay Zeka
- Supervised Learning → SL → Denetlenen Öğrenme
- Unsupervised & Self-supervised Learning → USL → Denetlenmeyen ve Kendini Denetleyen Öğrenme
- Aktararak Öğrenme → TL → Transfer Learning
- Reinforcement Learning → RL → Güç Kullanarak Öğrenme
- Graphic Processing Unit → GPU → Grafik İşlemci Birimi
- Tensor Processing Unit → TPU → Tensör İşlemci Birimi
- Compute Unified Device Architecture → CUDA → Birleşik Cihaz Mimarisi Hesaplama
- Peripheral Component Interconnect-Express → PCI-E → Çevresel Bileşen Ara Bağlantısı-Ekspress
- Mean Absolute Error → MAE → Ortalama Mutlak Hata
- Application Programming Interface → API → Uygulama Programlama Arabirimi
- Convolutional Neural Networks → CNN → Evrişimsel Sinir Ağları
- VisionTransformer → ViT → Ana Sinir Ağı Mimarisi
- VisionTransformer Paper → ViT Paper → Ana Sinir Ağı Mimarisini Tanıtan Makale
- Image is Worth 16x16 Words → Ölçekte Görüntü Tanıma için Transformatörler


## PyTorch Nedir? Deep Learning Ne İşe Yarar?
Öncelikle makine öğrenmesi nedir buna bakmak lazım. Makine öğrenmesi verileri(resim, metin vs.) sayılara ve bu sayıların içerisinde bir örüntü bulmak demektir. 

> *Deep Learning (DL) → Machine Learning (ML) → Artifical Intelligence (AI) demektir.*
> 

Bu kurs sayesinde DL ve ML arasında bir bağlantı kuracağız. DL’i pyTorch ile sağlayacak bir sistem geliştireceğiz. 

- Öncelikle geleneksel programlama elimizdeki verileri sıralı adımlara koyup harika bir çıktı elde etme çalışmamızdan ibaret.
- Ancak makine öğrenmesi algoritması, elimizdeki şeylerle nasıl ve ne gibi güzel içerikler üretebileceğimizi içeren bir algoritma yapısıdır. Yani ben size yemek malzemeleri vereceğim ve aynı zamanda sizden o malzemelerle yapılabilecek en güzel yemeği isteyeceğim. Sizde sonuç olarak bana bu yemeğin nasıl yapılacağının yollarını bulacaksınız. İşte bu kadar basit!
- ML bütün olasılıkları hesaplama, belirli bir örüntü kurarken bizim neleri atladığımızı bize verebilen bir yapıdır. Elinizdeki veri ne kadar iyiyse siz o verilerin ışığında en doğru yola ulaşabilirsiniz.
- Basit kurallı bir sistem kurulabilir ancak bu o kadar basit bir şey değil. Elinizdeki verilerle çözmek istediğiniz problemin aşamalarının ne kadar yeterli olduğunu bilecekseniz gayet mantıklı bir çözüm. Ancak probleminizi çözecek verileriniz fazla ve bu problemin çözüm aşamaları uzun kurallarla doluysa işte burada ML size yardımcı oluyor.
- Ayrıca problem durumu yeni senaryolarda veya farklı durumlarda tekrar ediyorsa basit bir kuralla çözemezsiniz. Bunun için yine ML’ye ihtiyaç duyacaksınız.
- Farklı sonuçlarda elde etmek, farklı çıktıların nasıl sonuçlanacağını bilmek isterseniz yine size ML yardımcı olacaktır.
- **Makine öğrenmesi bir sürecin nasıl açıklanması gerektiğini bilmek istediğimizde, gerçekten basit bir sistemimiz varsa ve bu kadar karmaşık sonuçlar istemiyorsak ayrıca yaptığımız işte kesin, hatasız sonuçların oluşmasını istiyorsak ve elimizde çok veri yoksa ML kullanmamıza gerek yoktur.**


## Nöral Ağlar Nedir?
![Neural Network](https://github.com/user-attachments/assets/2210f8e9-2851-4252-b7bb-fe1ca6f2924c)
- Elimizde farklı türlerden veriler var. Bu veriler resim, ses kaydı, metin veya pek çok şey olabilir. Ben bunları önce sayıya çeviriyorum. Belirli kalıplara sokarak onları listeliyorum ve ilgili bir dizinin veya bir setin içerisine yerleştiriyorum. İşte bu sayısal yani dijitalleştirilen verilerin ne olduğunu bilgisayar çok iyi anlayabiliyor. Yani birbirleriyle ilişkili olup olmadığını ölçmeye, anlamlandırmaya hazır hale geliyor.
- İşte bu sayıya çevirdiğimiz verilerle biz problemimizin çözümüne doğru bir yaklaşım belirliyoruz. Bu yaklaşım sayesinde eldeki verileri birbirleriyle karşılaştırıyor, birbirleriyle ilişkilendiriyoruz. Bu ilişkilerden anlamlı bir sonuç elde edebiliyorsak bunu öğreniyoruz. Eğer elde edemiyorsak bu seçeneği eliyoruz ve diğer yolların güvenilirliğine inanıyoruz. En doğru sonuca ulaşana kadar benzer süreçleri takip ediyoruz.
- En sonunda elimizde belirli sonuçlar ortaya çıkıyor. Bunlar bizim yaklaşımımıza uygunsa bunun geçerli bir cevap olduğunu gözlemliyoruz.
![Neural Network Example](https://github.com/user-attachments/assets/35307c35-4e71-46e8-92c0-1455343ad974)

## PyTorch Akışı ve Temel Kütühaneler
| TorchVision Kütüphaneleri  | Ne İşe Yarar? |
| ------------- | ------------- |
| torchvision.datasets  | Veri setlerini alma ve verileri fonksiyonlara doldurmak için kullanılan fonksiyonları içinde barındırır.  |
| torchvision.models  | Önceden eğitilmiş Computer Vision modelleri kullanarak kendi algoritmalarınıza dökmenizi sağlar.  |
| torchvision.transforms | Verilerinizi kendi Computer Vision’ınıza uygulayarak (manipüle ederek) ML için uygun hale getiren fonksiyonları kapsar. Görselleri sayılara çevirir. |
| torch.utils.data.Dataset | Dataset sınıfının temel PyTorch sınıfıdır. |
| torch.utils.data.DataLoader | Veri setlerini Python nesnelerine dönüştürmedir. |

## PyTorch ve Yardımcı Kütüphanelerin Kurulumu
- PyTorch’u çalıştırmak için öncelikle PyTorch kurmamız lazım. Zekice! O zaman kurulum için şunları yapmak uygun olacaktır:
- [colab.research.google.com](http://colab.research.google.com) adresine giriyoruz. Bu adres, Github’taki projelere ulaşmamızı da sağlıyor. Github ağında çok fazla zaman geçiriyorsanız kullanabilirsiniz.
- Açılınca yeni bir not defteri oluşturuyoruz ve ismini “01_pyTorch” olarak değiştiriyoruz.
- Yukarıdaki menülerden birinde Runtime (Çalışma Zamanı) yazıyor. Tıklıyoruz.
- Çalışma Zamanı Türünü Değiştir (Change Runtime Type) butonuna basıyoruz. Standard ayarda GPU seçeneğini seçiyoruz.
- Kod olarak aşağıdakini yazıyoruz;
```PowerShell
!nvidia-smi 
//Bu kod ekranı kartının hangi PCI-E yola sahip //PCI-E: RAM ve ekran kartlarının ana kartta kullandıkları yol türüdür. Alt versiyonları vardır.
//olduğunu belirtmektedir. Yapmak gerekli değil.
```
- Kütüphaneyi entegre ediyoruz.
```Python
#Torch Kütüphanesi
import torch #torch kütüphanesi
#Pandas, veri işleme ve veri analizi için kullanılan bir kütüphanedir. 
#Kütüphane temel olarak zaman etiketli serileri ve sayısal tabloları işler.
import pandas as pd 
#NumPy, büyük, çok boyutlu dizileri ve matrisleri destekleyen, bu diziler üzerinde çalışacak üst düzey matematiksel işlevler ekleyen bir kitaplıktır.
import numpy as np
#Matplotlib, NumPy için yapılmış bir çizim kitaplığıdır.
import matplotlib.pyplot as plt
print(torch.__version__)
```
- Devam edebilmek için yerel PyTorch kurulumu yapılmalıdır. Kurulum için [pytorch.org](http://pytorch.org) sitesine girip bahsettiği adımlara bakabiliriz. Windows kurulumunda CUDA yüklemesine ayrıca gerek duyulmaktadır. Lütfen PyTorch kütüphanesini kurmadan önce Python programlama dilini bilgisayarınıza yükleyin!
- **Mac** kurulumu için;
```PowerShell
-Torch Kurulumu-
pip3 install torch torchvision torchaudio
veya
conda install pytorch::pytorch torchvision torchaudio -c pytorch
-Numpy Kurulumu-
conda install numpy
-Pandas Kurulumu-
conda install pandas
-Matplotlib Kurulumu-
conda install matplotlib
```
- **Windows** kurulumu için;
```PowerShell
-Torch Kurulumu-
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/wh1/cu121
-Numpy Kurulumu-
pip install numpy
-Pandas Kurulumu-
pip install pandas
-Matplotlib Kurulumu-
python -m pip install -U matplotlib
```
**NOT:**< Eğer CUDA sürümünü yüklememişseniz lütfen nVidia CUDA Toolkit’inin sürümlerini yükleyiniz. Yukarıdaki !nvidia-smi kodu ile sistem bilginizi kontrol edin! Kurulum sitesine ulaşmak için tıklayın!
