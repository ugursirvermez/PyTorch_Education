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
Yapay zekaya doğru giden öğrenme modelleri, Pytorch kurulumu, tensörlerin skaler büyüklüklerden yola çıkarak tanımlanması ve PyTorch kütüphanelerinde kodlanmasına kadar geniş bir kodlama içeriği oluşturduk. Şimdi Kendi nöral ağlarımızı kurmak, derin öğrenme modellerine geçmek için PyTorch’un iş akışı yukarıdaki dosyalarda yer almaktadır.
- PyTorch Workflow, ile ilgili aşağıdaki adımları uygulandı:
    - veri hazırlama ve doldurma,
    - Bir model inşa etme,
    - Modelin içerisine verileri yerleştirme ve eğitme,
    - Tahminler yaptırma ve modelin değerlendirmesi,
    - Modeli kaydetme ve başka yerlerde kullanma.
    - Hepsini bir arada kullanabilme.
![Pytorch workflow](https://github.com/user-attachments/assets/cb0c8cfb-e21f-4440-8acb-5b6fc2bc56a6)

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
**NOT:** Eğer CUDA sürümünü yüklememişseniz lütfen nVidia CUDA Toolkit’inin sürümlerini yükleyiniz. Yukarıdaki !nvidia-smi kodu ile sistem bilginizi kontrol edin! Kurulum sitesine ulaşmak için tıklayın!

## Regresyon Eğrileri (Bütün Örnekler)
![linear_non_linear](https://github.com/user-attachments/assets/a6baafbf-4507-4338-8e66-5fbbf7afb956)

## PyTorch Nöral Ağ Sınıflandırması (NN Classification)
Makine öğrenmesinde önemli bir noktaya doğru giriş yapıyoruz. Derin öğrenme aşamasında doğrusal yaklaşıma önem verdik. Şimdi sınıflandırma aşamasına geliyoruz. Bu sınıflandırma aslında bizim derin öğrenmenin sonucunda karar alma aşamamızın önemli bir parçasını oluşturuyor. Örneğin bir e-postanın spam olup olmadığını nasıl anlarız? Yani bir e-postayı spam yapan nedir? Yahut resimdeki kişi kadın mı yoksa erkek mi olduğuna nasıl karar veririz? Yukarıda verdiğimiz iki örnekte sadece iki seçenekli bir sınıflandırmadan oluşuyor. Spamdır yoksa spam değildir. Erkektir yada kadındır. Bu tip ikili sınıflandırma yani “ya biri ya da diğeridir” şeklinde olan sınıflandırmalara “Binary Classification” denir. Binary denmesinin sebebi 0 ve 1’i temsil eden True/False ikilisini oluşturmasıdır. Bir başka sınıflandırma ise birden fazla seçeneğimiz olan durumlarda kullanılan “Multiclass Classification” bulunmaktadır. Multiclass Classification, birden fazla seçenekte kullanılır. Örneğin resimdeki yemeğin hamburger, pizza veya biftek olması gibi pek çok seçeneğin olduğu durumlarda kullanılır. Multiclass ve Binary sınıflandırmalarına bakıldığında her bir nesnenin (resim, e-posta vs.) sadece bir özelliği var ve tek bir kritere bakılarak sonuca varılır ve kararın oluşmasında etkilidir. Ancak birden fazla durumun ve ihtimalin olduğu durumlarda olunabilir. Örneğin, “Derin Öğrenme” adlı başlıklı makalenin içerisinde yazılanlara bakılarak, makalenin adı değiştirilebilir veya düzenlenebilir. Bu durumda makalenin içerisinde başlığa uyumlu, anlamlı metinlerin elde edilmesi önem kazanır. İçindeki metinleri anlamlandırmak için etiketler kullanılır. Bu etiketler bir veya birden fazla durumla karşılaştırılarak oluşturulabilir. İşte bu tip durumların oluşmasında ve etiket kullanılarak yapılan çok boyutlu sınıflandırmaya “Multilabel Classification” denir. Bu üç tane sınıflandırmayı özetleyelim.

|  Sınıflandırma  | Tanımı |
| ------------- | ------------- |
| Binary Classification  | Binary iki durumdan birisinin seçilmesi yani 0 veya 1 ya da True/False değerlerindeni birini seçmeyi tanımlayan sınıflandırmadır.  |
| Multiclass Classification  | Multiclass Classification, birden fazla seçenekte her bir seçeneğin sadece bir etiket olduğu durumlarda kullanılır.  |
| Multilabel Classification | Bir durumda, birden fazla olasılık ve sınıflandırma için olasılıkların etiketlendiği çok boyutlu karşılaştırmalı sınıflandırmadır. |

![IOClass_Data](https://github.com/user-attachments/assets/19f8c440-a0aa-4c11-a788-d7bbdcee1ad1)

|  Hiper Parametreler  | Binary Classification | Multiclass Classfication |
| ------------- | ------------- | ------------- |
| Girdi Katman Şekli | Özellikleri sayısal olarak listeleme (cinsiyet, yaş, boy, kilo v.b.). | Binary ile aynı özelliktedir. |
| Gizli Katmanlar | Minimum 1, Maksimum sınırsız katman sayısı olabilir. | Binary ile aynı özelliktedir. |
| Her Bir Gizli Katmandaki Nöron Sayısı | 1 sınıf veya farklı herhangi bir şey olmalı. | Her biri ayrı 1 sınıf olur (3 yiyecek, insan veya köpek fotoğrafı v.b.). |
| Çıktı Katmanının Şekli | 1 sınıf veya farklı herhangi bir şey olmalı. | Binary ile aynı özelliktedir. |
| Gizli Katman Aktivasyonu | Genelde ReLU kullanılır, Yapay Nöron Ağlarında Aktivasyon Fonksiyonları kullanılır. |  Binary ile aynı özelliktedir. |
| Çıktı Aktivasyonu | Sigmoid aktivasyonu kullanılır (bunlar açıklanacak). | Softmax aktivasyonu kullanılır. |
| Loss (Kayıp) Fonksiyonu | Binary Cross Entropy kullanılır. | Cross Entropy kullanılır. |
| Optimizer | SGD veya Adam kullanılır. | Binary ile aynı özelliktedir. |

## Multiclass Örneği
![multiclass_final](https://github.com/user-attachments/assets/0ff49211-6e2a-447f-8305-902c6db32916)

## Convolutional Neural Networks Nedir?
Evrişimsel sinir ağları olarak Türkçe’ye çevrilmektedir. Sinir ağları derin öğrenmenin temel yapı taşlarıdır. Bu zamana kadar onları üretmeyi ve birbirlerine istenilen etiketlere uygun nasıl oluşturulabileceğini öğrendik. Bu açıdan bakıldığında her bir nöronun bir kütlesinin ve belirli değerleri taşıdığını biliyoruz. Bu değerlerin katmanlar aracılığı ile aktarılırken uğradıkları değişimler onları yorumlamamızı sağlıyor. Evrişimsel ağlar verinin bu süreç esnasında uğranan değişimleri yorumlamamızı sağlayan temel içeriktir. Yani verinin hangi yönde nasıl bir şekil aldığını katmanlarda uğradığı değişimlerden yola çıkarak yorumluyoruz. Evrişimsel sinir ağlarının en yaygın veri örneği, resim, ses veya ses frekanslarıdır. Bu modelin genel olarak 3 katmanı bulunmaktadır. Bunlar;

- Evrişimsel Katman (Convolutional Layer)
- Havuzlama Katmanı (Pooling Layer)
- Tamamen Bağlı Katman (Fully-Connected (FC) Layer)

şeklinde isimlendirilmektedir. Her birinin hangi PyTorch fonksiyonu ile ne işe yaradığını aşağıdaki tabloda açıklayacağız. Her bir katman aslında ayrı bir araştırma konusudur. Bu konu için [linkteki içeriğe](https://www.ibm.com/topics/convolutional-neural-networks) göz atabilirsiniz. Üretilmiş birçok evrişimsel sinir ağı modeli bulunmaktadır. Hatta bu modelleri IBM şirketi 1980’lerde oluşturmaya başlamıştır. Evrişimsel sinir ağının aşamalarını şu şekilde görselleştirebiliriz:
![CNN_Figure](https://github.com/user-attachments/assets/3484b0c5-441b-48a9-b39c-6201b9359a80)

|  Hiper Parametre/Katman Türü  | Ne İşe Yarar? | Tipik Değeri |
| ------------- | ------------- | ------------- |
| Girdi Resimler | İşinize yarayacak resimlerde ortak yollar keşfedeceğiniz dilediğiniz görseller | Kullanıcının / yazılımcının tercihine dayalı görseller|
| Girdi Katmanı (Input Layer) | Hedeflenen resimleri işlem öncesi katmanlara ayırma | input_shape = [batch_size, image_height, image_width, color_channels]|
| Convolutional Layer (Evrişimsel Katman) | Öğrenme çıktısını oluşturacak önemli özellikleri hedeflenen resimlerden çıkaracak katman | Çoklu işlem hacmi vardır; **torch.nn.ConvXd()** (X → çoklu değerdir) |
| Gizli Aktivasyon /  Doğrusal Olmayan Aktivasyon (Hidden / Non_linear) | Doğrusal olmayan öğrenilmesi gereken özellikler | Genelde **ReLU → torch.nn.ReLU()** kullanılır. |
| Havuzlama Katmanı (Pooling Layer) | Resimlerden öğrenilen bilgilerdeki çok boyutluluğu ortadan kaldırma | **Maksimum → torch.nn.MaxPool2d() veya Ortalama → torch.nn.AvgPool2d()** |
| Çıktı Katmanı / Doğrusal Katman (Output / Linear Layer) | Öğrenilmiş özellikleri ve çıktıları hedeflenen etiketlere yerleştirme | **torch.nn.Linear(out_features = [number_of_classes])** |
| Çıktı Aktivasyonu (Output Activation) | Çıkan logaritmaları tahmin olasılıklarına dönüştürme | **Binary → torch.sigmoid() veya Multiclass → torch.softmax()** |




