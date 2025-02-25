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
- Elimizde farklı türlerden veriler var. Bu veriler resim, ses kaydı, metin veya pek çok şey olabilir. Ben bunları önce sayıya çeviriyorum. Belirli kalıplara sokarak onları listeliyorum ve ilgili bir dizinin veya bir setin içerisine yerleştiriyorum. İşte bu sayısal yani dijitalleştirilen verilerin ne olduğunu bilgisayar çok iyi anlayabiliyor. Yani birbirleriyle ilişkili olup olmadığını ölçmeye, anlamlandırmaya hazır hale geliyor.
- İşte bu sayıya çevirdiğimiz verilerle biz problemimizin çözümüne doğru bir yaklaşım belirliyoruz. Bu yaklaşım sayesinde eldeki verileri birbirleriyle karşılaştırıyor, birbirleriyle ilişkilendiriyoruz. Bu ilişkilerden anlamlı bir sonuç elde edebiliyorsak bunu öğreniyoruz. Eğer elde edemiyorsak bu seçeneği eliyoruz ve diğer yolların güvenilirliğine inanıyoruz. En doğru sonuca ulaşana kadar benzer süreçleri takip ediyoruz.
- En sonunda elimizde belirli sonuçlar ortaya çıkıyor. Bunlar bizim yaklaşımımıza uygunsa bunun geçerli bir cevap olduğunu gözlemliyoruz.

## PyTorch Akışı ve Temel Kütühaneler
| TorchVision Kütüphaneleri  | Ne İşe Yarar? |
| ------------- | ------------- |
| torchvision.datasets  | Veri setlerini alma ve verileri fonksiyonlara doldurmak için kullanılan fonksiyonları içinde barındırır.  |
| Content Cell  | Content Cell  |
