# ReinForcement Learning: CartPole Oyunu
Bu bölüme kadar Supervised ve Unsupervised Learning’te etiketleme ve görselleri işleme konusunda epey işlem yaptık. 
Makine öğrenmesinin bir başka alt dalı olan ve çok fazla tercih edilmeyen “Reinforcement Learning” konusuna değineceğiz. 
Reinforcement Learning’in Türkçe karşılı takviyeli öğrenme veya güç kullanarak öğrenme olarak tanımlanabilir. 
Bu konuda en sık kullanılan “Deep Q Learning” kavramı üzerinde duracağız. Konulara başlamadan önce Python’da kurulu olması gereken kütüphanelere göz atalım.
- NOT: Bölüm 14’teki eğitim yine freeCodeCamp.org tarafından sunulan “Reinforcement Learning Course - Full Machine Learning Tutorial” adlı eğitimden yola çıkarak hazırlanmıştır.
- Devamında uygulanan eğitim farklı araçlarla ve geliştiricilerle desteklenmiştir. Kaynakları Colab dosyalarında yer almaktadır.

Bütün takviyeli öğrenme sürecine başlamadan önce yukarıdaki kütüphanelerden birinin üzerinde durarak aslında bütün bu süreci daha iyi anlamış olacağız. Bize bütün bu süreci daha iyi ortaya koyacak olan kütüphane Gymnasium kütüphanesidir.
Gymnasium (Gym), en temelinde makine öğrenmesi modeline bağlı birimlerin çevre ile etkileşiminden veri toplayan ve onlara uygun çevreleri tasarlayan bir kütüphanedir.![reinforcementlearning](https://github.com/user-attachments/assets/643090dc-b951-48c2-95dd-f6f3fc15079b)
Görselde görüldüğü üzere takviyeli öğrenme (RL) sürecini görebiliriz. Birim (Agent), oyundaki bir karakterimiz veya veri toplayan bir nesne olarak görebiliriz.
Bu birim, ortamda bazı eylemler gerçekleştirir (genellikle ortama bazı kontrol girdileri geçirerek, örn. motorların tork girdileri) ve ortamın durumunun nasıl değiştiğini gözlemler.
Bu tür eylem-gözlem değişimlerinden birine zaman adımı (timestep) denir. RL'deki amaç, ortamı (Çevre - Environment) belirli bir şekilde manipüle etmektir. 
Örneğin, birimin bir robotu uzayda belirli bir noktaya yönlendirmesini istiyoruz. Bunu başarırsa (veya bu hedefe doğru bir miktar ilerleme kaydederse), bu zaman adımı için gözlemle birlikte pozitif bir ödül alacaktır. 
Ödül, ajan henüz başarılı olamadıysa (veya herhangi bir ilerleme kaydedemediyse) negatif veya 0 da olabilir. Daha sonra birim, birçok zaman adımında biriktirdiği ödülü en üst düzeye çıkarmak için eğitilecektir. 
Bazı zaman adımlarından sonra, ortam bir son duruma girebilir. Örneğin, birim olarak kulanılan robot artık çalışamıyor olabilir. Bu durumda, ortamı yeni bir başlangıç durumuna sıfırlamak istiyoruz. Ortam, birim böyle bir son duruma girerse ona bir tamamlandı sinyali gönderir. 
Tüm tamamlandı sinyallerinin "felaket niteliğinde bir arıza" tarafından tetiklenmesi gerekmez: Bazen, belirli sayıda zaman adımından sonra veya etken ortamda bir görevi tamamlamayı başardığında da bir tamamlandı sinyali göndermek isteriz. İşte buna benzer işlemlerle bir RL eğitim döngüsü oluşturmuş oluyoruz.

![RL-Test](https://github.com/user-attachments/assets/fbae8692-54e7-46e1-ba4b-606273c497bf)
![CartPolev1](https://github.com/user-attachments/assets/270c25ed-71bd-40e3-848f-6c4ce4e20ac9)
![CartPolev2](https://github.com/user-attachments/assets/a62ab3bd-d956-4029-b12b-ff531bbef133)

