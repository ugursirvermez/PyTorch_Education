# Genel Bilgi
Bütün bu 5 dosya PyTorch kütüphanelerinin temel mantığını çalıştırmaktadır. FoodVision Mini projesi için yazılmış TinyVGG modelinin 
aşağıdaki PyTorch akış şemasındaki gibi çalışmaktadır.
![pytorch_workflow_funcs](https://github.com/user-attachments/assets/bcdd97af-28ba-42ed-9b29-ab4489b98eb3)
## Dosyalar Ne İşe Yarıyorlar?
- **data_setup.py** → Verilerimizi Transform ile dönüştürüp DataLoader’a yüklediğimiz kodlar yer almaktadır.
- **engine.py** → Train_Step, Test_Step ve Train fonksiyonlarının olduğu eğtimin başlatıldığı kodlar yer alır.
- **model_builder.py** → Modelin kendisinin yer aldığı .py dosyasıdır.
- **utils.py** → Modelin sonuçlarını güncellemeyi ve modeli kaydetmeyi sağlayan kodlar yer almaktadır.
- **train.py** → Train ise bütün dosyaların yer aldığı verinin işlenmesini sağlayan bütün .py dosyalarını çalıştıran kodlar yer almaktadır.

Daha fazla bilgi için 01_Pytorch.ipynb adlı dosyadan başlayarak bütün eğitim kodlarına bakabilirsiniz.
