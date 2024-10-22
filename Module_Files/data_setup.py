#going_modular adında olusturdugumuz dosyanın üzerine bu kodu data_setup.py olarak yaz.
import os
import zipfile
from pathlib import Path
import requests

import torch
#Torchvision Kutuphaneleri
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

#-------------------------------------------------------------------------------------------------------------
#VERİ SETLERİ

# Dosyaların cikacagi yolu ayarla
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# Eger dosya yoksa indirmeye basla
if image_path.is_dir():
    print(f"{image_path} dosya zaten var.")
else:
    print(f"{image_path} Dosyası olusturuluyor")
    image_path.mkdir(parents=True, exist_ok=True)

# Yemek bilgilerini indir
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("İndiriliyor...")
    f.write(request.content)

# Dosyaları zipten cikar
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Sıkıştırılmış dosyada çıkarılıyor...")
    zip_ref.extractall(image_path)

# Zip dosyasini sil.
os.remove(data_path / "pizza_steak_sushi.zip")

# train ve test degiskenlerine dosyalari at.
train_dir = image_path / "train"
test_dir = image_path / "test"

# Transform ile verileri donustur.
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# ImageFolder'dan bir train_data olustur. Data Load etme işlemi
train_data = datasets.ImageFolder(root=train_dir, # hedeflenen görseller
                                  transform=data_transform, # data_transform şekline çevir
                                  target_transform=None) # Etikete özel dnosutrme yapılmayacak.

#Test_data benzer şekilde olusturuldu.
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}") #veriler geldi mi?
#--------------------------------------------------------------------------------------------------------

#DATA_SETUP.PY BURADA BASLIYOR
#İşlemci sayilari
NUM_WORKERS = os.cpu_count()

#DataLoader Fonksiyonu DataSet -> DataLoader
#Train, test ve sınıfların etiketlerini deger olarak dondurecek bir fonksiyondur.
def create_dataloaders(
    train_dir: str, #Train Klasoru
    test_dir: str, #Test Klasoru
    transform: transforms.Compose, #transform dosyalarını bir araya getir. (train ve test)
    batch_size: int, #DataLoader'dan ne kadar batch yapılacak?
    num_workers: int=NUM_WORKERS #DataLoader'da kaç işlemci çalışacak?
):
   # ImageFolder Kullanarak dataset olusturma
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Train_data'da etiketlenmiş verileri al.
  class_names = train_data.classes

  # Resimleri DataLoader'a donustur
  train_dataloader = DataLoader(train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names #Loader dosyaları ve etiketler dondursun.

#data_setup.py ile dosyaları DataLoader'ı kullanabiliriz.
