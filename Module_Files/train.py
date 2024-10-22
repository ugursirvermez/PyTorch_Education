import os
import torch
import data_setup, engine, model_builder, utils #Bütün oluşturduğumuz dosyaları alıyoruz.
from torchvision import transforms

# Girdi Parametrelerimizi oluşturuyoruz. -> HiperParametre
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Klasörleri hazırladık.
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Cihaz CPU mu yoksa GPU mu buna karar veriyoruz.
device = "cuda" if torch.cuda.is_available() else "cpu"

#Transform Oluşturma
data_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

#data_setup.py ile DataLoader oluşturma
#3 Parametre döndürüyordu. train ve test dataloader ile etiketlerdi.
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir, 
                                                                               transform=data_transform,batch_size=BATCH_SIZE) #HiperParametre var.

#model_builder.py ile birlikte modeli oluşturuyoruz.
model = model_builder.TinyVGG(input_shape=3, hidden_units=HIDDEN_UNITS,output_shape=len(class_names)).to(device)#HiperParametre var.

#loss ve optimizer hazırlama
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE) #HiperParametre var.

# engine.py ile eğitimi başlatabiliriz.
engine.train(model=model, train_dataloader=train_dataloader,test_dataloader=test_dataloader,loss_fn=loss_fn, optimizer=optimizer,
             epochs=NUM_EPOCHS, device=device) #HiperParametre var.

#utils.py ile modelin son halini kaydetme
utils.save_model(model=model, target_dir="models", model_name="07_modular_script_mode_tinyvgg_model.pth")
