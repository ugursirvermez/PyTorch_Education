### 1. Sınıfları ve kütüphaneleri ekleme ### 
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# sınıf isimlerini belirleme
class_names = ["pizza", "steak", "sushi"]

### 2. Model ve transform hazırlama ###

# EffNetB2 modelini oluştur
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=len(class_names),
)

# weight kaydet
effnetb2.load_state_dict(
    torch.load(
        f="09_effnetb2_pizza_steak_sushi_20_percent.pth",
        map_location=torch.device("cpu"),  # CPU
    )
)

### 3. Tahmin Fonksiyonu ###
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Zamanlayıcıyı başlat.
    start_time = timer()
    
    # Resmi ekle ve batch boyutu aç.
    img = effnetb2_transforms(img).unsqueeze(0)
    
    # Modeli çalıştır.
    effnetb2.eval()
    with torch.inference_mode():
        # Modeldeki resmin olasılıklarını değişkene aktar.
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    #Resmin sınıflardan hangisine ait olduğuna dair tahminleri ve süreleri al
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Tahmin Süresini hesapla
    pred_time = round(timer() - start_time, 5)
    
    # etiketleri ve süreyi yeniden hesapla. 
    return pred_labels_and_probs, pred_time

### 4. Gradio Uygulaması (Arayüz) ###

# Başlık vs. içerik bilgisini oluştur.
# Create title, description and article strings
title = "FoodVision Mini 🍕🥩🍣"
description = "EfficientNetB2 modeli resmin pizza, biftek veta suşi olup olmadığını kontrol ediyor (mrdbourke tarafından hazırlandı). Ben Uğur Sırvermez eğitim amaçlı, konuyu öğrenmek için bütün bunları tekrar hazırladım."
article = "İlham Alınan Çalışma: [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

# "examples/" yolundan örnekleri getir.
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Gradio Demosu
demo = gr.Interface(fn=predict, # girdi ve çıktıyı ayarlayan temel fonksiyon
                    inputs=gr.Image(type="pil"), # girdilerimiz
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # çıktılarımız
                             gr.Number(label="Prediction time (s)")], # predict fonksiyonu iki değer döndürüyor
                    examples=example_list, #Veri listemiz
                    title=title, #Başlık
                    description=description, #Açıklamalar
                    article=article) #Çalışmanın nereden alındığını yayınlama

# Demo'yu sunma
demo.launch()
