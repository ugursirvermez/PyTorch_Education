### 1. Kütüphaneler ve dosyaları dahil edelim.### 
import gradio as gr
import os
import torch
import torchvision

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Sınıf isimlerini çağıralım
with open("class_names.txt", "r") as f: # reading them in from class_names.txt
    class_names = [food_name.strip() for food_name in  f.readlines()]
    
### 2. Modeli ve transformları doldurlaım ###    

# Model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=101, # could also use len(class_names)
)

# Verileri yükleme (Weights)
effnetb2.load_state_dict(
    torch.load(
        f="11_pretrained_effnetb2_food101.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Tahmin yürütme ###
def predict(img) -> Tuple[Dict, float]:

    # Zsmanlayıcıyı başlat
    start_time = timer()
    
    # Transform'a batch ekle
    img = effnetb2_transforms(img).unsqueeze(0)
    
    # Veriyi modele at.
    effnetb2.eval()
    with torch.inference_mode():
        # Olasılıkları geçir
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Etiket tahminlerine bak
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Ortalama zaman jesaplama
    pred_time = round(timer() - start_time, 5)
    
    # Tahmini ve olasılığı geri döndür.
    return pred_labels_and_probs, pred_time

### 4. Gradio Uygulaması ###
title = "FoodVision101 🍔👁"
description = "EfficientNetB2 ve PyTorch'taki food101 veri setinin kullanımını içeren programdır. Burada Veri seti ve model testi yapılmıştır. Eğitim amaçlı olup Uğur Sırvermez tarafından oluşturuldu. [101 sınıfın listesi:](https://github.com/ugursirvermez/PyTorch_Education/blob/main/Module_Files/food101_class_names.txt)."
article = "Esinlenilen makale: [09. PyTorch Model Deployment](https://github.com/mrdbourke/pytorch-deep-learning/tree/main)."

# Örnekler
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Tahminler"),
        gr.Number(label="Tahmin Süresi (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article,
)

# Uygulamayı Yayınlama
demo.launch()
