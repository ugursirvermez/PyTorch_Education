import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):


    # Resmi aç.
    img = Image.open(image_path)

    # Resmin dönüşümü sağlanmamışsa sağla.
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # Device'a resmi yönlendireceğiz
    model.to(device)

    # Eval modunu başlat.
    model.eval()
    with torch.inference_mode():
      #Ekstra boyutlar ekleniyor (batchsize, color, height,width)
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # Ekstra boyutlarla birlite araca yolla.
      target_image_pred = model(transformed_image.to(device))

    #Logaritmaları olasılığa çevir (torch.softmax() -> multi-class)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Olasılıkları tahmin etiketlerine çevir.
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Sonuçları görselleştir.
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False);
