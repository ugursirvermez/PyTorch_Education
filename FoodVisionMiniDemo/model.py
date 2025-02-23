import torch
import torchvision

from torch import nn

#Modelin çalışma fonksiyonu
def create_effnetb2_model(num_classes:int=3, 
                          seed:int=42):
    """EfficientNetB2 aracının tasarımı

    Args:
        num_classes (int, optional): sınıf yani ürün çıktı türlerimiz burada.
            en yüksek değer 3'tür.
        seed (int, optional): reastgele seed atıyor. Seed değeri 42.

    Returns:
        model (torch.nn.Module): EffNetB2
        transforms (torchvision.transforms): EffNetB2 resim transform.
    """
    # EffNetB2 eğtilmiş weights, transforms ve model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Tüm temel katmanları dondurulmuş model
    for param in model.parameters():
        param.requires_grad = False

    # Başlıkların değiştirildiği ve seed atıldığı veri seti
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )
    
    return model, transforms
