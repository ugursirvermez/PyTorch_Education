import torch
import torchvision

from torch import nn

#EffnetB2 Modeli Oluşturma
def create_effnetb2_model(num_classes:int=3,
                          seed:int=42):

    # 1, 2, 3. EffNetB2 eğitilmiş weights, transforms ve model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    # 4. Temel Katmanları Dondurma
    for param in model.parameters():
        param.requires_grad = False

    # 5. Tekrarlanabilirlik için sınıflandırma başlığını rastgele seed atma
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )

    return model, transforms
