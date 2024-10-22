import torch
from pathlib import Path

#Modeli Kaydetme İşlemi
def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
   #Hedeflenen klasörü ve yolu oluşturma
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  #Modelin kaydedileceği yolu oluşturma
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name sonunda '.pt' veya '.pth' ile bitmelidir!"
  model_save_path = target_dir_path / model_name

  #state_dict() ile son durumu güncelleme
  print(f"[INFO] Modelin Kaydedildiği Yer: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)
