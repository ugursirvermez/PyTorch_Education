import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  
  # modeli eğitmeye başla
  model.train()
  
  # loss ve accuracy yani kayıp ve tutarlılık miktarı
  train_loss, train_acc = 0, 0

   
  # Donguyu dataloader'dan çalıştır
  for batch, (X, y) in enumerate(dataloader):
      # device ? cpu : gpu -> cihaza yolla
      X, y = X.to(device), y.to(device)

      # 1.Forward etme
      y_pred = model(X)

      # 2. Kaybı hesaplama
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Batch'teki değerleri kümeleyerek hesapla.
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Ortalama kayıp ve tutarlılığı hesapla.
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

  # Modeli Değerlendir
  model.eval() 
  
  # test kayıp ve tutarlılıkları hesaplayacağız
  test_loss, test_acc = 0, 0
  # İçeriği kontrol edeceğiz.
  with torch.inference_mode():
      # DataLoader'daki Batch'i yükle
      for batch, (X, y) in enumerate(dataloader):
          # device ? cpu : gpu -> cihaza yolla
          X, y = X.to(device), y.to(device)
  
          # 1. Forward et
          test_pred_logits = model(X)

          # 2. loss ve acc hesapla
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()
          
          test_pred_labels = test_pred_logits.argmax(dim=1) #logaritmadan çevir.
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
          
  # ortalama test loss ve acc'ı yazdır.
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

#TRAIN_STEP VE TEST_STEP ILE DONGUYU CALISTIR!
def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, epochs: int, device: torch.device) -> Dict[str, List]:
  # İki step'teki sonuçları yazdır.
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }
  
  # train ve test step'lerini döngüyle çalıştır.
  for epoch in tqdm(range(epochs)):
      #Train_Step kısmı
      train_loss, train_acc = train_step(model=model,dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
      #Test_Step kısmı
      test_loss, test_acc = test_step(model=model, dataloader=test_dataloader,loss_fn=loss_fn, device=device)
      
      # Neler olduğunu adım adım yazdır.
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Güncellenen sonuçları yazdir
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # Toplam sonucu ver.
  return results
