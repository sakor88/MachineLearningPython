import os
import torch
import torchvision
import tarfile
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader

#%matplotlib inline


# ----------------------    testy i inicjalizacje
data_dir = 'fruits-360/'

#wyswietlanie wszystkich klas
#print(os.listdir(data_dir))
dataset = ImageFolder(data_dir + '/Training', transform=ToTensor())
classes = os.listdir(data_dir + "/Training")


#wyswietlanie wszystkich klas
#print(classes)

#wyswietlanie przykladowego obrazka owocu
#def show_example(img, label):
    #print('Label: ', dataset.classes[label], "("+str(label)+")")
    #plt.imshow(img.permute(1, 2, 0))
    
#show_example(*dataset[34567])

#Przypisanie zmiennej dataset wszystkich obrazkow z folderu Training
dataset = ImageFolder(data_dir + '/Training', transform=ToTensor())
img, label = dataset[0]

#wypisanie wymiarow tensora img
#print(img.shape, classes[label]) 

#Przypisanie zmiennej test wszystkich obrazkow z folderu Test
test = ImageFolder(data_dir + '/Test', transform=ToTensor())
#print('Size of raw dataset :', len(dataset))
#print('Size of test dataset :', len(test))

#inicjalizacja losowosci
random_seed = 8
torch.manual_seed(random_seed);
#wyswietlanie dlugosci training datasetu
#print(len(dataset))

#podzielenie danych treningowych na treningowe i walidacyjne
val_percent = 0.05 
val_size = int(val_percent*len(dataset))
train_size = len(dataset) - val_size

#wyswietlanie wielkosci obu datasetow
train_ds, val_ds = random_split(dataset, [train_size, val_size])
#print(len(train_ds), len(val_ds))

batch_size=128

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
    
#definicja sieci modelu
def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)   
        acc = accuracy(out, labels)           
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        

class Fruits360CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(5, 5),

            nn.Flatten(), 
            nn.Linear(128*5*5, 512),
            nn.ReLU(),
            nn.Linear(512, 131))
            
        
    def forward(self, xb):
        return self.network(xb)


if __name__ == '__main__':
    #show_batch(train_dl)
    
    
        # ----------------------- preprocessing
        
    
    
    """for images, labels in train_dl:
        print('images.shape:', images.shape)
        out = model(images)
        print('out.shape:', out.shape)
        print('out[0]:', out[0])
        break
        """
    
    #uzywamy GPU lub CPU -------------------------------------
    
    
    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
        
    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl: 
                yield to_device(b, self.device)
    
        def __len__(self):
            """Number of batches"""
            return len(self.dl)
        
   # ---------------------------------------------------------------------------
    
    

    model = Fruits360CnnModel()
    
    #Loadowanie do urzadzenia trenujacego (CPU lub GPU)------------------------------------------
    
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)
    
    #Funkcja trenująca --------------------------------------
    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)
    
    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            # Training Phase 
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
        return history
        model = to_device(Fruits360CnnModel(), device)

    #Performance modelu przed trenowaniem
    #print(evaluate(model, val_dl))
    
    
    num_epochs = 3
    opt_func = torch.optim.Adam
    lr = 0.001
    
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    #zapisywanie modelu do transformacji
    #torch.save(model, 'nowymodel.pt')

    #przykladowe przewidywanie dla jednego obrazka
    """
    def predict_image(img, model):
        # Convert to a batch of 1
        xb = to_device(img.unsqueeze(0), device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, preds  = torch.max(yb, dim=1)
        # Retrieve the class label
        return dataset.classes[preds[0].item()]"""
    
    #sprawdzenie
    """
    train = False
    if(train == False):
       # model.torch.state_dict("tu_wpisac_nazwe_modelu_zapisanego_jako_state_dict")
       # model.eval()
        img, label = test[500]
        plt.imshow(img.permute(1, 2, 0))
        print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))"""
       
    
    #Sprzawdznie performance'u dla calego testowego datasetu
    test_dl = DeviceDataLoader(DataLoader(test, batch_size*2), device)
    result = evaluate(model, test_dl)
    print(result)
