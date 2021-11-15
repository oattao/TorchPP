import os
import copy
import glob

import cv2 as cv
from sklearn.model_selection import train_test_split

import torch
from torchvision.transforms import ToTensor
from torchvision.models import alexnet

from utils.model import GrayNet
from utils.data import ImageDataset
from config import label_class_dict, datapath, horse_powers, noise_levels, RDS


# parameter
spliter = ['train', 'test']
num_epoch = 47
model_name = 'Alexnet'

num_class = 10
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# prepare data
hp = 0
nlv = 'nonoise'

datapath = datapath.format(hp, nlv)
print(datapath)

train_list, test_list = [], []
subfolder = os.listdir(datapath)

for fname in subfolder:
    list_all_file = glob.glob(datapath + fname + '/scalo/*.png')
    _train, _test = train_test_split(list_all_file, test_size=0.3)
    train_list.extend(_train)
    test_list.extend(_test)

img_list = {'train': train_list, 'test': test_list} 
print(len(img_list['train']), len(img_list['test']))

datasets = {tp: ImageDataset(img_list[tp], label_class_dict, cv.IMREAD_COLOR, ToTensor())
            for tp in spliter}
dataloaders = {tp: torch.utils.data.DataLoader(datasets[tp], batch_size=batch_size, 
                                               shuffle=True)
               for tp in spliter}

model = alexnet(pretrained=True)
for param in model.parameters():
    param.requires_grad = True
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, num_class)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

print('Training...')
best_acc = 0.0

for epoch in range(num_epoch):
    print('Epoch {}/{}'.format(epoch, num_epoch-1))
    for phase in spliter:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        running_correct = 0

        for image, label in dataloaders[phase]:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase=='train'):
                outputs = model(image) 
                _, pred = torch.max(outputs, 1)
                loss = criterion(outputs, label)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * image.size(0)
            running_correct += torch.sum(pred == label.data)

        epoch_loss = running_loss / len(datasets[phase])
        epoch_acc = running_correct.double() / len(datasets[phase])

        if phase == 'train':
            exp_lr.step()

        if phase == 'test' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))    
  
model.load_state_dict(best_model)
