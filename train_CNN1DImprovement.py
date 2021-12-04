import os
import copy
import glob
import torch
from datetime import datetime
import numpy as np

from utils.model import CNN1DImprovement
from utils.data import CNN1DImprovementDataset
from utils.signal import read_mat_file, add_noise

#-----------------------------------------------------------------------------------
# Change these parameters
hp = 0     # horse power
nlv = -2      # noise level

num_epoch = 50
# activation = 'Tanh'
activation = 'ReLU'

sample_length = 512  # 1024, 784, 512

learning_rate = 0.001
#-----------------------------------------------------------------------------------

data_path = os.path.join('.', 'data', 'original_signals')
fault_types = ['NoFault', 
               'Inner7', 'Inner14', 'Inner21',
               'Ball7', 'Ball14', 'Ball21', 
               'Outer7', 'Outer14', 'Outer21']
horse_power_0_files = [97, 105, 118, 130, 169, 185, 197, 209, 222, 234] 

# parameter
spliter = ['train', 'test']
model_name = 'CNN1DImprovement'

num_class = 10
batch_size = 8
test_size = 0.2

# Read file
data = {}
for label in range(len(horse_power_0_files)):
    file_name = horse_power_0_files[label] + hp
    file_path = os.path.join(data_path, f'{file_name}.mat')
    signal = read_mat_file(file_path)
    if nlv != 0:
        signal = add_noise(signal, nlv)
    data[label] = signal

# Prepare dataset
dataset = CNN1DImprovementDataset(data, sample_length)
num_samples = len(dataset)
num_train = int(num_samples*test_size)
num_test = num_samples - num_train
dataset_train, dataset_test = torch.utils.data.random_split(dataset, [num_train, num_test])

datasets = {'train': dataset_train,
            'test': dataset_test}
dataloaders = {tp: torch.utils.data.DataLoader(datasets[tp], batch_size=batch_size, shuffle=True)
               for tp in spliter}

# Prepare model
# num_features : 1024-448, 512-192, 784-328
if sample_length == 1024:
    num_features = 448
elif sample_length == 784:
    num_features = 328
elif sample_length == 512:
    num_features = 192

model = CNN1DImprovement(num_features=num_features, activation=activation)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
exp_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

print('Training...')
best_acc = 0.0
best_loss = np.inf

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
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase=='train'):
                outputs = model(image) 
                _, pred = torch.max(outputs, 1)
                loss = criterion(outputs,label)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * image.size(0)
            running_correct += torch.sum(pred == label)

        epoch_loss = running_loss/len(datasets[phase])
        epoch_acc = running_correct.double()/len(datasets[phase])

        if phase == 'train':
            exp_lr.step()

        if phase == 'test' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_loss = running_loss
            best_model = copy.deepcopy(model.state_dict())

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))    

# save test results
with open("./log/train_cnn1d_improvement.txt", 'a') as f:
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    f.write('-'*20 + now + '-'*20 + '\n')
    f.write(f'Num epochs: {num_epoch}\n')
    f.write(f'Activation: {activation}\n')
    f.write(f'Horse power: {hp}\n')
    f.write(f'Noise level: {nlv}\n')
    f.write(f'Best loss: {best_loss}\n')
    f.write(f'Best accuracy: {best_acc}\n')
  
# model.load_state_dict(best_model)
