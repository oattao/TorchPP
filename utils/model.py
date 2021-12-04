import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DImprovement(nn.Module):
    """For the paper: 
       An Improved Fault Diagnosis Using 1D-Convolutional NeuralNetwork Model
    """
    def __init__(self, input_size=1024, num_features=448, num_classes=10, activation='ReLU'):
        super(CNN1DImprovement, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_features = num_features
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Not support this activation: ', activation)

        self.extractor = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=16, stride=1),
            self.activation,
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2, 2),
            nn.Dropout(p=0.3),

            nn.Conv1d(128, 64, kernel_size=8, stride=1),
            self.activation,
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2, 2),
            nn.Dropout(p=0.3),

            nn.Conv1d(64, 32, kernel_size=4, stride=1),
            self.activation,
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2, 2),
            nn.Dropout(p=0.3),

            nn.Conv1d(32, 16, kernel_size=4, stride=1),
            self.activation,
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2, 2),
            nn.Dropout(p=0.3),

            nn.Conv1d(16, 8, kernel_size=4, stride=1),
            self.activation,
            nn.Dropout(p=0.3), 
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 10),
        )


    def forward(self, x):
        x = self.extractor(x)
        batch_size = x.shape[0]
        feature = x.view(batch_size, -1)
        output = self.classifier(feature)

        return output

class CNN1D(nn.Module):
    def __init__(self, input_size=784, kernel_size=9,
                stride=2, num_cnn=4, num_kernel=3, padding=0):
        super(CNN1D,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.nk = num_kernel
        self.num_cnn = num_cnn
        self.input_size = input_size
        self.padding = padding
        self.output_size = self.compute_output()
        self.extract = nn.Sequential(nn.Conv1d(1, self.nk, kernel_size = self.kernel_size, stride = self.stride),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(self.nk),
                                     nn.Conv1d(self.nk,2*self.nk, self.kernel_size, stride = self.stride),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(2*self.nk),
                                     nn.Conv1d(2*self.nk,4*self.nk, kernel_size = self.kernel_size, stride = self.stride),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(4*self.nk))
        self.classify = nn.Sequential(nn.Linear(1092, 512),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),
                                      nn.Linear(512,10))
    def compute_output(self):
        w = self.input_size
        s = self.stride
        f = self.kernel_size
        for i in range(self.num_cnn):
            w = (w-f) // s+1
        return w

    def forward(self,sig):
        feature = self.extract(sig)
        size = feature.shape[-1]
        feature = feature.view(-1,1092)

        return self.classify(feature)

class GrayNet(nn.Module):
    def __init__(self, num_classes):
        super(GrayNet,self).__init__()
        self.num_classes = num_classes
        self.gray = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2))
        self.full = nn.Sequential(nn.Linear(16*7*7, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.num_classes))

    def forward(self,gray_img):
        sc = self.gray(gray_img)
        sc = sc.view(-1,16*7*7)
        return self.full(sc)

class ScaloNet(nn.Module):
    def __init__(self, num_classes):
        super(ScaloNet, self).__init__()
        self.num_classes = num_classes
        self.scal = nn.Sequential(nn.Conv2d(3,8,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2,2),
            nn.Conv2d(8,16,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2))
        self.full = nn.Sequential(nn.Linear(128*7*7,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, self.num_classes))

    def forward(self, scal_img):
        sc = self.scal(scal_img)
        sc = sc.view(-1, 128*7*7)
        return self.full(sc)

class WideNet(nn.Module):
    def __init__(self, num_classes):
        super(WideNet, self).__init__()
        self.num_classes = num_classes
        self.gray_branch = GrayNet(num_classes)
        self.scalo_branch = ScaloNet(num_classes)

        self.full = nn.Sequential(nn.Linear((128+16)*7*7,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,10))
        
    def forward(self, scal_img, gray_img):
        sc = self.scalo_branch.scal(scal_img)
        gr = self.gray_branch.gray(gray_img)
        sc = sc.view(-1,128*7*7)
        gr = gr.view(-1,16*7*7)
        sm = torch.cat((sc,gr),1)
        sm = sm.view(-1,(128+16)*7*7)
        return self.full(sm)