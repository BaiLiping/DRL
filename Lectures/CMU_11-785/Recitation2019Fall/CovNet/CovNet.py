import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Net import Network


class ImageDataset(Dataset):
    def __init__(self,file_list,target_list):
        self.file_list=file_list
        self.target_list=target_list
        self.n_class=len(list(set(target_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,index):
        img=Image.open(self.file_list[index])
        img=torchvision.transforms.ToTensor()(img)
        label=self.target_list[index]
        return img, label

def parse_data(datadirectory):
   img_list=[]
   ID_list=[]
   for rootdirectory, subdirectories, filenames in os.walk(datadirectory):
       for i in filenames:
           if i.endswith('.jpg'):
               fileindex=os.path.join(rootdirectory,i)
               img_list.append(fileindex)
               ID_list.append(rootdirectory.split('/')[-1])
   uniqueID_list=list(set(ID_list))
   class_n=len(uniqueID_list)
   target_dict=dict(zip(uniqueID_list,range(class_n)))
   label_list=[target_dict[ID_key] for ID_key in ID_list]
   print('{}\t\t{}\n{}\t\t{}'.format('#Images','#Labels',len(img_list),len(set(label_list))))
   return img_list,label_list,class_n


def init_weights(m):
    if type(m)==nn.Conv2d or type(m)==nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

def train(model, data_loader, test_loader,task='Classification'):
    model.train()
    global numEpochs
    global device
    for epoch in range(numEpochs):
        avg_loss=0.0
        for batch_num, (feats,labels) in enumerate(data_loader):
            feats, labels=feats.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(feats)[1]
            loss=criterion(outputs,labels.long())
            loss.backward()
            optimizer.step()
            avg_loss+=loss.item()

            print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1,batch_num+1,avg_loss))
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss

        if task == 'Classification':
            val_loss, val_acc = test_classify(model, test_loader)
            train_loss, train_acc = test_classify(model, data_loader)
            print('Train Loss: {:.4f}\tTrain Accuracy:{:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:4f}'.format(train_loss,train_acc, val_loss, val_acc))
        else:
            test_verify(model,test_loader)

def test_classify(model, test_loader):
    model.eval()
    test_loss=[]
    accuracy=0
    total=0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        loss = criterion(outputs, labels.long())

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total

def test_verify(model, test_loader):
    raise NotImplementedError














def main():
    img_list,label_list,class_n=parse_data('medium')
    trainset=ImageDataset(img_list,label_list)
    train_data_item,train_data_label=trainset[300]
    dataloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1, drop_last=False)
    imageFolder_dataset = torchvision.datasets.ImageFolder(root='medium/',
                                                       transform=torchvision.transforms.ToTensor())
    imageFolder_dataloader = DataLoader(imageFolder_dataset, batch_size=10, shuffle=True, num_workers=1)
    imageFolder_dataloader = DataLoader(imageFolder_dataset, batch_size=10, shuffle=True, num_workers=1)
    train_dataset = torchvision.datasets.ImageFolder(root='medium/', 
                                                 transform=torchvision.transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, 
                                                   shuffle=True, num_workers=8)
    
    dev_dataset = torchvision.datasets.ImageFolder(root='medium_dev/', 
                                                   transform=torchvision.transforms.ToTensor())
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=10, 
                                                 shuffle=True, num_workers=8)
    global numEpochs
    global num_features
    numEpochs = 4

    num_features = 3
    
    learningRate = 1e-2
    weightDecay = 5e-5
    
    hidden_sizes = [32, 64]
    num_classes = len(train_dataset.classes)
    global device    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = Network(num_features, hidden_sizes, num_classes)
    network.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
    network.train()
    network.to(device)
    train(network, train_dataloader, dev_dataloader)








if __name__=='__main__':
    main()

