import os
import glob
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from PIL import Image
import cv2
import matplotlib.pyplot as plt

from sift_flow_gpu.sift_flow_torch import SiftFlowTorch

from colorbalance import simplest_cb


print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


""" Get Data """
class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx):        
        indices = torch.tensor(end_idx[:-1])
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)

class MyDataset(Dataset):
    def __init__(self, image_paths, seq_length, transform, length):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length

    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        #print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))

        images = []        
        for i in indices:
            Image.open(self.image_paths[i][0]).show()
            imshow(self.transform(Image.open(self.image_paths[i][0])))
            images.append(self.transform(Image.open(self.image_paths[i][0])))
            # """ for colorbalancing """
            # image_path = self.image_paths[i][0]
            # image = cv2.imread(image_path)
            # image = simplest_cb(image,2) # colorbalancing
            # image = Image.fromarray(cv2.cvtColor(image*255,cv2.COLOR_BGR2RGB).astype(np.uint8)) # cv2 to PIL
            # image = self.transform(image)
            # images.append(image)
        targets = torch.tensor([self.image_paths[start][1]], dtype=torch.long)

        x = torch.stack(images)
        y = targets

        return x, y

    def __len__(self):
        return self.length


""" define model """
class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.resnet = resnet
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 300)
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  
            out, hidden = self.lstm(x.unsqueeze(0), hidden)
            
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)

        return x


def get_loader(data_paths):
    """ Iterate through Dateset """

    class_image_paths = []
    end_idx = []

    for (d,c) in data_paths:
        paths = sorted(glob.glob(os.path.join(d.path, '*.png')),
                    key=lambda path: int(str(path).split("_")[-1][:-4]))
        # Add class idx to paths
        paths = [(p, c) for p in paths]
        paths = paths[0::2] # every 2nd element
        class_image_paths.extend(paths)
        end_idx.extend([len(paths)])
      
    end_idx = [0, *end_idx]
    end_idx = torch.cumsum(torch.tensor(end_idx), 0)
   

    seq_length = 8

    sampler = MySampler(end_idx)
    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MyDataset(
        image_paths=class_image_paths,
        seq_length=seq_length,
        transform=transform,
        length=len(sampler))

    loader = DataLoader(
        dataset,
        batch_size=128, 
        sampler=sampler,
        num_workers=0, # can be more worker
        pin_memory=False
    )

    return loader





""" define train & test """
def train(epoch,model,optimizer,train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, torch.flatten(target))
        loss.backward()
        optimizer.step()
        

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.data.item()))
        del output, loss


def test(model,test_loader, epoch, optimizer):
    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        output = model(data)

        loss = criterion(output, torch.flatten(target))
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        c = pred.eq(target.data.view_as(pred)).long().to(device)
        
        correct += c.sum()
        total += len(data)
        del output, loss, c

    accuracy = correct/total
    print(
        '\nTest set: Accuracy: {}/{} ({:.3f}%)\n'.format(
            correct, total, accuracy*100.))
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, "model_checkpoint.pt")


"""Show Images"""
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



if __name__ == '__main__':
    classes = {
        "OOO":0,
        "BOO":1,
        "OLO":2,
        "BLO":3,
        "OOR":4,
        "BOR":5,
        "OLR":6,
        "BLR":7
    }


    """ Get Paths """
    root_dir = 'preprocessed_data/'
    class_paths = dict((c, []) for c in range(8)) # Dictionary with classes as keys and paths as value
    footage_paths = [f for state in os.scandir(root_dir) if state.is_dir()
        for f in os.scandir(state.path) # Footage
        ]

    for f in footage_paths:
        c_name = str(f.path).split("/")[-2]
        c = classes[c_name] # classes[XXX]
        class_paths[c] += [(f,c)]

    train_paths = []
    test_paths = []
    for c in class_paths:
        test_paths += random.sample(class_paths[c],int(len(class_paths[c])/10))
        train_paths += [path for path in class_paths[c] if path not in test_paths]
    random.shuffle(train_paths)
    random.shuffle(test_paths)
    
    test_loader = get_loader(test_paths)
    train_loader = get_loader(train_paths)


    """ Model """
    resnet = torchvision.models.resnet50(pretrained=True)
    model = Combine().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss() # loss function

    for epoch in range(1, 15):
        train(epoch,model,optimizer,train_loader)
        test(model,test_loader,epoch,optimizer)

    """ save model """
    import time, datetime
    ts = time.time() 
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    print(timestamp)

    torch.save(model.state_dict(),"saved_models/model_balanced_" + timestamp)
