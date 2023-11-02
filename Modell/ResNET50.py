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


"""sift algorithm"""
def find_local_matches(desc1, desc2, kernel_size=9):
    # Computes the correlation between each pixel on desc1 with all neighbors
    # inside a window of size (kernel_size, kernel_size) on desc2. The match
    # vector if then computed by linking each pixel on desc1 with
    # the pixel with desc2 with the highest correlation.
    #
    # This approch requires a lot of memory to build the unfolded descriptor.
    # A better approach is to use the Correlation package from e.g.
    # https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package
    desc2_unfolded = F.unfold(desc2, kernel_size, padding=kernel_size//2)
    desc2_unfolded = desc2_unfolded.reshape(
        1, desc2.shape[1], kernel_size*kernel_size, desc2.shape[2], desc2.shape[3])
    desc1 = desc1.unsqueeze(dim=2)
    correlation = torch.sum(desc1 * desc2_unfolded, dim=1)
    _, match_idx = torch.max(correlation, dim=1)
    hmatch = torch.fmod(match_idx, kernel_size) - kernel_size // 2
    vmatch = match_idx // kernel_size - kernel_size // 2
    matches = torch.cat((hmatch, vmatch), dim=0)
    return matches


def warp(x,flo):
    # Warps Image
    B,H,W,C = x.size()
    # mesh grid
    xx = torch.arange(0,W).view(1,-1).repeat(H,1)
    yy = torch.arange(0,H).view(-1,1).repeat(1,W)

    xx = xx.view(1,H,W,1).repeat(B,1,1,1)
    yy = yy.view(1,H,W,1).repeat(B,1,1,1)


    grid = torch.cat((xx,yy),3).float()

    if x.is_cuda:
        grid = grid.cuda()

    vgrid = Variable(grid) + flo

    ## scale grid to [-1,1]
    vgrid[:,:,:,0] = 2.0*vgrid[:,:,:,0].clone()/max(W-1,1)-1.0
    vgrid[:,:,:,1] = 2.0*vgrid[:,:,:,1].clone()/max(H-1,1)-1.0

    x = x.permute(0,3,1,2)

    output = torch.nn.functional.grid_sample(x,vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size()))
    mask = torch.nn.functional.grid_sample(mask,vgrid)

    mask[mask<0.9999]=0
    mask[mask>0]=1

    return output*mask


""" Get Data """
class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):        
        indices = []
        for i in range(len(end_idx)-1):
            start = end_idx[i]
            end = end_idx[i+1] - seq_length
            # print(start,end)
            if end > start:
                indices.append(torch.arange(start, end)[0::seq_length+1]) # every 16 Frames is a Sample
                #indices.append(torch.arange(start, end))
        indices = torch.cat(indices)
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
        self.sift_flow = SiftFlowTorch(
                cell_size=1,
                step_size=1,
                is_boundary_included=True,
                num_bins=8,
                cuda=True,
                fp16=True,
                return_numpy=False)

    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        first_image = self.transform(Image.open(self.image_paths[start][0]))
        images = []
        
        for i in indices:
            image_path = self.image_paths[i][0]
            image = cv2.resize(cv2.imread(image_path),(227,227))
            image = simplest_cb(image,2) # colorbalancing

            images.append(image/255.0)  
        targets = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        
        # Sift Algorithm
        descs = self.sift_flow.extract_descriptor(images) # descriptors
        flows = []
        for i in range(0,len(descs)-1):
            flow = find_local_matches(descs[i+1:i+2], descs[i:i+1], 7) # matching
            flow = flow.permute(1, 2, 0).detach().cpu().numpy()
            flows.append(flow)
        flows = torch.tensor(flows, dtype = torch.float) # tensor of flows
        images = torch.tensor(images[:-1], dtype = torch.float) # tensor of images

        warp_imgs = warp(images,flows) # warp images
        warp_imgs = torch.squeeze(warp_imgs)
        warp_imgs = warp_imgs.permute(0,2,3,1)
        warp_imgs = np.float32(warp_imgs)
        
        x = [first_image]
        for i in range(len(warp_imgs)):
            warp_imge = np.array(warp_imgs[i])
            img = np.array(images[i])
            diff = cv2.absdiff(warp_imge,img) # absolute difference
            diff = Image.fromarray(cv2.cvtColor(diff*255,cv2.COLOR_BGR2RGB).astype(np.uint8)) # cv2 to PIL
            diff = self.transform(diff)
            x.append(diff)
        
        x = torch.stack(x)
        y = targets
        return x, y

    def __len__(self):
        return self.length



def get_loader(data_paths):
    """ Gehe durch das Dateset """

    class_image_paths = []
    end_idx = []

    for (d,c) in data_paths:
        paths = sorted(glob.glob(os.path.join(d.path + "/light_mask/", '*.png')))
        # Add class idx to paths
        paths = [(p, c) for p in paths]
        paths = paths[0::2] # every 2nd element
        class_image_paths.extend(paths)
        end_idx.extend([len(paths)])
      
    end_idx = [0, *end_idx]
    end_idx = torch.cumsum(torch.tensor(end_idx), 0)
   

    seq_length = 8

    sampler = MySampler(end_idx, seq_length)
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
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )

    return loader

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
        

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.data.item()))
        del output, loss


def test(model,test_loader):
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

    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': test_loss,
    #     }, "model_checkpoint.pt")
    


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
    root_dir = 'rear_signal_dataset/'
    class_paths = dict((c, []) for c in range(8)) # Dictionary with classes as keys and paths as value
    footage_paths = [l for d in os.scandir(root_dir) if d.is_dir()
        for l in os.scandir(d.path) # Footage_name_XXX
        ]

    for l in footage_paths:
        c = classes[l.path[-3:]] # classes[XXX]
        for d in os.scandir(l): # Footage_name_XXX_DDD
            if d.is_dir:
                class_paths[c] += [(d,c)]

    train_paths = []
    test_paths = []
    for c in class_paths:
        test_paths += random.sample(class_paths[c],int(len(class_paths[c])/5))
        train_paths += [path for path in class_paths[c] if path not in test_paths]
    print(len(train_paths),len(test_paths))
    random.shuffle(train_paths)
    random.shuffle(test_paths)
    
    test_loader = get_loader(test_paths)
    train_loader = get_loader(train_paths)
    print(train_paths[:1])
    print(len(train_loader),len(test_loader))

    # """Show Images"""
    # def imshow(img):
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    # i=0
    # for data, target in train_loader:
    #     if i>=5: break
    #     # show images
    #     imshow(torchvision.utils.make_grid(data[0]))
    #     print(target)
    #     i+=1
    
    """ Model """
    resnet = torchvision.models.resnet50(pretrained=True)
    model = Combine().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss() # loss function

    for epoch in range(1, 11):
        train(epoch,model,optimizer,train_loader)
        test(model,test_loader)

    """ save model """
    import time, datetime
    ts = time.time() 
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    print(timestamp)

    torch.save(model.state_dict(),"saved_models/model_balanced_" + timestamp)