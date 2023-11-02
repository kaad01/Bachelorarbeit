import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms

from PIL import Image
import cv2
import matplotlib.pyplot as plt

from helpers.sift_flow_torch import SiftFlowTorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""" define model """
class Combine(nn.Module):
    global resnet
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
        print(x)
        x = torch.max(x,1)[1]
        return x.item()

def load_model(PATH):
    """ Model """
    global resnet
    resnet = torchvision.models.resnet50(pretrained=True)
    model = Combine().to(device)

    """load checkpoint"""
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint)

    return model


class Transformer:

    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.sift_flow = SiftFlowTorch(
                cell_size=1,
                step_size=1,
                is_boundary_included=True,
                num_bins=8,
                cuda=True,
                fp16=True,
                return_numpy=False)

    """sift algorithm"""
    def find_local_matches(self,desc1, desc2, kernel_size=9):
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

    def warp(self,x,flo):
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

    def transform(self, frames):
        #first_image = self.transforms(Image.fromarray(frames[0])) # numpyarray to PIL
        first_image = self.transforms(Image.fromarray(cv2.cvtColor(frames[0],cv2.COLOR_BGR2RGB).astype(np.uint8)))
        resized = []
        for img in frames:
            img = cv2.resize(img,(227,227))
            resized.append(img/255.0)

        # Sift Algorithm
        descs = self.sift_flow.extract_descriptor(resized) # descriptors
        flows = []
        for i in range(0,len(descs)-1):
            flow = self.find_local_matches(descs[i+1:i+2], descs[i:i+1], 7) # matching
            flow = flow.permute(1, 2, 0).detach().cpu().numpy()
            flows.append(flow)
        flows = torch.tensor(flows, dtype = torch.float) # tensor of flows
        images = torch.tensor(resized[:-1], dtype = torch.float) # tensor of images

        warp_imgs = self.warp(images,flows) # warp images
        warp_imgs = torch.squeeze(warp_imgs)
        warp_imgs = warp_imgs.permute(0,2,3,1)
        warp_imgs = np.float32(warp_imgs)
        
        x = [first_image]
        for i in range(len(warp_imgs)):
            warp_imge = np.array(warp_imgs[i])
            img = np.array(images[i])
            diff = cv2.absdiff(warp_imge,img) # absolute difference
            diff = Image.fromarray(cv2.cvtColor(diff*255,cv2.COLOR_BGR2RGB).astype(np.uint8)) # cv2 to PIL
            diff = self.transforms(diff)
            x.append(diff)
        
        x = torch.stack(x)
        x = x.unsqueeze(0).cuda()

        # """ show images """
        # for img in x:
        #     for imgs in img:
        #         imshow(imgs)


        return x

"""Show Images"""
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    tensor=np.transpose(tensor, (1, 2, 0))
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)