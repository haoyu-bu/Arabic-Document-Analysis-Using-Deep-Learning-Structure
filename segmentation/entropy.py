import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from data_loader import DocDataset
import os
import numpy as np
from PIL import Image
import cv2

num_epochs = 51
batch_size = 128
learning_rate = 1e-3
img_width = 256
img_height = 256
train = False

if not os.path.exists('./result'):
    os.mkdir('./result')

def to_img(x):
    x = x.max(1)[1]
    x = x.view(x.size(0), 1, img_width, img_height)
    return x

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 2, 2, stride=2, padding=1),  
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

if __name__ == "__main__":
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = DocDataset("../CRAFT-pytorch/bbox/", "../bce_augmented/", img_width, img_height, split='train', transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    testset = DocDataset("../CRAFT-pytorch/bbox/", "../bce_augmented/", img_width, img_height, split='test', transform=img_transform)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    model = autoencoder().cuda()
    
    if (train):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)
        for epoch in range(num_epochs):
            for data in dataloader:
                img, target = data
                img = Variable(img).cuda()
                target = Variable(target.to(torch.int64)).cuda()
                target = target.squeeze(1)

                output = model(img)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('epoch [{}/{}], loss:{:.4f}'
                .format(epoch+1, num_epochs, loss.data[0]))
            if epoch % 10 == 0:
                pic = to_img(output.cpu().data)
                save_image(pic, './result/image_{}.png'.format(epoch))
                torch.save(model.state_dict(), './checkpoint.pth')

        torch.save(model.state_dict(), './checkpoint.pth')
    else:
        model.eval()
        model.load_state_dict(torch.load("./checkpoint.pth"))
        preds = []
        gts = []
        paths = []
        with torch.no_grad():
            for data in testloader:
                img, target, path = data
                img = Variable(img).cuda()
                target = Variable(target.to(torch.int64)).cuda()
                target = target.squeeze(1)

                output = model(img)
                pred = output.cpu().data.max(1)[1].squeeze(0).numpy()
                preds.append(pred)
                gts.append(target.cpu().data.squeeze(0).numpy())
                paths += path
        acc, acc_cls, mean_iu, fwavacc = evaluate(preds, gts, 2)
        print(acc, acc_cls, mean_iu, fwavacc)
        for p, pr in zip(paths, preds):
            size = Image.open(p).size
            img = Image.fromarray(pr.astype('int8') * 255)
            img = img.resize(size)
            cv2.imwrite(p + "_res.png", np.array(img))


