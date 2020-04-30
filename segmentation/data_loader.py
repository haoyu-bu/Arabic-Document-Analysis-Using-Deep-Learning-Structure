import torch.utils.data as data
import glob
import os
from PIL import Image
import numpy as np

class DocDataset(data.Dataset):
    def __init__(self, img_dir, target_dir, img_width=256, img_height=256, split='train', transform=None):
        self.split = split
        if split == 'train':
            self.imgs, self.targets = load_data(img_dir, target_dir)
        elif split == 'test':
            self.imgs, self.targets = load_testdata(img_dir, target_dir)
        self.transform = transform
        self.img_width = img_width
        self.img_height = img_height

    def __getitem__(self, index):

        img_path = self.imgs[index]
        target_path = self.targets[index]

        img = Image.open(img_path).resize((self.img_width,self.img_height))
        target = Image.open(target_path)
        red, _, _ = target.split()
        red = np.array(red)
        red[red > 0] = 255
        target = Image.fromarray(red).resize((self.img_width,self.img_height))

        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)
        if self.split == 'test':
            return img, target, img_path
        return img, target

    def __len__(self):
        return len(self.imgs)

def load_data(img_dir, target_dir):
    img_lists = []
    target_lists = []
    for i in range(1, 6):
        img_lists += glob.glob(os.path.join(img_dir, str(i), "*"))
        target_lists += glob.glob(os.path.join(target_dir, str(i), "PedMasks", "*"))
    img_lists.sort()
    target_lists.sort()
    # print(len(img_lists))
    # print(len(target_lists))
    # index = 1300
    # print(target_lists[index])
    # print(img_lists[index])
    # img = Image.open(img_lists[index])
    # img.show()
    # img = Image.open(target_lists[index])
    # print(img.size)
    # red, green, blue = img.split()
    # red = np.array(red)
    # red[red > 0] = 255
    # red = Image.fromarray(red)
    # red.show()
    # w = []
    # h = []
    # r = []
    # for i in img_lists:
    #     img = Image.open(i)
    #     w.append(img.size[0])
    #     h.append(img.size[1])
    #     r.append(1.0 * w[-1] / h[-1])
    # print(min(w), max(w), np.mean(w))
    # print(min(h), max(h), np.mean(h))
    # print(min(r), max(r), np.mean(r))
    # test_img = []
    # test_target = []
    # test_img += glob.glob(os.path.join(img_dir, "0", "*"))
    # test_target += glob.glob(os.path.join(target_dir, "0", "PedMasks", "*"))
    return img_lists, target_lists

def load_testdata(img_dir, target_dir):
    img_lists = []
    target_lists = []
    img_lists += glob.glob(os.path.join(img_dir, "0", "*"))
    target_lists += glob.glob(os.path.join(target_dir, "0", "PedMasks", "*"))
    img_lists.sort()
    target_lists.sort()
    return img_lists, target_lists

if __name__ == "__main__":
    load_data("../CRAFT-pytorch/bbox", "../bce_augmented/")