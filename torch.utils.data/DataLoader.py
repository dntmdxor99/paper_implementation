import torch
from torch.utils.data import DataLoader
from Dataset import MyDataset


if __name__ == "__main__":
    data = torch.randn(100, 3, 32, 32)
    target = torch.randint(0, 10, (100, ))

    mydataset = MyDataset(data, target)
    
    mydataloader = DataLoader(mydataset, batch_size=9, shuffle=True, drop_last=False)

    for epoch in range(2):
        print(epoch)
        for i, (data, target) in enumerate(mydataloader):
            print(i, data.shape, target.shape, target)