import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    # 해당 클래스는 torch.utils.data.Dataset을 상속받습니다.
    def __init__(self, data, target):
        # 해당 클래스의 인스턴스를 초기화합니다.
        self.data = data
        self.target = target

    def __len__(self):
        # 해당 데이터셋의 총 데이터 수를 리턴합니다.
        return len(self.data)
    
    def __getitem__(self, idx):
        # 인덱스 idx에 해당하는 데이터를 가져옵니다.
        return self.data[idx], self.target[idx]
    

if __name__ == "__main__":
    data = torch.randn(100, 3, 32, 32)
    target = torch.randint(0, 10, (100, ))
    mydataset = MyDataset(data, target)

    myData, myTarget = mydataset[30]

    print(len(mydataset))
    print(myData.shape)
    print(myTarget)
    