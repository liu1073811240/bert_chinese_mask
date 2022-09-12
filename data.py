from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

# 定义数据集
class ChnDataset(Dataset):
    def __init__(self, split):
        dataset = load_from_disk("./data/ChnSentiCorp")
        if split == 'train':
            dataset = dataset["train"]
        elif split ==  'validation':
            dataset = dataset["validation"]
        else:
            dataset = dataset["test"]

        # 定义数据过滤条件，筛选出字符长度大于30的数据
        def f(data):
            return len(data['text']) > 30

        self.dataset = dataset.filter(f)

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, item):
        text = self.dataset[item]['text']

        return text

if __name__ == '__main__':
    dataset = ChnDataset('test')
    for text in dataset:
        print(text)

    print(len(dataset))