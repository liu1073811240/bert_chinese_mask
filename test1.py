import torch
from transformers import BertTokenizer
from net import Model
from data import ChnDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')
def collate_fn(data):
    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=data,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=30,
                                   return_tensors='pt',
                                   return_length=True)

    # input_ids: 编码之前的数字
    # attention_mask: 是补零的位置是0， 其它位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    # 把第15个词固定替换为mask
    labels = input_ids[:, 15].reshape(-1).clone()
    input_ids[:, 15] = token.get_vocab()[token.mask_token]

    print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels

# 测试
def test():
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load('param/20-bert_chinese_mask.pth'))
    model.eval()
    correct = 0
    total = 0
    loader_test = torch.utils.data.DataLoader(dataset=ChnDataset('test'),
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), \
                                                             attention_mask.to(DEVICE), \
                                                            token_type_ids.to(DEVICE), \
                                                            labels.to(DEVICE)
        print(i)
        if i == 15:
            break

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        correct += (out==labels).sum().item()
        total += len(labels)

        print(token.decode(input_ids[0]))
        print("标签：", token.decode(labels[0]))
        print("输出：", token.decode(out[0]))

    print(correct / total)

if __name__ == '__main__':
    test()

