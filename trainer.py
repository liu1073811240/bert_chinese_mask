import torch
from net import Model
from data import ChnDataset
from transformers import BertTokenizer
from transformers import AdamW


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 1000

# 加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')

def collate_fn(data):
    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=data,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=30,  # 根据情况取最大长度
                                   return_tensors='pt',
                                   return_length=True)

    # input_ids: 编码之后的数字
    # attention_mask: 是补零的位置是0， 其它位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    # 把第15个词固定替换为mask,
    # print(input_ids)
    labels = input_ids[:, 15].reshape(-1).clone()  # 取其中第15个标签进来训练。
    # print(labels)
    input_ids[:, 15] = token.get_vocab()[token.mask_token]  # 返回加了掩码的ids
    # print(input_ids)

    return input_ids, attention_mask, token_type_ids, labels

# 数据集
train_dataset = ChnDataset('train')
val_dataset = ChnDataset('validation')

# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=32,
                                           collate_fn=collate_fn,
                                           shuffle=True,
                                           drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=32,
                                         collate_fn=collate_fn,
                                         shuffle=True,
                                         drop_last=True)

model = Model().to(DEVICE)

# 训练
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(EPOCH):
    sum_val_acc = 0
    sum_val_loss = 0
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), \
                                                            token_type_ids.to(DEVICE), \
                                                            labels.to(DEVICE)
        # print(input_ids.shape, attention_mask.shape, token_type_ids.shape)  # torch.Size([32, 30])
        # print(input_ids)
        # print(attention_mask)
        # print(token_type_ids)
        # print(labels)
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print(out)
        # print(out.shape)  # torch.Size([32, 21128])
        # print(labels)  # tensor([ 678, 4638, 2411, 2523,  511, 2, ...]

        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 50 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            print(epoch, i, loss.item(), accuracy)

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(val_loader):
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), \
                                                            token_type_ids.to(DEVICE), labels.to(DEVICE)
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)

        loss = criterion(out, labels).item()
        out = out.argmax(dim=1)

        accuracy = (out == labels).sum().item() / len(labels)
        sum_val_loss = sum_val_loss + loss  # 累加批次损失 成 这一轮 的总损失
        sum_val_acc = sum_val_acc + accuracy  # 累加批次正确个数 成 这一轮 的总正确个数

    avg_val_loss = sum_val_loss / len(val_loader)
    avg_val_acc = sum_val_acc / len(val_loader)
    print(f"val==>epoch:{epoch}, avg_val_loss:{avg_val_loss}, avg_val_acc:{avg_val_acc}")

    torch.save(model.state_dict(), f"param/{epoch}-bert_chinese_mask.pth")
    print(epoch, "参数保存成功")





