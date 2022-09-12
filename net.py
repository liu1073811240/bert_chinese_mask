import torch
from transformers import BertModel


# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese').to(DEVICE)
# print(pretrained)

from transformers import BertTokenizer

# 加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')

# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.decoder = torch.nn.Linear(768, token.vocab_size, bias=False)  # 输出字典词汇的个数
        self.bias = torch.nn.Parameter(torch.zeros(token.vocab_size))

        self.decoder.bias = self.bias  # 为了不影响后续的处理，将偏置置为0.

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        # print(out)
        # print(out.last_hidden_state.shape)  # torch.Size([5, 30, 768])  NSV
        # print(out.last_hidden_state[:, 15])
        # print(out.last_hidden_state[:, 15].shape)  # torch.Size([5, 768])
        out = self.decoder(out.last_hidden_state[:, 15])  # 选择索引节点15 作为输出
        return out

if __name__ == '__main__':
    model = Model().to(DEVICE)
    # print(model)

    input_ids = torch.ones(5, 30).to(DEVICE).long()
    attention_mask = torch.ones(5, 30).to(DEVICE).long()
    token_type_ids = torch.zeros(5, 30).to(DEVICE).long()

    out = model(input_ids, attention_mask, token_type_ids)
    print(out.shape)  # torch.Size([5, 21128])
