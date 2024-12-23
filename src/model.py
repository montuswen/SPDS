import torch
import torch.nn as nn
from transformers import AutoModel

class PlagiarismModel(nn.Module):
    def __init__(self, pretrained_model_name, embedding_dim):
        super(PlagiarismModel, self).__init__()
        # 加载预训练模型
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        # 添加一个全连接层用于进一步处理句子嵌入
        self.fc = nn.Linear(self.encoder.config.hidden_size, embedding_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, anchor_input, sentence_input):
        # 从预训练模型提取嵌入
        anchor_output = self.encoder(**anchor_input)["pooler_output"]  # [batch_size, hidden_size]
        sentence_output = self.encoder(**sentence_input)["pooler_output"]  # [batch_size, hidden_size]

        # 将嵌入投影到低维空间（可选）
        anchor_embed = self.fc(anchor_output)  # [batch_size, embedding_dim]
        sentence_embed = self.fc(sentence_output)  # [batch_size, embedding_dim]

        # 计算余弦相似度
        cosine_similarity = torch.cosine_similarity(anchor_embed, sentence_embed, dim=-1)
        return cosine_similarity
