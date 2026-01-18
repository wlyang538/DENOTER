# model/bert_vanilla.py
from torch import nn
from .attention import *

# ✅ 从原始 bert.py 复用 BERTModel（保持 backbone 完全一致）
from .bert import BERTModel


class VanillaBERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = VanillaBERTEmbedding(args)
        self.model = BERTModel(args)

    def forward(self, x):
        x, mask = self.embedding(x)
        scores = self.model(x, self.embedding.token.weight, mask)
        return scores


class VanillaBERTEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 2
        hidden = args.bert_hidden_units
        max_len = args.bert_max_len
        dropout = args.bert_dropout

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=hidden)
        self.position = PositionalEmbedding(max_len=max_len, d_model=hidden)
        self.dropout = nn.Dropout(p=dropout)

    def get_mask(self, x):
        return (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

    def forward(self, x):
        mask = self.get_mask(x)
        x = self.token(x) + self.position(x)
        return self.dropout(x), mask
