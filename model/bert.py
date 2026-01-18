from torch import nn as nn
from .attention import *
import math
import torch.nn.functional as F
from torch.distributions.normal import Normal





class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = BERTEmbedding(self.args)
        self.model = BERTModel(self.args)
        
        self.truncated_normal_init()

        # 添加IB层
        self.h_ib = IBLayer(args, args.bert_hidden_units, args.h_dim1)
        self.h_ib2 = IBLayer2(args, args.h_dim1, args.h_dim2)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(args.bert_hidden_units*2, args.class_num)
        self.r_mu = torch.randn(args.bert_hidden_units)
        self.r_std = torch.randn(args.bert_hidden_units)

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.model.named_parameters():
                if not 'layer_norm' in n:
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)
        
    def forward(self, x):
        x, kl, mask = self.embedding(x) #[batch_size, 200, 500]
        scores = self.model(x, self.embedding.token2.weight, mask)
        return  scores


class BERTEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 2
        hidden = args.bert_hidden_units
        max_len = args.bert_max_len
        dropout = args.bert_dropout
        self.args = args
        self.r_mu = torch.randn(args.class_num)
        self.r_std = torch.randn(args.class_num)

        self.linear2 = nn.Linear(args.bert_hidden_units*2, args.class_num)
        self.h_ib = IBLayer(args, args.bert_hidden_units, args.h_dim1)
        self.h_ib2 = IBLayer2(args, args.h_dim1, args.h_dim2)
        self.h_ib3 = IBLayer3(args, args.h_dim2, args.h_dim3)
        self.token = TokenEmbedding(
            vocab_size=vocab_size, embed_size=hidden)

        self.token2 = TokenEmbedding(
            vocab_size=vocab_size, embed_size=50)
        self.position = PositionalEmbedding(
            max_len=max_len, d_model=hidden)

        self.layer_norm = LayerNorm(50)
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化正确的参数维度
        self.w1 = nn.Parameter(torch.randn(199))
        self.w2 = nn.Parameter(torch.randn(149)) # 对应 num_ones(150) - 1
        self.w3 = nn.Parameter(torch.randn(99))  # 对应 num_ones2(100) - 1

    def get_mask(self, x):
        if len(x.shape) > 2:
            return (x[:, :, 1:].sum(-1) > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        return (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    
    def kl_divergence(self, mu1, std1, mu2, std2):
        std1_2 = torch.pow(std1,2)
        std2_2 = torch.pow(std2,2)
        mu_2 = torch.pow((mu1-mu2),2)
        std1_2 = std1_2.to(device=mu1.device) 
        std2_2 = std2_2.to(device=mu1.device)
        mu_2 = mu_2.to(device=mu1.device)
        kl = torch.log(std2/std1) - 0.5 + (mu_2 + std1_2) / std2_2 / 2
        return kl
    
    def threshold_binary(self, logits, num_ones):
        sorted_indices = torch.argsort(logits, descending=True)
        binary_tensor = torch.zeros_like(logits)
        binary_tensor[sorted_indices[:num_ones]] = 1
        return binary_tensor

    def forward(self, x): 
        mask = self.get_mask(x)
        if len(x.shape) > 2:
            pos = self.position(x[:, :, 1:].sum(-1))
            x = torch.matmul(x, self.token.weight) + pos
        else:
            x = self.token(x) + self.position(x) 

        # =========================================================
        # ✅ Vanilla 模式：关闭 DENOTER（但参数仍然存在）
        # =========================================================
        if not self.args.use_denoter:
            print('11111111111')
            # 不做 item filtering
            # 不做 feature filtering (IB layers)
            # 直接返回 backbone 需要的 embedding
            return self.dropout(x), torch.zeros(1, device=x.device), mask

        print('22222222222')
        # =========================================================
        # 1. 项目过滤 (Item Filtering / OT 逻辑) - 调整为首先执行
        # =========================================================
        data = x[:, :199, :] # 使用原始隐藏层维度 (如 64) 的嵌入数据
        indices_list = torch.tensor(list(range(1, 200)), device=x.device)
        
        ''' 第一次OT '''
        num_ones = 150
        w1 = self.w1
        pr = F.sigmoid(w1.view(-1,1)).view(-1)
        sorted_indices = torch.argsort(pr, descending=True)
        binary_tensor = torch.zeros_like(pr)
        binary_tensor[sorted_indices[:num_ones]] = 1
        indices = torch.nonzero(binary_tensor == 1).flatten() 
        indices_list = indices_list[indices] 
        
        ''' 第二次OT '''
        num_ones2 = 100
        w2 = self.w2
        pr2 = F.sigmoid(w2.view(-1, 1)).view(-1)
        sorted_indices2 = torch.argsort(pr2, descending=True)
        binary_tensor2 = torch.zeros_like(pr2)
        binary_tensor2[sorted_indices2[:num_ones2]] = 1
        indices2 = torch.nonzero(binary_tensor2 == 1).flatten() 
        indices_list =  indices_list[indices2] 
        
        ''' 第三次OT '''
        num_ones3 = 50
        w3 = self.w3
        pr3 = F.sigmoid(w3.view(-1, 1)).view(-1)
        sorted_indices3 = torch.argsort(pr3, descending=True)
        binary_tensor3 = torch.zeros_like(pr3)
        binary_tensor3[sorted_indices3[:num_ones3]] = 1
        indices3 = torch.nonzero(binary_tensor3 == 1).flatten() 
        indices_list =  indices_list[indices3] 
        
        selected_data = data[:,indices_list,:]      
        average_features = selected_data.mean(dim=1).unsqueeze(1)
        
        # 得到经过项目过滤后的序列
        item_filtered_x = torch.cat((data, average_features), dim=1)

        # =========================================================
        # 2. 特征过滤 (Feature Filtering / IB 层) - 调整为对过滤后的项目执行
        # =========================================================
        mu, std, sample = self.h_ib(item_filtered_x)
        mu, std, sample = self.h_ib2(sample)
        mu, std, sample = self.h_ib3(sample)
        
        self.r_std = self.r_std.to(device=x.device)       
        self.r_mu = self.r_mu.to(device=x.device)
        kl = self.kl_divergence(mu, std, self.r_std, self.r_mu)
        
        sample = self.dropout(sample)
        return self.dropout(self.layer_norm(sample)), kl, mask


class BERTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden = args.bert_hidden_units
        heads = args.bert_num_heads
        head_size = args.bert_head_size
        dropout = args.bert_dropout
        attn_dropout = args.bert_attn_dropout
        layers = args.bert_num_blocks
        rec_units = args.rec_units

        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            hidden, heads, head_size, hidden * 4, dropout, attn_dropout) for _ in range(layers)])
        self.linear = nn.Linear(rec_units, rec_units)
        self.bias = torch.nn.Parameter(torch.zeros(args.num_items + 2))
        self.bias.requires_grad = True
        self.activation = GELU()

    def forward(self, x, embedding_weight, mask):
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        x = self.activation(self.linear(x))
        scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias 
        return scores


class IBLayer(nn.Module):
    def __init__(self, args, i_dim, h_dim1):
        super(IBLayer, self).__init__()
        self.args = args
        self.h_dim1 = args.h_dim1
        self.encoder = nn.Linear(i_dim, 2*h_dim1) 

    def forward(self, embeds):
        enc = self.encoder(embeds)  
        mu = enc[: , :, :self.h_dim1 ]
        std = F.softplus(enc[: , :, self.h_dim1:])
        sample = self.reparameter(mu, std)
        return mu, std, sample

    def reparameter(self, mu, std):
        eps = Normal(mu,std)
        return eps.rsample()
    

class IBLayer2(nn.Module):
    def __init__(self, args, h_dim1, h_dim2):
        super(IBLayer2, self).__init__()
        self.encoder = nn.Linear(h_dim1, 2*h_dim2)
        self.h_dim2 = h_dim2

    def forward(self, embeds):
        enc = self.encoder(embeds)  
        mu = enc[: , :, :self.h_dim2 ]
        std = F.softplus(enc[: , :, self.h_dim2:])
        sample = self.reparameter(mu, std)
        return mu, std, sample

    def reparameter(self, mu, std):
        eps = Normal(mu,std)
        return eps.rsample()
    
class IBLayer3(nn.Module):
    def __init__(self, args, h_dim2, h_dim3):
        super(IBLayer3, self).__init__()
        self.encoder = nn.Linear(h_dim2, 2*h_dim3)
        self.h_dim3 = h_dim3

    def forward(self, embeds):
        enc = self.encoder(embeds)  
        mu = enc[: , :, :self.h_dim3 ]
        std = F.softplus(enc[: , :, self.h_dim3:])
        sample = self.reparameter(mu, std)
        return mu, std, sample

    def reparameter(self, mu, std):
        eps = Normal(mu,std)
        return eps.rsample()
