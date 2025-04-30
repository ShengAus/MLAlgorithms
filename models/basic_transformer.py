import torch
import torch.nn as nn
import math

# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        output = attn @ V
        return output, attn

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0
        self.d_k = embed_size // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.fc = nn.Linear(embed_size, embed_size)
        self.attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        out, _ = self.attention(Q, K, V, mask)

        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        out = self.fc(out)
        return out

# Position-wise FeedForward Network
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, embed_size)
        )

    def forward(self, x):
        return self.net(x)

# Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size):
        super().__init__()
        self.mha = MultiHeadAttention(embed_size, num_heads)
        self.ff = FeedForward(embed_size, ff_hidden_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x, mask=None):
        attn_out = self.mha(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

# Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size):
        super().__init__()
        self.self_mha = MultiHeadAttention(embed_size, num_heads)
        self.enc_dec_mha = MultiHeadAttention(embed_size, num_heads)
        self.ff = FeedForward(embed_size, ff_hidden_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        self_attn_out = self.self_mha(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn_out)
        enc_dec_attn_out = self.enc_dec_mha(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + enc_dec_attn_out)
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        return x

# Mini Transformer Encoder
class MiniTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, ff_hidden_size, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_size))
        self.encoder = TransformerEncoderLayer(embed_size, num_heads, ff_hidden_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        x = self.encoder(x)
        return x

# Mini Transformer Decoder
class MiniTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, ff_hidden_size, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_size))
        self.decoder = TransformerDecoderLayer(embed_size, num_heads, ff_hidden_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        batch_size, seq_len = x.shape
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        x = self.decoder(x, enc_out, src_mask, tgt_mask)
        logits = self.fc_out(x)
        return logits

# Example usage
vocab_size = 20
embed_size = 32
num_heads = 4
ff_hidden_size = 64

encoder = MiniTransformerEncoder(vocab_size, embed_size, num_heads, ff_hidden_size)
decoder = MiniTransformerDecoder(vocab_size, embed_size, num_heads, ff_hidden_size)

src = torch.randint(0, vocab_size, (2, 10))  # (batch_size, src_seq_len)
tgt = torch.randint(0, vocab_size, (2, 10))  # (batch_size, tgt_seq_len)

enc_out = encoder(src)
output = decoder(tgt, enc_out)
print("Output shape:", output.shape)  # (batch_size, tgt_seq_len, vocab_size)
