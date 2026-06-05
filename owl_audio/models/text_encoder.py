import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BidirectionalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask):
        B, T, D = x.shape
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, H, T, Hd]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = self.dropout(F.softmax(attn, dim=-1))
        out  = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)
    

class MultiLayerAggregator(nn.Module):
    """Collapses all T5 hidden layers into a single rich embedding per token"""
    def __init__(self, int, num_layers: int, hidden_dim: int = 32):
        super().__init__()
        self.num_layers = num_layers
        self.layer_mixer = nn.Sequential(
            nn.Linear(num_layers, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hidden_states: list[torch.Tensor]):
        """
        Args:
            hidden_states: list of L tensors each shaped [B, T, D]
        Returns:
            [B, T, target_dim]
        """
        assert len(hidden_states) == self.num_layers

        stacked = torch.stack(hidden_states, dim=-1) # [B, T, D, L]
        aggregated = self.layer_mixer(stacked).squeeze(-1)  # [B,T,D]
        return aggregated

class EnricherBlock(nn.Module):
    """Pre-norm transformer block (bidirectional) — no causal masking."""
    def __init__(self, dim: int, num_heads: int = 8, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = BidirectionalAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, ff_mult, dropout)

    def forward(self, x, key_padding_mask):
        x = x + self.attn(x, key_padding_mask)
        x = x + self.ff(x)
        return x


class FeatureEnricher(nn.Module):
    """Lightweight bidirectional transformer that 
    refines the aggregated multi-layer embeddings"""
    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        output_dim = output_dim or dim

        self.blocks = nn.ModuleList([
            EnricherBlock(dim, num_heads, ff_mult, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(
        self, x, attention_mask):
        """Returns [B, T, output_dim]."""
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0) # [B, T]

        for block in self.blocks:
            x = block(x, key_padding_mask)

        return self.norm(x)  # [B, T, dim]


class RichTextEncoder(nn.Module):
    def __init__(
        self,
        base_encoder_name: str = "t5-base",
        d_text: int = 768,
        num_enricher_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_encoder_name)
        self.base_encoder = AutoEncoderModel.from_pretrained(
            base_encoder_name,
            output_hidden_states=True,
        )
        for p in self.base_encoder.parameters():
            p.requires_grad_(False)

        encoder_layers = self.base_encoder.config.num_layers
        self.aggregator = MultiLayerAggregator(encoder_layers)
        self.enricher = FeatureEnricher(
            dim=d_text,
            num_layers=num_enricher_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def tokenize(self, texts, max_length=512, device='cuda'):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        return tokens

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            enc_out = self.base_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden_states = list(enc_out.hidden_states[1:]) # L × [B, T, D]
        rich_emb = self.aggregator(hidden_states) # [B, T, d_model]
        return self.enricher(rich_emb, attention_mask) # [B, T, output_dim]
   

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = RichTextEncoder(
        base_encoder_name="t5-base",
        d_text=768,
        num_enricher_layers=4,
        num_heads=8
    ).to(device)