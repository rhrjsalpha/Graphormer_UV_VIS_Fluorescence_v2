import torch
import torch.nn as nn
from GP.modules_All.multihead_attention import MultiheadAttention
from GP.modules_All.quant_noise import quant_noise


class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        activation_fn="relu",
        pre_layernorm=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super(GraphormerGraphEncoderLayer, self).__init__()

        self.pre_layernorm = pre_layernorm

        # Layer Norms
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        # Self-Attention
        self.self_attn = MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # Feedforward network
        self.fc1 = quant_noise(
            nn.Linear(embedding_dim, ffn_embedding_dim), q_noise, qn_block_size
        )
        self.fc2 = quant_noise(
            nn.Linear(ffn_embedding_dim, embedding_dim), q_noise, qn_block_size
        )

        # Activation Function
        self.activation_fn = (
            nn.ReLU() if activation_fn == "relu" else nn.GELU()
        )

        # Dropouts
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)

    def forward(
        self,
        x,
        self_attn_bias=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """
        Forward pass for the encoder layer.

        Args:
            x: Input tensor of shape (T, B, C)
            self_attn_bias: Attention bias
            self_attn_mask: Attention mask
            self_attn_padding_mask: Padding mask for attention

        Returns:
            Output tensor and attention weights
        """
        residual = x

        # Apply LayerNorm if pre-layernorm is enabled
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        # Self-attention
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
        )

        # Dropout and residual connection
        x = self.dropout(x)
        x = residual + x

        # Apply LayerNorm if post-layernorm is used
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        # Feedforward
        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x

        if not self.pre_layernorm:
            x = self.final_layer_norm(x)

        return x, attn
