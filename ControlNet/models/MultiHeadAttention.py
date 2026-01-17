import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        input_proj_bias: bool = True,
        output_proj_bias: bool = True,
    ):
        """
        Initialize the MultiHeadMultiHeadSelfAttention module.

        Args:
            num_heads (int): The number of attention heads.
            embed_dim (int): The dimension of the input and output embeddings.
            input_proj_bias (bool, optional): Whether to include bias in the input projection layer. Defaults to True.
            output_proj_bias (bool, optional): Whether to include bias in the output projection layer. Defaults to True.

        Attributes:
            input_proj (nn.Linear): The linear layer that combines the Wq, Wk, and Wv matrices into one matrix.
            output_proj (nn.Linear): The linear layer that represents the Wo matrix.
            num_heads (int): The number of attention heads.
            head_dim (int): The dimension of each attention head.
        """
        super().__init__()

        # This combines the Wq, Wk and Wv matrices into one matrix
        self.input_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=input_proj_bias)
        # This one represents the Wo matrix
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=output_proj_bias)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

    def forward(self, inputs, apply_causal_mask=False):
        """
        Perform the forward pass of the self-attention mechanism.

        Args:
            inputs (torch.Tensor): Input tensor of shape (Batch_Size, Seq_Len, Dim).
            apply_causal_mask (bool, optional): If True, applies a causal mask to prevent attending to future positions. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, Seq_Len, Dim) after applying self-attention.
        """
        # inputs: (Batch_Size, Seq_Len, Dim)

        # (Batch_Size, Seq_Len, Dim)
        input_shape = inputs.shape

        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, embed_dim = input_shape

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.input_proj(inputs).chunk(3, dim=-1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=apply_causal_mask
        )

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2).contiguous()

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.output_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        cross_dim,
        input_proj_bias=True,
        output_proj_bias=True,
    ):
        """
        Initialize the MultiHeadMultiHeadCrossAttention module.

        Args:
            num_heads (int): The number of attention heads.
            embed_dim (int): The dimension of the input and output embeddings for the query.
            cross_dim (int): The dimension of the input embeddings for the key and value.
            input_proj_bias (bool, optional): Whether to include bias in the input projection layers. Defaults to True.
            output_proj_bias (bool, optional): Whether to include bias in the output projection layer. Defaults to True.

        Attributes:
        We can define separately q, k, v instead of one matrix like in self-attention.

            query_proj (nn.Linear): The linear layer for projecting the query.
            key_proj (nn.Linear): The linear layer for projecting the key.
            value_proj (nn.Linear): The linear layer for projecting the value.
            output_proj (nn.Linear): The linear layer for projecting the output.
            num_heads (int): The number of attention heads.
            head_dim (int): The dimension of each attention head.
        """
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=input_proj_bias)
        self.key_proj = nn.Linear(cross_dim, embed_dim, bias=input_proj_bias)
        self.value_proj = nn.Linear(cross_dim, embed_dim, bias=input_proj_bias)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=output_proj_bias)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

    def forward(self, query, context):
        # query (latent): # (Batch_Size, Seq_Len_Q, Dim_Q) - queries
        # context (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768) - keys and values - context == prompt

        input_shape = query.shape
        batch_size, seq_length_q, embed_dim = input_shape
        # Divide each embedding of Q into multiple heads such that head_dim * num_heads = Dim_Q
        interim_shape = (batch_size, -1, self.num_heads, self.head_dim)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.query_proj(query)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.key_proj(context)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.value_proj(context)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()

        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.output_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output
