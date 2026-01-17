import math

import pytest
import torch

from ControlNet.models.MultiHeadAttention import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)


def _manual_self_attention(module, inputs, apply_causal_mask):
    batch_size, seq_len, embed_dim = inputs.shape
    head_dim = module.head_dim
    num_heads = module.num_heads

    q, k, v = module.input_proj(inputs).chunk(3, dim=-1)
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    weight = q @ k.transpose(-1, -2)
    if apply_causal_mask:
        mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
        weight = weight.masked_fill(mask, -torch.inf)
    weight = weight / math.sqrt(head_dim)
    weight = torch.softmax(weight, dim=-1)

    output = weight @ v
    output = output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    output = module.output_proj(output)
    return output


def _manual_cross_attention(module, query, context):
    batch_size, seq_len_q, embed_dim = query.shape
    head_dim = module.head_dim
    num_heads = module.num_heads

    q = module.query_proj(query)
    k = module.key_proj(context)
    v = module.value_proj(context)

    q = q.view(batch_size, seq_len_q, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

    weight = q @ k.transpose(-1, -2)
    weight = weight / math.sqrt(head_dim)
    weight = torch.softmax(weight, dim=-1)

    output = weight @ v
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, embed_dim)
    output = module.output_proj(output)
    return output


def test_self_attention_matches_manual():
    torch.manual_seed(0)
    module = MultiHeadSelfAttention(num_heads=2, embed_dim=8)
    inputs = torch.randn(2, 4, 8)

    expected = _manual_self_attention(module, inputs, apply_causal_mask=False)
    actual = module(inputs, apply_causal_mask=False)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_self_attention_causal_mask_matches_manual():
    torch.manual_seed(1)
    module = MultiHeadSelfAttention(num_heads=4, embed_dim=8)
    inputs = torch.randn(1, 5, 8)

    expected = _manual_self_attention(module, inputs, apply_causal_mask=True)
    actual = module(inputs, apply_causal_mask=True)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_cross_attention_matches_manual():
    torch.manual_seed(2)
    module = MultiHeadCrossAttention(num_heads=2, embed_dim=8, cross_dim=6)
    query = torch.randn(2, 3, 8)
    context = torch.randn(2, 5, 6)

    expected = _manual_cross_attention(module, query, context)
    actual = module(query, context)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_cross_attention_invalid_heads_raises():
    with pytest.raises(AssertionError):
        MultiHeadCrossAttention(num_heads=3, embed_dim=8, cross_dim=8)
