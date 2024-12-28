import torch

"""
Lets say we have a text with 4 tokens as input.

input_token = torch.tensor([2, 3, 5, 1])

Lets say vocabulary size is 6 (instead of 50,257 words in BPE tokenizer vocabulary)
and embedding size is 3 (instead of embedding of size 12,288 dimensions in GPT-3).

vocab_size = 6
embedding_dim = 3
"""
vocab_size = 6
embedding_dim = 3
embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
print(embedding_layer.weight)
"""

Parameter containing:
tensor([[ 0.4777,  0.2426,  0.8995],
        [-0.4467,  1.2093,  0.9864],
        [ 0.1823,  0.1200,  0.9880],
        [-0.4133, -0.2543, -2.1815],
        [-0.2282, -1.4648, -0.1361],
        [ 1.1859, -0.3399,  0.9679]], requires_grad=True)

Torch embedding layer is a lookup table that stores embeddings of a fixed
dictionary and size.

Initially, the weight matrix of the embedding layer contains small,
random values. These values are optimized during LLM training as part of
the LLM optimization itself.
In our case, we assumed vocabulary of size 6, which is represented by 6 rows.
We assumed embedding dim to be 3, which is the size of each row.

In short, there is one row for each of the six possible tokens in the
vocabulary, and there is one column for each of the three embedding dimensions.

Since this layer is a lookup table, we can retrieve embeddings for specific token.
"""

print(embedding_layer(torch.tensor([3])))

"""
tensor([[-0.1045, -0.6146,  0.3275]], grad_fn=<EmbeddingBackward0>)
"""
