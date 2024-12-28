import torch
from tiktoken_tokenize import create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

"""
Now instead of toy example in `token_embeddings.py` module,
lets consider our small dataset `the-verdict.txt`,
vocabulary size of 50257 (as that of GPT2) and embedding
dimensions of 256.
"""
vocab_size = 50257
embedding_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

seq_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=4, max_length=seq_length, stride=seq_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, target = next(data_iter)
print(inputs)
print(inputs.shape)
"""
tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257]])
torch.Size([4, 4])
"""

inputs_token_embeddings = token_embedding_layer(inputs)
print(inputs_token_embeddings.shape)
"""
torch.Size([4, 4, 256])

Note that for each token, embedding vector of size 256 is created. Thus, result vector has shape 4x4x256.
"""

position_embedding_layer = torch.nn.Embedding(seq_length, embedding_dim)
"""
we want number of position embeddings equal to sequence length (i.e. number of tokens selected in input data).
One position embedding vector is added to token embedding vector of each token.
"""
input_pos_embeddings = position_embedding_layer(torch.arange(seq_length))
print(input_pos_embeddings.shape)
"""
torch.Size([4, 256])

As we can see, the positional embedding tensor consists of four 256-dimensional vectors.
We can now add these directly to the token embeddings, where PyTorch will add the 4x256 dimensional
`input_pos_embeddings` tensor to each 4x256 dimensional token embedding tensor in each of the four batches
"""

input_embeddings = inputs_token_embeddings + input_pos_embeddings
print(input_embeddings.shape)
"""
torch.Size([4, 4, 256])
"""
