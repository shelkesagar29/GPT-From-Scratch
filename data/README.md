## Ch-02. Working with text data
- The concept of converting data into a vector format is often referred as `embedding`.
Here data can be text, image or some other modality.
- At its core, `embedding` is a mapping from discrete objects, such as words, images or even entire
documents, to points in a continuous vector space (i.e. convert nonnumerical data to numerical one).
- `Word2Vec` is one of the earlier and most popular framework to generate word embeddings.
- While pretrained NNs (Neural Networks) such as `Word2Vec`can be used to generate embeddings, LLMs
commonly produce their own embeddings that are part of the input layer and are update during training.
- The advantage of optimizing the embeddings as part of the LLM training instead of using `Word2Vec` is that the embeddings are optimized to the specific task and data at hand.
- Embedding size varies based on model variant and size. For example,
    - The Smallest GPT-2 models use embedding size of 768 dimensions.
    - The largest GPT-3 model use an embedding size of 12,288 dimensions.
- To create embeddings for an LLM, first, input text is split into individual tokens. These tokens are either individual words or special characters. <br>
For example, <br>
Input Text: This is an example. <br>
Tokens (set): {"This", "is", "an", "example", "."} <br>
**Note** how special character is also a token.

- Next, a unique token ID is given to each token. For this step, we need to build a `vocabulary` that defines how to map each unique word and special character to a unique integer.
    - We also need special context tokens such as `<|endoftext|>` (used to separate two unrelated text sources).
    - Depending on LLM, tokens such as `[BOS]` (beginning of sequance), `[EOS]` (end of sequence) and `[PAD]` (padding) are considered.
    - However, GOT only uses `<|endoftext|>` token.
    - Is it possible that token or special character is not found in `vocabulary`? Do we need some special token like `<|unk|>`?
        - GPT uses BPE [`byte pair encoding`](https://en.wikipedia.org/wiki/Byte_pair_encoding) based tokenizer, named [`tiktoken`](https://github.com/openai/tiktoken). It breaks words down into subword units.
        - BPE tokenizer used in GPT-2, GPT-3 has a total vocabulary size of 50,257, with `<|endoftext|>` being assigned the largest token ID.
        - BPE tokenizer encodes and decodes unknown words.
        - The algorithm underlying BPE breaks down words that aren’t in its predefined vocabulary into smaller subword units or even individual characters, enabling it to handle out-of-vocabulary words.
        - Example use.
        ```python
        import tiktoken
        text = "The cat sat on mat"
        tokenizer = tiktoken.get_encoding("gpt2")
        enc_text = tokenizer.encode(text)
        print(enc_text)
        ```
        Gives
        ```
        [464, 3797, 3332, 319, 2603]
        ```
- Training `foundation model` is the `next word prediction task`. Input-target pair in this case can be generated using sliding window.
For example, for the text below, <br>
    `The cat sat on the mat` <br>
    input blocks can be extracted as subsample, acting as LLM input and target for LLM is next predicting the next word after input block. <br>
    `464 ---> 3797` <br>
    `464 3797 ---> 3332` <br>
    `464 3797 3332 ---> 319` <br> and so on. <br>
    Everything left of the arrow (`--->`) refers to the input an LLM would receive (after tokenization) and word on right side is what LLM is supposed to predict.
- Creating `token embeddings`.
    - After text tokens are converted into integer token ID's, next step is converting these token IDs into embedding vectors.
    - These embedded vectors are initialized randomly and learned during LLM training process.
    - [`torch.nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) layer can be used to create token embeddings for LLM training.
        - It is a lookup table that stores embeddings of a fixed dictionary and size.
        - Since token embeddings is a simple lookup, irrespective of position of token within the sequence, same embeddings are picked.
    - Check `token_embeddings.py` module for code and explanation.
- Creating `position embeddings`.
    - Token embeddings themselves are suitable input to LLM.
    - However, self-attention mechanism in LLM doesn't have a notion of position or order of tokens within a sequence and token embeddings are not position dependent.
    - To help self attention, it os helpful to inject additional position information into the LLM.
    - Position embedding can be absolute OR relative
        - Absolute positional embeddings are directly associated with specific positions in a sequence. For each position in the input sequence, a unique embedding is added to the token’s embedding to convey its exact location.
        - Emphasis of relative positional embeddings is on the relative position or distance between tokens.This means the model learns the relationships in terms of “how far apart” rather than “at which exact position.” The advantage here is that the model can generalize better to sequences of varying lengths, even if it hasn’t seen such lengths during training.
    - GPT uses absolute position embeddings that are optimized during training.
    - Check `final_embeddings.py` to see how token and position embeddings are added to get final embeddings.
- Final input embeddings is position embeddings added to the token embeddings. <br>
    `input_embeddings = token_embeddings + position_embeddings`





