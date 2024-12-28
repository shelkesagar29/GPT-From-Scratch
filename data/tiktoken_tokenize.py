import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ["create_dataloader_v1"]


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride) -> None:
        self.inpt_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + 1 + max_length]
            self.inpt_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.inpt_ids)

    def __getitem__(self, index):
        return self.inpt_ids[index], self.target_ids[index]


def create_dataloader_v1(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(raw_text)
    print(enc_text[:20])

    """
    [40, 367, 2885, 1464, 1807, 3619, 402, 271, 10899, 2138, 257, 7026, 15632, 438, 2016, 257, 922, 5891, 1576, 438]
    """

    dataloader = create_dataloader_v1(
        raw_text, batch_size=4, max_length=4, stride=4, shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    """
    [tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257]]), tensor([[  367,  2885,  1464,  1807],
        [ 3619,   402,   271, 10899],
        [ 2138,   257,  7026, 15632],
        [  438,  2016,   257,   922]])]
    """
    second_batch = next(data_iter)
    print(second_batch)
    """
    [tensor([[ 922, 5891, 1576,  438],
        [ 568,  340,  373,  645],
        [1049, 5975,  284,  502],
        [ 284, 3285,  326,   11]]), tensor([[5891, 1576,  438,  568],
        [ 340,  373,  645, 1049],
        [5975,  284,  502,  284],
        [3285,  326,   11,  287]])]
    """
