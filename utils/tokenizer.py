import torch

from typing import Union, Tuple, List


class TokenizerForCTC(object):
    """A class for tokenizing and detokenizing strings for use in a Connectionist
        Temporal Classification (CTC) model.

    Args:
        tokens: A string containing the characters to be used as tokens.

    Attributes:
        tokens (str): A string containing the characters to be used as tokens.
        UNKNOWN_token (int): An integer representing the unknown token.
        n_token (int): The number of tokens, including the unknown token.
        char_idx (dict): A dictionary mapping characters to token indices.
        idx_char (dict): A dictionary mapping token indices to characters.
    """

    def __init__(self, tokens: str):
        self.tokens = tokens
        self.UNKNOWN_token = 0
        self.n_token = len(tokens) + 1
        self.char_idx = {}

        for i, char in enumerate(tokens):
            self.char_idx[char] = i + 1

        self.idx_char = {v: k for k, v in self.char_idx.items()}

    def tokenize(self, text: str) -> Tuple[torch.LongTensor, int]:
        """Tokenizes a string by converting it to a list of token indices.

        Args:
            text: The string to be tokenized.

        Returns:
            A tuple containing a tensor of token indices and the length of the tensor.
        """
        result = []

        for char in text:
            result.append([self.char_idx[char]])

        return torch.LongTensor(result), len(result)

    def int2text(
            self,
            word_indexes: torch.Tensor,
            length: torch.Tensor
    ) -> List:
        """Converts a tensor of token indices and a tensor of lengths back into a list of strings.

        Args:
            word_indexes: A tensor of token indices.
            length: A tensor of lengths.

        Returns:
            A list of strings.
        """
        texts = []
        for row, l in zip(word_indexes, length):
            row = row[:l]
            text = "".join(self.idx_char[int(i)] for i in row)
            texts.append(text)
        return texts

    def translate(
            self,
            t: torch.Tensor,
            length: torch.Tensor,
            raw: bool = False
    ) -> Union[List, str]:
        """Converts a tensor of token indices and a tensor of lengths into a list of strings or a single string.

        Args:
            t: A tensor of token indices.
            length: A tensor of lengths.
            raw: If True, the resulting string will not be filtered for repeated or unknown tokens.
                Defaults to False.

        Returns:
            A list of strings or a single string, depending on the shape of the input tensors.

        Raises:
            ValueError: If the number of elements in `t` does not match the sum of the elements in `length`.
        """
        if length.numel() == 1:
            length = length.squeeze()
            assert t.numel() == length, (
                "text with length: {} does not match "
                "declared length: {}".format(t.numel(), length)
            )
            if raw:
                return "".join([self.tokens[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.tokens[t[i] - 1])
                return "".join(char_list)
        else:
            assert t.numel() == length.sum(), (
                "texts with length: {} does not match "
                "declared length: {}".format(t.numel(), length.sum())
            )
            texts = []
            index = 0
            for i in range(length.numel()):
                char_l = length[i]
                texts.append(
                    self.translate(
                        t[index : index + char_l],
                        torch.LongTensor([char_l]),
                        raw=raw,
                    )
                )
                index += char_l
            return texts
