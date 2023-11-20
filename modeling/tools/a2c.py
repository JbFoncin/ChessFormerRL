import torch as t
from torch.nn.utils.rnn import pad_sequence

from chesstools.tools import PADDING_LM_ID

# TODO : implement iterator to convert episode data to A2C chunked batch 