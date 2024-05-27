import torch
from torch import nn
from torch.nn import functional as F


def attention(query, key, value, mask = None):
	"""scaled dot product attention function"""

	d_k = torch.tensor(key.size(-1))
	score = torch.matmul(query, key.transpose(-2,-1)) / torch.sqrt(d_k)

	if mask is not None:
		score.masked_fill_(mask, -1e9)

	weight = F.softmax(score, dim = -1)
	attention = torch.matmul(weight, value)

	return attention