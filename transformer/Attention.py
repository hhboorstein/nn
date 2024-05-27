import torch
from torch import nn
from torch.nn import functional as F


def attention(query, key, value, mask = None):
	"""scaled dot product attention function"""

	d_k = torch.tensor(key.size(-1))
	score = torch.matmul(query, key.transpose(-2,-1)) / torch.sqrt(d_k)

	if mask is not None:
		score.masked_fill_(mask, float("-inf"))

	weight = F.softmax(score, dim = -1)
	attention = torch.matmul(weight, value)

	return attention



class MultiHeadAttention(nn.Module):

	def __init__(self, d_model: int, heads: int):

		super().__init__()

		self.d_model = d_model
		self.heads   = heads
		self.d_k 	 = d_model // heads
		assert d_model == heads * self.d_k, "Number of heads must divide embedding dimension (d_model)!"

		self.W_Q = nn.Linear(d_model, d_model, bias = False)
		self.W_K = nn.Linear(d_model, d_model, bias = False)
		self.W_V = nn.Linear(d_model, d_model, bias = False)

		self.W_out = nn.Linear(d_model, d_model, bias = False)


	def forward(self, q, k, v, mask = None):

		batch_size = q.size(0)

		query = self.W_Q(q)
		key   = self.W_K(k)
		value = self.W_V(v)

		mh_query = query.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, sequence_len, d_k)
		mh_key   =   key.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, sequence_len, d_k)
		mh_value = value.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, sequence_len, d_k)

		# implement masking, check shape
		# if mask is not None:
		# 	mask = mask.unsqueeze(1) # what's the shape of this to start?

		mha        = attention(mh_query, mh_key, mh_value, mask = mask)
		concat_mha = mha.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
		output     = self.W_out(concat_mha)

		return output






