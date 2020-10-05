import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init

class DRN(nn.Module):
	r"""Distribution regression as described in the paper
	`A Compact Network Learning Model for Distribution Regression`_ .

	Args:
		in_features: 

		in_bins: 

		out_features: 

		out_bins:

	Shape:
		- Input: :math:`(, , )`
		- Output: :math:`(, , )`

	Examples::
	
		>>> input = torch.normal(0.5, 0.1, (1,200,1))


	.. _`A Compact Network Learning Model for Distribution Regression`:
		https://arxiv.org/abs/1804.04775v3
	"""

	__constants__ = ['in_features', 'in_bins', 'out_features', 'out_bins']

	def __init__(self, in_features, in_bins, out_features, out_bins):
		super(DRN, self).__init__()

		self.in_features = in_features
		self.in_bins = in_bins
		self.out_features = out_features
		self.out_bins = out_bins

		D = self.initD(ql=in_bins, qu=out_bins)
		self.register_buffer('D', D)

		s0 = torch.arange(out_bins, dtype=torch.float32, requires_grad=False)
		s0 = torch.reshape(s0,(1,out_bins))
		self.register_buffer('s0', s0)
		
		self.W = Parameter(torch.Tensor(out_features, in_features))
		self.ba = Parameter(torch.Tensor(out_features, 1))
		self.bq = Parameter(torch.Tensor(out_features, 1))
		self.lama = Parameter(torch.Tensor(out_features, 1))
		self.lamq = Parameter(torch.Tensor(out_features, 1))

		self.reset_parameters()

	def reset_parameters(self):
		init.kaiming_uniform_(self.W, a=math.sqrt(5))

		fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
		bound = 1 / math.sqrt(fan_in)
		init.uniform_(self.ba, -bound, bound)
		init.uniform_(self.bq, -bound, bound)

		init.uniform_(self.lama, 0, 1)
		init.uniform_(self.lamq, 0, 1)

	def forward(self, input):
		return self._forward(input)

	def extra_repr(self):
		return 'in_features={}, in_bins={}, out_features={}, out_bins={}'.format(
			self.in_features, self.in_bins, self.out_features, self.out_bins
		)


	def initD(self, ql, qu):
		# initialize constant matrix D(s1,s0), only depends on q, s1 is upper, s0 is lower
		D = torch.zeros((qu, ql), dtype=torch.float32, requires_grad=False)

		for s1 in range(qu):
			for s0 in range(ql):
				D[s1, s0] = math.exp(-((float(s0)/ql - float(s1)/qu) ** 2))

		D = torch.reshape(D,(qu, ql, 1, 1))
		D = D.repeat(1, 1, self.out_features, self.in_features)
		# D.size() --> (out_bins,in_bins,out_features,in_features)

		return D


	def _forward(self, input):
		# input.size() --> (batch_size,in_features,in_bins)

		Ptile = torch.reshape(input,(-1, 1, self.in_features, self.in_bins, 1))
		Ptile = Ptile.repeat(1, self.out_features, 1, 1, 1) # Ptile.size() --> (batch_size,out_features,in_features,in_bins,1)

		T = torch.pow(self.D, self.W)
		T = T.permute(2, 3, 0, 1) # T.size() --> (out_features,in_features,out_bins,in_bins)

		Pw_unclipped = torch.einsum('jklm,ijkmn->ijkln', T, Ptile)
		Pw_unclipped = torch.squeeze(Pw_unclipped, dim=4) # Pw_unclipped.size() --> (batch_size,out_features,in_features,out_bins)

		# clip Pw by value to prevent zeros when weight is large
		Pw = torch.clamp(Pw_unclipped, min=1e-15, max=1e+15) # Pw.size() --> (batch_size,out_features,in_features,out_bins)

		# perform underflow handling (product of probabilities become small as no. neighbors increase)
		# 1. log each term in Pw
		logPw = torch.log(Pw) # logPw.size() --> (batch_size,out_features,in_features,out_bins)
		# 2. sum over neighbors
		logsum = torch.sum(logPw, dim=2) # logsum.size() --> (batch_size,out_features,out_bins)
		# 3. log of exp of bias terms: log(expB) = exponent_B
		exponent_B = -self.bq * torch.pow(self.s0 / self.out_bins - self.lamq, 2) - self.ba * torch.abs(self.s0 / self.out_bins - self.lama) # exponent_B.size() --> (out_features,out_bins)
		# 4. add B to logsum
		logsumB = logsum + exponent_B # logsumB.size() --> (batch_size,out_features,out_bins)
		# 5. find max over s0
		max_logsum = torch.max(logsumB, dim=2, keepdim=True)[0] # max_logsum.size() --> (batch_size,out_features,out_bins)
		# 6. subtract max_logsum and exponentiate (the max term will have a result of exp(0) = 1, preventing underflow)
		# Now all terms will have been multiplied by exp(-max)
		expm_P = torch.exp(logsumB - max_logsum) # expm_P.size() --> (batch_size,out_features,out_bins)
		# normalize
		Z = torch.sum(expm_P, dim=2, keepdim=True) # Z.size() --> (batch_size,out_features,out_bins)
		ynorm = expm_P / Z # ynorm.size() --> (batch_size,out_features,out_bins)

		return ynorm











