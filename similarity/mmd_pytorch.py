# Compute MMD distance using pytorch

import torch
import torch.nn as nn


# Copyright (c) 2018 Jindong Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Code from https://github.com/jindongwang/transferlearning

class MMD_loss(nn.Module):
	def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
		super(MMD_loss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		self.kernel_type = kernel_type

	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0]) + int(target.size()[0])
		total = torch.cat([source, target], dim=0)
		total0 = total.unsqueeze(0).expand(
			int(total.size(0)), int(total.size(0)), int(total.size(1)))
		total1 = total.unsqueeze(1).expand(
			int(total.size(0)), int(total.size(0)), int(total.size(1)))
		L2_distance = ((total0-total1)**2).sum(2)
		if fix_sigma:
			bandwidth = fix_sigma
		else:
			bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		#print(bandwidth)
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul**i)
						  for i in range(kernel_num)]
		#print(bandwidth_list)
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
					  for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)

	def linear_mmd2(self, f_of_X, f_of_Y):
		loss = 0.0
		delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
		loss = delta.dot(delta.T)
		return loss

	def forward(self, source, target):
		source = torch.from_numpy(source)
		target = torch.from_numpy(target)
		if self.kernel_type == 'linear':
			return self.linear_mmd2(source, target)
		elif self.kernel_type == 'rbf':
			batch_size = int(source.size()[0])
			kernels = self.guassian_kernel(
				source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
			XX = torch.mean(kernels[:batch_size, :batch_size])
			YY = torch.mean(kernels[batch_size:, batch_size:])
			XY = torch.mean(kernels[:batch_size, batch_size:])
			YX = torch.mean(kernels[batch_size:, :batch_size])
			loss = torch.mean(XX + YY - XY - YX)
			return loss
