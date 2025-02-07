import numpy as np
from datahandler import read_subset
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from utils import dps_to_np, dict_to_np


def load_mcs(dataset, seed):
	mcs = np.load(f"./param_data/mcs_{dataset}_subset_{seed}.npy", allow_pickle=True)
	print(mcs.shape)
	steps = np.unique(mcs[:,0])
	print(steps)
	assign = np.load(f"./param_data/assign_{dataset}_subset_{seed}.npy")
	data_x, data_y = read_subset(f"{dataset}_subset_{seed}")
	print(data_x.shape, data_y.shape)
	print(assign.shape)
	mindex = 0
	for step in sorted(steps):
		data_x_step = data_x[mindex:step+1]
		data_y_step = data_y[mindex:step+1]

		assign_step = assign[mindex:step+1]

		mcs_step = mcs[mcs[:,0] == step]
		print(dps_to_np(mcs_step[:, 2])[:,0].shape)
		print(mcs_step[:,4].shape)

		mindex = step+1
		#plt.figure(figsize=(10,10))
		#plt.scatter(data_x_step[:,0], data_x_step[:,1], c=data_y_step)
		#plt.show()
		plt.figure(figsize=(10, 10))
		plt.scatter(data_x_step[:, 0], data_x_step[:, 1], c=assign_step)
		plt.scatter(dps_to_np(mcs_step[:, 2])[:,0], dps_to_np(mcs_step[:, 2])[:,1], c="black", s=(mcs_step[:,4]*10).tolist())
		plt.scatter(dps_to_np(mcs_step[:, 2])[:,0], dps_to_np(mcs_step[:, 2])[:,1], c="black", s=(mcs_step[:,4]*10).tolist())

		plt.show()

	#plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y)



if __name__ == '__main__':
	load_mcs("powersupply", 0)