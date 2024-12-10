import numpy as np
from numpy.random import PCG64
from sklearn.preprocessing import MinMaxScaler

def read_file(str):
	dic = {}
	classmember = 0
	try:
		file = open("./data/artificial" + "/" + str + ".arff", "r")
	except:
		try:
			file = open("./data/real-world" + "/" + str + ".arff", "r")
		except:
			try:
				file = open("./data/ucipp-master/uci/" + str + ".arff", "r")
			except:
				raise FileNotFoundError

	x = []
	label = []
	for line in file:
		# print(line)
		if (line.startswith("@") or line.startswith("%") or len(line.strip()) == 0):
			pass
		else:
			j = line.split(",")
			if ("?" in j):
				continue
			k = []

			for i in range(len(j) - 1):
				k.append(float(j[i]))
			if (not j[len(j) - 1].startswith("noise")):
				str = j[len(j) - 1].rstrip()
				if (str in dic.keys()):
					label.append(dic[str])
				else:
					dic[str] = classmember
					label.append(dic[str])
					classmember += 1
			else:
				label.append(-1)
			x.append(k)
	return np.array(x), np.array(label).reshape(1, len(label))[0]

def load_data(str, seed = 0):
	#np.random.seed(seed)

	generator = np.random.Generator(PCG64(seed))
	scaler = MinMaxScaler()
	X, y = read_file(str)
	X = scaler.fit_transform(X)
	idx = generator.permutation(len(X))
	X, y = X[idx], y[idx]
	return X,y