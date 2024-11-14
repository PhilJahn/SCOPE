import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_file(str, i = "uci"):
	dic = {}
	classmember = 0
	if (i == "artificial"):
		file = open("./data/artificial" + "/" + str + ".arff", "r")
	elif i == "real":
		file = open("./data/real-world" + "/" + str + ".arff", "r")
	elif i == "uci":
		#ucipp-master/uci/abalone-3class.arff
		file = open("./data/ucipp-master/uci/" + str + ".arff", "r")
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

def load_data(str, i, seed = 0):
	np.random.seed = seed
	scaler = MinMaxScaler()
	X, y = read_file(str, i)
	X = scaler.fit_transform(X)
	idx = np.random.permutation(len(X))
	X, y = X[idx], y[idx]
	return X,y