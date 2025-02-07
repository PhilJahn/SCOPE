import numpy as np
from numpy.random import PCG64
from sklearn.preprocessing import MinMaxScaler

def read_file(dsname):
	dic = {}
	classmember = 0


	try:
		if dsname == "covertype":
			file = open("./data/ForestCoverType.csv", "r")
		elif dsname == "kddcup":
			file = open("./data/KDDCup99.csv", "r")
		elif dsname == "gassensor":
			file = open("./data/GasSensorArray.csv", "r")
		elif dsname == "powersupply":
			file = open("./data/Powersupply.csv", "r")
		elif dsname == "starlight":
			file = open("./data/StarLightCurves.csv", "r")
		elif dsname == "densired":
			file = open("./data/densired.csv", "r")
		elif dsname == "rbf3":
			file = open("./data/RBF3_40000.csv", "r")
		else:
			file = open("./data/artificial" + "/" + dsname + ".arff", "r")
	except:
		try:
			file = open("./data/real-world" + "/" + dsname + ".arff", "r")
		except:
			try:
				file = open("./data/ucipp-master/uci/" + dsname + ".arff", "r")
			except:
				raise FileNotFoundError

	x = []
	label = []
	for line in file:
		# print(line)
		if (line.startswith("@") or line.startswith("%") or "class" in line or "duration" in line or len(line.strip()) == 0):
			pass
		else:
			j = line.split(",")
			if ("?" in j):
				continue
			k = []

			if dsname == "kddcup":
				k.append(float(j[0]))
				k.append(j[1])
				k.append(j[2])
				k.append(j[3])
				for i in range(4, len(j) - 1):
					k.append(float(j[i]))
			else:
				for i in range(len(j) - 1):
					k.append(float(j[i]))
			if (not j[len(j) - 1].startswith("noise")):
				clsname = j[len(j) - 1].rstrip()
				if (clsname in dic.keys()):
					label.append(dic[clsname])
				else:
					dic[clsname] = classmember
					label.append(dic[clsname])
					classmember += 1
			else:
				label.append(-1)
			x.append(k)

	if dsname == "kddcup":
		#print(np.sort(np.char.lower(np.unique(np.array(x)[:,2]))))
		#print(np.sort(np.char.lower(np.unique(np.array(x)[:,3]))))

		third_uniques = dict(enumerate(np.char.lower(np.sort(np.unique(np.array(x)[:,2])))))
		#from https://stackoverflow.com/questions/1031851/how-do-i-exchange-keys-with-values-in-a-dictionary
		third_dict = dict((v, k) for k, v in third_uniques.items())
		fourth_uniques = dict(enumerate(np.char.lower(np.sort(np.unique(np.array(x)[:, 3])))))
		fourth_dict = dict((v, k) for k, v in fourth_uniques.items())

		for k in x:
			if k[1] == 'tcp':
				k[1] = 1
			elif k[1] == 'udp':
				k[1] = 2
			elif k[1] == 'icmp':
				k[1] = 3
			else:
				raise ValueError
			k[2] = third_dict[k[2].lower()]
			k[3] = fourth_dict[k[3].lower()]
	return np.array(x), np.array(label).reshape(1, len(label))[0]

def read_synth_drift(dsname):

	if dsname == "movingrbf":
		try:
			data_file = open("./data/drifting/movingRBF.data", "r")
			label_file = open("./data/drifting/movingRBF.labels", "r")
		except:
			raise FileNotFoundError
	else:
		try:
			data_file = open("./data/drifting/rotatingHyperplane.data", "r")
			label_file = open("./data/drifting/rotatingHyperplane.labels", "r")
		except:
			raise FileNotFoundError
	x = []
	label = []

	for line in data_file:
		j = line.split(" ")
		if ("?" in j):
			continue
		k = []

		for i in range(len(j)):
			num = j[i].replace("\n", "")
			k.append(float(num))
		x.append(k)

	for line in label_file:
		label.append(int(line.strip()))

	return np.array(x), np.array(label).reshape(1, len(label))[0]



def read_mnist():
	try:
		file = open("./data/mnist_train.csv", "r")
	except:
		raise FileNotFoundError
	x = []
	label = []
	dic = {}
	classmember = 0
	for line in file:
		# print(line)
		if line.startswith("label") or len(line.strip()) == 0:
			pass
		else:
			if line == "128.3,93.3,2":
				line = "128.3,93.3"
			j = line.split(",")
			if ("?" in j):
				continue
			k = []

			for i in range(1,len(j)):
				k.append(float(j[i]))
			if (not j[0].startswith("noise")):
				clsname = j[0].rstrip()
				if (clsname in dic.keys()):
					label.append(dic[clsname])
				else:
					dic[clsname] = classmember
					label.append(dic[clsname])
					classmember += 1
			else:
				label.append(-1)
			x.append(k)
	return np.array(x), np.array(label).reshape(1, len(label))[0]

def read_subset(dsname):
	X_file_name = "./data/rand_subset/X_"
	y_file_name = "./data/rand_subset/y_"
	suffix = ""
	if "covertype" in dsname:
		suffix = "covertype_subset_11620"
	elif "kddcup" in dsname:
		suffix = "kddcup_subset_4940"
	elif "powersupply" in dsname:
		suffix = "powersupply_subset_5986"
	elif "gassensor" in dsname:
		suffix = "gassensor_subset_6955"
	elif "rotatinghyperplane" in dsname:
		suffix = "rotatinghyperplane_subset_10000"
	elif "movingrbf" in dsname:
		suffix = "movingrbf_subset_10000"
	elif "rbf3" in dsname:
		suffix = "rbf3_subset_8000"
	elif "starlight" in dsname:
		suffix = "starlight_subset_4618"
	for i in range(5):
		if f"_{i}" in dsname:
			suffix += f"_{i}.npy"
	if "_default" in dsname:
		suffix += f"_0.npy"
	X_file_name += suffix
	y_file_name += suffix
	X = np.load(X_file_name)
	y = np.load(y_file_name)
	return X,y



static_list = {"mnist", "optdigits", "pendigits", "densired", "complex9", "diamond9"}
def load_data(dsname, seed = 0):

	print(dsname)
	generator = np.random.Generator(PCG64(seed))
	scaler = MinMaxScaler()
	if "subset" in dsname:  # no need to scale subsampled data
		X, y = read_subset(dsname)
	elif dsname == "mnist":
		X, y = read_mnist()
		X = X/255
	elif dsname == "movingrbf" or dsname == "rotatinghyperplane":
		X, y = read_synth_drift(dsname)
		X = scaler.fit_transform(X)
	else:
		X, y = read_file(dsname)
		X = scaler.fit_transform(X)
	if dsname in static_list:
		idx = generator.permutation(len(X))
		X, y = X[idx], y[idx]
	return X,y