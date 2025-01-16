class full_dataset_leaner:
	def __init__(self):
		self.dataset = []

	def learn_one(self, dp):
		self.dataset.append(dp)

	def offline_processing(self):
		return

	def predict_one(self, dp, return_mc):
		return 0, 0