import os

import numpy as np
import numpy.random as npr
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt

# torch.set_default_dtype(torch.float64)


class PINWHEEL(Dataset):
	def __init__(self, 
		num_per_class, dat_dir, gen=False, plot=False,
		num_classes=5, radial_std=0.15, tangential_std=0.05, rate=0.25
	):
		self.radial_std = radial_std
		self.tangential_std = tangential_std
		self.rate = rate
		self.num_classes = num_classes
		self.num_per_class = num_per_class
		self.length = self.num_classes * self.num_per_class

		self.dat_dir = dat_dir
		file_path = os.path.join(self.dat_dir, "pinwheel.csv")
		# if generate new data
		if gen:
			output = self.sample()
			df = pd.DataFrame(output, columns=['x1', 'x2', 'label'])
			df.to_csv(file_path)
			if plot:
				cmap = plt.get_cmap('gist_rainbow', num_classes)

				plt.figure(figsize=(6.5,6))
				plt.scatter(df['x1'], df['x2'], c=df['label'], cmap=cmap)
				plt.savefig("gen_figs/pinwheel{}.png".format(num_classes))
				plt.colorbar()
				plt.close()

		self.data = pd.read_csv(file_path, index_col=0)
		self.X = self.data.iloc[:, 0:2].values
		self.y = self.data.iloc[:, 2].values


	def sample(self):
	    rads = np.linspace(0, 2*np.pi, self.num_classes, endpoint=False)

	    features = npr.randn(self.num_classes*self.num_per_class, 2) \
	        * np.array([self.radial_std, self.tangential_std])
	    features[:,0] += 1.
	    labels = np.repeat(np.arange(self.num_classes), self.num_per_class)

	    angles = rads[labels] + self.rate * np.exp(features[:,0])
	    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
	    rotations = np.reshape(rotations.T, (-1, 2, 2))

	    data = 10*np.einsum('ti,tij->tj', features, rotations)
	    return np.concatenate([data, labels[:,np.newaxis]], axis=1)


	def __len__(self):
	    return self.length

	def __getitem__(self, idx):
	    sample = [torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)]
	    return sample

def logit_trans(data, lam=1e-6):
	data = lam + (1 - 2 * lam) * data
	return torch.log(data) - torch.log1p(-data)

def inverse_data_transform(config, X):
	if config.data.logit_transform:
		X = logit_trans(X)
	else:
		raise NotImplementedError("Only 'logit_transform'.")
	return X

def data_transform(config, X):
	if config.data.logit_transform:
		X = torch.sigmoid(X)
	else:
		raise NotImplementedError("Only 'logit_transform'.")
	return X


def get_dataset(num_classes, dat_dir, num_per_class, num_per_class_test):
	dataset = PINWHEEL(num_per_class, dat_dir, num_classes=num_classes, gen=True)
	test_dataset = PINWHEEL(num_per_class_test, dat_dir+"_test", num_classes=num_classes, gen=True)

	return dataset, test_dataset

if __name__ == "__main__":
	num_classes = 5
	num_per_class = 5000 // num_classes
	dat_dir = "data"

	# PINWHEEL(num_per_class, dat_dir, gen=True, plot=True)

	batch_size = 2000
	dataloader = DataLoader(
		PINWHEEL(
			num_per_class, dat_dir, num_classes=num_classes, 
			gen=True, plot=True), batch_size=batch_size, shuffle=True)

	# for i, (Xbatch, ybatch) in enumerate(dataloader):
	# 	if i > 0:
	# 		break
	# 	# print(Xbatch)
	# 	# print(ybatch)
	# 	Xbatch = torch.sigmoid(Xbatch)
	# 	Xbatch = logit_trans(Xbatch)
	# 	print(Xbatch)

	# 	plt.figure()
	# 	plt.scatter(Xbatch[:,0], Xbatch[:,1], c=ybatch)
	# 	# plt.savefig("data/pinwheel_{}.png".format(num_classes))
	# 	plt.show()
	# 	plt.close()
