import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
import pandas as pd

import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import classification_report

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, adjusted_mutual_info_score, roc_auc_score, f1_score
import argparse
import csv
from sklearn.decomposition import PCA

mapped_subcat = {0:"manual mode (0)",
				1:"altitude control mode (1)",
				2:"position control mode (2)",
				3:"mission mode (3)",
				4:"loiter mode (4)",
				5:"return to launch mode (5)",
				6:"RC recovery (6)",
				8:"free slot (8)",
				9:"free slot (9)",
				10:"acro mode (10)",
				11:"free slot (11)",
				12:"descend mode (no position control) (12)",
				13:"termination mode (13)",
				14:"offboard (14)",
				15:"stabilized mode (15)",
				16:"rattitude (16)",
				17:"takeoff (17)",
				18:"land (18)",
				19:"follow (19)",
				20:"precision land with landing target (20)",
				21:"orbit in circle (21)",
				22:"takeoff, transition, establish loiter (22)"}

mapped_cat = {"RC recovery (6)":"auto",
				"manual mode (0)":"manual",
				"altitude control mode (1)":"guided",
				"position control mode (2)":"guided",
				"mission mode (3)":"auto",
				"loiter mode (4)":"auto",
				"return to launch mode (5)":"auto",
				"free slot (8)":"undefined",
				"free slot (9)":"undefined",
				"acro mode (10)":"guided",
				"free slot (11)":"undefined",
				"descend mode (no position control) (12)":"auto",
				"termination mode (13)":"auto",
				"offboard (14)": "auto",
				"stabilized mode (15)":"guided",
				"rattitude (16)":"guided",
				"takeoff (17)":"auto",
				"land (18)":"auto",
				"follow (19)":"auto",
				"precision land with landing target (20)":"auto",
				"orbit in circle (21)":"guided",
					"takeoff, transition, establish loiter (22)":"auto"}

mapped_label = {"guided":0, "auto":1, "manual":2}            

# 32 
# twoD_mean = (0.9791, 0.9873, 0.9930)
# twoD_stdev = (0.0640, 0.0387, 0.0204)
# twoDT_mean = (0.9859, 0.9914, 0.9953)
# twoDT_stdev = (0.0463, 0.0283, 0.0111)

# 64
mean_std_map = {"2D": {"mean": [0.9791, 0.9874, 0.9930], "stdev": [0.0810, 0.0491, 0.0264]},
			   "2DT": {"mean": [0.9859, 0.9914, 0.9953], "stdev": [0.0626, 0.0384, 0.0169]},
			   "2DC": {"mean": [0.9777, 0.9777, 0.9777], "stdev": [0.0700, 0.0700, 0.0700]},
			   "2DP": {"mean": [0.9778, 0.9865, 0.9925], "stdev": [0.0823, 0.0501, 0.0282]},
			   "2DPT": {"mean": [0.9853, 0.9910, 0.9951], "stdev": [0.0639, 0.0392, 0.0166]},
			   # "3DS": {"mean": [240.2531, 246.0379, 250.0554], "stdev": [54.6453, 33.1876, 18.3097]},
			   "3DS": {"mean": [240.6180, 246.2591, 250.1778], "stdev": [54.0128, 32.8087, 18.0915]}}

#128
# twoD_mean = [0.9792, 0.9874, 0.9931]
# twoD_stdev = [0.0991, 0.0599, 0.0326]
# twoDT_mean = [0.9859, 0.9914, 0.9953]
# twoDT_stdev = [0.0799, 0.0492, 0.0243]

shape = 64

def micros_to_secs(micros):
	return micros/1e6

def secs_to_micros(secs):
	return secs*1e6


def to_category(y):
	new_y = [mapped_label[mapped_cat[mapped_subcat[i]]] for i in y]

	return new_y


class CNN(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		if shape == 64:
			self.fc1 = nn.Linear(2704, 120)
		elif shape == 128:
			self.fc1 = nn.Linear(13456, 120)
		self.fc2 = nn.Linear(120, 2)
		self.fc3 = nn.Linear(2, num_classes)
	
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
	
class ImageDataset(Dataset):
	def __init__(self, X, y, transforms):
		self.X = X
		self.y = y
		self.transforms = transforms
	
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, index):
		img, target = self.X[index], self.y[index]
		img = Image.fromarray(img)

		if self.transforms is not None:
			img = self.transforms(img)
		
		return img, target


class CNN3D(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		self.model= nn.Sequential(
		
		#Conv layer 1    
		nn.Conv3d(3, 32, kernel_size=(2, 2, 2), padding=0),
		nn.ReLU(),
		nn.MaxPool3d((2, 2, 2)),   
		
		#Conv layer 2  
		nn.Conv3d(32, 64, kernel_size=(2, 2, 2), padding=0),
		nn.ReLU(),
		nn.MaxPool3d((2, 2, 2)),
			   
		#Flatten
		nn.Flatten(),  
		#Linear 1
		# nn.Linear(57600, 128), 
		nn.Linear(86400, 128),
		#Relu
		nn.ReLU(),
		#BatchNorm1d
		nn.BatchNorm1d(128),
		#Dropout
		nn.Dropout(p=0.15),
		#Linear 2
		nn.Linear(128, num_classes)
		)
	

	def forward(self, x):
		# Set 1
		out = self.model(x)
		return out


class ImageDataset3D(Dataset):
	def __init__(self, filenames, y, transforms=None):
		self.X = filenames
		self.y = y
		self.transforms = transforms
	
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, index):
		file, target = self.X[index], self.y[index]

		with open(file, "rb") as f:
			img = pickle.load(f)

		img = np.transpose(img, (3, 1, 2, 0))

		if self.transforms:
			img = img - self.transforms["mean"]
			img = img / self.transforms["stdev"]
		
		return torch.tensor(img).to(torch.float), torch.tensor(target).to(torch.float)

def create_other_class(X, y, mapping, classes, n_samples=500, drop_other=False):
	new_X = []
	new_y = []
	kept_indices = []
	other_indices = []
	counts = Counter(y)

	modified_counts = {key:value for key, value in dict(counts).items() if value >= n_samples}
	viable_keys = list(modified_counts.keys())

	for i, y in enumerate(y):
		if y not in viable_keys:
			if not drop_other:
				# Keep those not in threshold but rename to other class
				new_y.append(len(viable_keys))
				new_X.append(X[i])
			other_indices.append(i)
		else:
			new_y.append(y)
			new_X.append(X[i])
			kept_indices.append(i)

	new_counts = list(Counter(new_y))

	if not drop_other:
		new_classes = [mapped_subcat[mapping[i]] for i in range(len(new_counts) - 1)]
		new_classes.append("other")
		return new_X, new_y, new_classes, []

	new_classes = [mapped_subcat[mapping[i]] for i in range(len(new_counts))]


	return new_X, new_y, new_classes, kept_indices

def get_raw_data(img_dir, y, mapping, target="cat", small_subset=False):
	files = os.listdir(img_dir)
	sorted_files = []
	indexed_files = {int(file.split("_")[0]):file for file in files if file != ".ipynb_checkpoints"}
	sorted_files = list(dict(sorted(indexed_files.items(), key=lambda x: x[0])).values())

	indices = [int(s.split("_")[0]) for s in sorted_files]

	X = [np.asarray(Image.open(os.path.join(img_dir, file)).convert('RGB')) for file in sorted_files]
	classes = None

	if small_subset:
		# class_counter = {i:100 for i in range(7)}
		# subset_y = []
		# subset_X = []

		# for i in range(len(X)):
		# 	target = y[i]
		# 	inst = X[i]
		# 	if target in class_counter.keys() and class_counter[target] != 0:
		# 		class_counter[target] -= 1
		# 		subset_y.append(target)
		# 		subset_X.append(inst)

		# X = subset_X
		# y = subset_y

		# print(Counter(y))


		with open("X_data.txt", "rb") as f:
			X = pickle.load(f)
		with open("y_data.txt", "rb") as f:
			y = pickle.load(f)		

	y = np.array(y)[indices].tolist()

	if target == "cat":
		cat_test_y = to_category(mapping[i] for i in y)
		reversed_mapping = {value:key for key, value in mapped_label.items()}
		classes = [reversed_mapping[i] for i in list(Counter(cat_test_y).keys())]
		y = cat_test_y
	elif target == "subcat":
		classes = [mapped_subcat[mapping[i]] for i in range(len(mapping))]
		X, y, classes, _ =  create_other_class(X, y, mapping, classes)


	return X, y, classes

def get_raw_data3D(img_dir, y, mapping, target):
	files = os.listdir(img_dir)
	sorted_files = []
	indexed_files = {int(file.split("_")[1][:-4]):os.path.join(img_dir, file) 
					for file in files if file.split(".")[1] != "png" and file != ".ipynb_checkpoints"}
	X = list(dict(sorted(indexed_files.items(), key=lambda x: x[0])).values())	
	print(len(X))
	y = y[:len(X)]

	if target == "cat":
		cat_test_y = to_category(mapping[i] for i in y)
		reversed_mapping = {value:key for key, value in mapped_label.items()}
		classes = [reversed_mapping[i] for i in list(Counter(cat_test_y).keys())]
		y = cat_test_y
	elif target == "subcat":
		classes = [mapped_subcat[mapping[i]] for i in range(len(mapping))]
		X, y, classes, _ =  create_other_class(X, y, mapping, classes)

	return X, y, classes

def get_dataloaders(data, image_type, normalize=False):
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


	if normalize:
		mean = mean_std_map[image_type]["mean"]
		stdev = mean_std_map[image_type]["stdev"]
		if image_type in ["3DS"]:
			reshaped_mean = np.array(mean)[:, np.newaxis, np.newaxis, np.newaxis]
			reshaped_stdev = np.array(mean)[:, np.newaxis, np.newaxis, np.newaxis]
			transform = {"mean": reshaped_mean, "stdev": reshaped_stdev}
		else:
			transform = transforms.Compose([
				transforms.Resize((shape,shape)),
				transforms.ToTensor(),
				transforms.Normalize(mean, stdev)
			])
	else:
		if image_type in ["3DS"]:
			transform = None
		else:
			transform = transforms.Compose([
				transforms.Resize((shape,shape)),
				transforms.ToTensor()
			])

	if image_type in ["3DS"]:
		train_dataset = ImageDataset3D(data["X_train"], data["y_train"], transform)
		test_dataset = ImageDataset3D(data["X_test"], data["y_test"], transform)	
		train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
		test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
	else:		
		train_dataset = ImageDataset(data["X_train"], data["y_train"], transform)
		test_dataset = ImageDataset(data["X_test"], data["y_test"], transform)
		train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
		test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

	return train_dataloader, test_dataloader

def get_mean_and_std(data_loader, image_type):
	batch_sum = 0
	batch_sum_sq = 0
	mean = 0
	num_batches = 0

	if image_type in ["3DS"]:
		dim = [0, 2, 3, 4]
	else:
		dim = [0, 2, 3]

	# calculate mean over width, height, and batch size
	for (_, data) in enumerate(data_loader):
		batch_sum += torch.mean(data[0], dim=dim)
		batch_sum_sq += torch.mean(data[0]**2, dim=dim)
		num_batches += 1

	mean = batch_sum / num_batches
	std = (batch_sum_sq / num_batches - mean**2)**0.5
	print(mean)
	print(std)
	return mean, std

def write_averages(file_name, results, stacks, n_folds, classes):
	means = np.mean(stacks["report_stack"], axis=0)
	stds = np.std(stacks["report_stack"], axis=0)
	macro_mean = np.mean(stacks["macro_f1_stack"])
	macro_std = np.std(stacks["macro_f1_stack"])
	conf_mean = stacks["conf_mat_stack"]/n_folds

	if file_name:

		with open(file_name, 'a', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow(["\n"]) 
			csv_writer.writerow(["Averages"])           
			csv_writer.writerow([""] + [c for c in classes])
			csv_writer.writerow(["Precision"] + [round(means[0][i], 4) for i in range(len(classes))])
			csv_writer.writerow(["Recall"] + [round(means[1][i], 4) for i in range(len(classes))])
			csv_writer.writerow(["F-score"] + [round(means[2][i], 4) for i in range(len(classes))])
			csv_writer.writerow(["Macro F1 Score", round(macro_mean, 4)])

			csv_writer.writerow(["\n"]) 
			csv_writer.writerow(["Standard Deviations"])           
			csv_writer.writerow([""] + [c for c in classes])
			csv_writer.writerow(["Precision"] + [round(stds[0][i], 4) for i in range(len(classes))])
			csv_writer.writerow(["Recall"] + [round(stds[1][i], 4) for i in range(len(classes))])
			csv_writer.writerow(["F-score"] + [round(stds[2][i], 4) for i in range(len(classes))])
			csv_writer.writerow(["Macro F1 Score", round(macro_std, 4)])

			csv_writer.writerow(["\n"])
			csv_writer.writerow([""] + [c for c in classes])  
			for i in range(len(results["conf_mat"])):	
				csv_writer.writerow([classes[i]] + conf_mean[i].tolist())
	else:
		print(means)

def write_per_fold(fold, file_name, results, classes, total_conf_mat, report_stack, macro_f1_stack):
	total_conf_mat = np.add(total_conf_mat, results["conf_mat"])
		

	for i, metric in enumerate(["precision", "recall", "f1-score"]):
		for j, c in enumerate(classes):
			report_stack[fold][i][j] = results["report"][c][metric]

	# auc_stack.append(auc_score)
	macro_f1_stack.append(results["macro_f1"])

	return total_conf_mat, report_stack, macro_f1_stack 

def train(classes, train_dataloader, test_dataloader, params, model):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])
	progress_bar = tqdm(range(params["num_epochs"]))

	# num_correct = 0
	# true_size = 0

	# pred_from_last_epoch = []

	val_loss = []

	for epoch in progress_bar:
		y_true = []
		y_pred = []
		loss_sum = []

		for phase in ("train", "eval"):
			if phase == "train":
				model.train()
				data_loader = train_dataloader
			else:
				model.eval()
				data_loader = test_dataloader

			for (_, data) in enumerate(data_loader):
				# print(data)
				optimizer.zero_grad()
				inputs = data[0].to(device)
				# print(inputs.shape)
				targets = data[1]
				# print(targets)
				targets = targets.to(device)

				with torch.set_grad_enabled(phase=="train"):
					predictions = model(inputs)
					# print(predictions.shape)
					loss = criterion(predictions, targets.long())

					if phase == "train":
						loss.backward()
						optimizer.step()

				out, max_indices = torch.max(predictions, dim=1)

				if phase == "eval":
					y_true += targets.tolist()
					y_pred += max_indices.tolist()
					loss_sum.append(loss.item())

	#                 # print(predictions)

				# num_correct += torch.sum(max_indices == targets.long()).item()

	#             # true_size += targets.size()[0]
	#             # print(targets.size())            

			if phase == "eval" and (epoch + 1) % params["num_epochs"]/4 == 0:
				val_loss.append(np.mean(np.array(loss_sum)))
				counts = Counter(y_pred)
				report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
				print(report)
				conf_mat = confusion_matrix(y_true, y_pred)
				macro_f1 = f1_score(y_true, y_pred, average='macro')

	results = {"macro_f1":macro_f1, "counts":counts, "report":report, "conf_mat":conf_mat}	

	print(results)

	return results

def main():
	with open("../experiment_original/mapping.txt", "rb") as f:
		mapping = pickle.load(f)
	with open("../experiment_original/y_data.txt", "rb") as f:
		all_y = pickle.load(f)

	parser = argparse.ArgumentParser()
	parser.add_argument('-c','--csv', type=str, help='csv file name output')
	parser.add_argument('-d','--dtype', type=str, help='2D, 2DT, 3DS')
	parser.add_argument("-t", "--target", type=str, help="subcategory or category", required=True)
	parser.add_argument("-e", "--n_epochs", type=int, help="number of epochs", default=40)
	parser.add_argument("-s", "--subset", action='store_true', help="use small subset", default=False)

	args = parser.parse_args()


	if args.dtype in ["3DS"]:
		# img_dir =  "../../../../../../../work/uav-ml/3DS_images"
		img_dir =  "../../../../../../../work/uav-ml/3DS_images_30"
		print(len(os.listdir(img_dir)))
		X, y, classes = get_raw_data3D(img_dir, all_y, mapping, target=args.target)
	else:
		img_dir = f"{args.dtype}_images"
		print(len(os.listdir(img_dir)))
		X, y, classes = get_raw_data(img_dir, all_y, mapping, target=args.target, small_subset=args.subset)


	n_folds = 3
	fold = 0
	file_name = args.csv

	report_stack = np.zeros((n_folds, 3, len(classes)))
	macro_f1_stack = []
	total_conf_mat = np.zeros((len(classes), len(classes)))	
	
	kf = StratifiedKFold(n_splits=n_folds)
	splits = kf.split(X, y)

	for train_index, test_index in splits:

		data = {"X_train": np.array(X)[train_index], "X_test": np.array(X)[test_index],
				"y_train": np.array(y)[train_index], "y_test": np.array(y)[test_index]}

		train_dataloader, test_dataloader = get_dataloaders(data, image_type=args.dtype, normalize=True)
		# mean, std = get_mean_and_std(train_dataloader, image_type=args.dtype)

		params = {"lr": 0.001, "num_epochs": args.n_epochs}

		if args.dtype in ["3DS"]:
			model = CNN3D(num_classes=len(classes))
		else:
			model = CNN(num_classes=len(classes))

		results = train(classes, train_dataloader, test_dataloader, params, model)

		total_conf_mat, report_stack, macro_f1_stack = write_per_fold(fold, file_name, results, classes,
																	  total_conf_mat, report_stack, macro_f1_stack)
		# fig = plt.figure()
		# plt.plot(list(range(params["num_epochs"])), val_loss)
		# fig.savefig("2DT_val_loss_subcat.png")

		fold += 1
		break
	
	stacks = {"report_stack": report_stack, "macro_f1_stack": macro_f1_stack, "conf_mat_stack": total_conf_mat}
	write_averages(file_name, results, stacks, n_folds, classes)

if __name__ == "__main__":
	main()
