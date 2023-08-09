import os
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, adjusted_mutual_info_score, roc_auc_score, f1_score
import pickle
from collections import Counter
import argparse
from sklearn.model_selection import KFold, StratifiedKFold
import random 
from itertools import combinations
import csv
import random
import numpy as np
import flight_mode_preprocess as fmp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Classical models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import torchvision
from torchvision import transforms
import torch.nn.functional as F


feats = ["vehicle_local_position | x", "vehicle_local_position | y",
		 "vehicle_local_position | z", "vehicle_attitude_setpoint | roll_body",
		 "vehicle_attitude_setpoint | pitch_body", "vehicle_attitude_setpoint | yaw_body",
		 "manual_control_setpoint | z", "vehicle_gps_position | alt", "battery_status | temperature"]

mapped_subcat = {-1: "other",
				0:"manual mode (0)",
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

ulog_folder = "../../../../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"
ulog_folder_hex = "../../../../../../work/uav-ml/px4-Ulog-Parsers/dataDownloadedHex"
json_file = "../../../../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"

##### PYTORCH DATA FUNCTIONS #####

class UAVDataset(Dataset):
	"""
	UAV Dataset class

	...

	Attributes
	----------
	X : list
		Parsed and feature engineered time series data
	y : list
		UAV labels

	Methods
	-------
	len():
		gets the size of the dataset
	getitem(index):
		gets an indexed instance
	"""
	def __init__(self, X, y):
		self.X = X
		self.y = y
						
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, index):
		return torch.Tensor(self.X[index]), torch.tensor(self.y[index]).float()   


def get_dataloaders(data):
	# Form datasets and dataloaders
	train_dataset = UAVDataset(data["X_train"], data["y_train"])
	test_dataset = UAVDataset(data["X_test"], data["y_test"])

	train_loader = DataLoader(train_dataset,
							  batch_size=8,
							  shuffle=False)
	test_loader = DataLoader(test_dataset,
							  batch_size=1,
							  shuffle=False)

	return train_loader, test_loader


##### CONCAT/CLASSICAL MODELS ##### 

def reshape_data(X_train, X_test):
	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

	return X_train, X_test

def dim_reduce_data(X_train, X_test, dim_reduce):
	model = PCA(n_components=dim_reduce, svd_solver='arpack').fit(X_train)
	# model = TSNE(n_components=dim_reduce).fit(X_train)


	X_train = model.transform(X_train)
	X_test = model.transform(X_test)

	print(X_train.shape)
	print(X_test.shape)

	return X_train, X_test

def apply_concat_models(classes, data, model_name="SVM", verbose=True, reshape=True, dim_reduce=None, adjust_with_prob=False):
	if reshape:
		X_train, X_test = reshape_data(data["X_train"], data["X_test"])
	else:
		X_train, X_test = data["X_train"], data["X_test"]

	if dim_reduce:
		X_train, X_test = dim_reduce_data(X_train, X_test, dim_reduce)

	if model_name == "SVM":
		# model = SVC(probability=True)
		model = SVC()
	elif model_name == "RFC":
		model = RandomForestClassifier()
	elif model_name == "XGBC":
		model = XGBClassifier(n_estimators=200)
	elif model_name == "GBC":
		model = GradientBoostingClassifier()
	elif model_name == "NN":
		model = MLPClassifier()
	elif model_name == "VOTE":
		# model = VotingClassifier(estimators=[('xgb',  XGBClassifier()),
		# 									 ('rf', RandomForestClassifier()),
		# 									 ('gb', GradientBoostingClassifier())], voting='hard')
		model = VotingClassifier(estimators=[('nb',  GaussianNB()),
											 ('svm', SVC()),
											 ('dtc', DecisionTreeClassifier())], voting='hard')
	elif model_name == "LGBM":
		model = LGBMClassifier()

	model = model.fit(X_train, data["y_train"])
	y_pred = model.predict(X_test)		


	

	counts = Counter(y_pred)
	report = classification_report(data["y_test"], y_pred, target_names=classes, output_dict=True)
	macro_f1 = f1_score(data["y_test"], y_pred, average='macro')
	conf_mat = confusion_matrix(data["y_test"], y_pred)

	if verbose:
		print(counts)
		print(report)
		print(macro_f1)
		print(conf_mat)

	if model_name in ["RFC", "XGBC"]:	
		return macro_f1, counts, report, conf_mat, model.feature_importances_, y_pred

	if adjust_with_prob:
		y_pred_prob = model.predict_proba(X_test)
		probs = y_pred_prob[:, y_pred]
		adjusted_pred = probs * y_pred
		y_pred = adjusted_pred

	return macro_f1, counts, report, conf_mat, [], y_pred


##### LSTM ##### 

class LSTM(nn.Module):
	"""
	LSTM class

	...

	Attributes
	----------
	input_size : int
		number of instances
	hidden_size : int
		hidden size of LSTM layer
	num_classes : int
		number of classes to predict
	num_layers : int
		number of LSTM layers

	Methods
	-------
	forward():
		forward propagation of LSTM
	"""
	def __init__(self, input_size, hidden_size, num_classes, num_layers):
		super(LSTM, self).__init__()
		self.LSTM = nn.LSTM(input_size=input_size,
							hidden_size=hidden_size,
							num_layers=num_layers,
							batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)
	
	def forward(self, x):
		# print(x[0])
		_, (hidden, _) = self.LSTM(x)
		out = hidden[-1]
		# print(out)
		# print(out.shape)
		out = self.fc(out)
		# print(out)
		# print(out.shape)
		return out

def get_model_lstm(input_size, hidden_size = 128, num_classes=2, num_layers=1):
	hidden_size = 128
	num_layers = 1
	model = LSTM(input_size=input_size,
				 hidden_size=hidden_size,
				 num_classes=num_classes,
				num_layers=num_layers)

	return model

##### CNN #####

class CNN(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 2)
		self.fc1 = nn.Linear(352, 120)
		self.fc2 = nn.Linear(120, num_classes)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = F.relu(self.conv2(x))
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

def get_model_cnn(num_classes):
	model = CNN(num_classes)

	return model


def apply_NNs(classes, model, train_loader, test_loader, params, verbose=True, test_index=None):
	'''
	Trains the LSTM 

			Parameters:
					model (object): Pytorch model
					train_loader (object): Iterable train loader
					test_loader (object): Iterable test loader
					params (dict): Model and training params

			Returns:
					pred_from_last_epoch (list) : Predictions from the last epoch
	'''
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
	progress_bar = tqdm(range(params["num_epochs"]))
	
	num_correct = 0
	true_size = 0

	pred_from_last_epoch = []
	
	for epoch in progress_bar:
		y_true = []
		y_pred = []
		for phase in ("train", "eval"):
			if phase == "train":
				model.train()
				data_loader = train_loader
			else:
				model.eval()
				data_loader = test_loader
				
			for (_, data) in enumerate(data_loader):
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
						# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
						optimizer.step()

				out, max_indices = torch.max(predictions, dim=1)

				if phase == "eval":
					y_true += targets.tolist()
					y_pred += max_indices.tolist()

					# print(predictions)
				
				# num_correct += torch.sum(max_indices == targets.long()).item()

				# true_size += targets.size()[0]
				# print(targets.size())

		if phase == "eval" and (epoch + 1) % params["num_epochs"]/4 == 0 :
			counts = Counter(y_pred)
			report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
			auc_score = 0
			# auc_score = roc_auc_score(y_true, y_pred, multi_class='ovo')
			macro_f1 = f1_score(y_true, y_pred, average='macro')
			conf_mat = confusion_matrix(y_true, y_pred)

			if verbose:
				print(counts)
				print(classification_report(y_true, y_pred, target_names=classes))
				print(auc_score)
				print(conf_mat)


		if phase == "eval" and epoch == params["num_epochs"] - 1:
			pred_from_last_epoch = y_pred

	return macro_f1, counts, model, report, conf_mat, pred_from_last_epoch



# def generate_true(y_true, y_pred, cat_subcat_test):
# 	new_true = [cat_subcat_test[i][1] for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)) if y_p == y_t]


def train_two_part(classes, model, train_loader, test_loader, params, cat_subcat_test, verbose=True, test_index=None):
	'''
	Trains the LSTM 

			Parameters:
					model (object): Pytorch model
					train_loader (object): Iterable train loader
					test_loader (object): Iterable test loader
					params (dict): Model and training params

			Returns:
					pred_from_last_epoch (list) : Predictions from the last epoch
	'''

	for (_, data) in enumerate(train_loader):
		print(data[0].tolist())

	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# model = model.to(device)
	# criterion = nn.CrossEntropyLoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
	# progress_bar = tqdm(range(params["num_epochs"]))
	
	# num_correct = 0
	# true_size = 0

	# pred_from_last_epoch = []
	
	# model.train()
	# for epoch in progress_bar:			
	# 	for (_, data) in enumerate(train_loader):
	# 		optimizer.zero_grad()
	# 		inputs = data[0].to(device)
	# 		targets = data[1]
	# 		# print(targets)
	# 		targets = targets.to(device)

	# 		with torch.set_grad_enabled(phase=="train"):
	# 			predictions = model(inputs)

	# 			# print(predictions.shape)
	# 			loss = criterion(predictions, targets.long())
	# 			loss.backward()
	# 			optimizer.step()


				
	# for (_, data) in enumerate(test_loader):
	# 	inputs = data[0].to(device)
	# 	targets = data[1]
	# 	out, max_indices = torch.max(predictions, dim=1)
	# 	y_pred += max_indices.tolist()

	return pred_from_last_epoch, auc_score, macro_f1, counts, model, report, conf_mat



# def modify_for_folds(X, y, mapping, classes, n_folds=5):
# 	new_X = []
# 	new_y = []
# 	removed_classes = set()

# 	counts = Counter(y)
# 	modified_counts = {key:value for key, value in dict(counts).items() if value >= n_folds}
# 	viable_keys = list(modified_counts.keys())

# 	for i, y in enumerate(y):
# 		if y in viable_keys:
# 			new_X.append(X[i])
# 			new_y.append(y)
# 		else:
# 			removed_classes.add(y)

# 	new_classes = [mapped_subcat[mapping[i]] for i in range(len(mapping)) if i not in removed_classes]

# 	return new_X, new_y, new_classes

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

def create_two_class(X, y, mapping, classes, n_samples=500):
	new_X = []
	new_y = []
	counts = Counter(y)

	modified_counts = {key:value for key, value in dict(counts).items() if value >= n_samples}
	viable_keys = list(modified_counts.keys())	

	for i, y in enumerate(y):
		if y not in viable_keys:
			new_y.append(1)
		else:
			new_y.append(0)

		new_X.append(X[i])

	new_classes = ["non_other", "other"]

	return new_X, new_y, new_classes

def get_only_other(X, y, mapping, n_samples_max=500, n_samples_min=5):
	new_X = []
	new_y = []
	other_indices = []
	counts = Counter(y)
	non_other_count = 5

	modified_counts = {key:value for key, value in dict(counts).items() 
					  if value <= n_samples_max and value >= n_samples_min} 
	total_classes = len(counts)
	viable_keys = list(modified_counts.keys())

	for i, y in enumerate(y):
		if y in viable_keys:
			new_y.append(y)
			new_X.append(X[i])


	new_y, new_mapping = fmp.remap_y(new_y)

	print(Counter(new_y))

	indices = [i for i in range(len(viable_keys))]
	new_classes = [mapped_subcat[mapping[new_mapping[i]]] for i in indices]

	print(new_classes)

	return new_X, new_y, new_classes

def append_y_to_x(X_train, X_test, y_train, y_test):
	X_train, X_test = reshape_data(X_train, X_test)
	X_train = X_train.tolist()
	X_test = X_test.tolist()

	for x, y in zip(X_train, y_train):
		x.append(y)

	for x, y in zip(X_test, y_test):
		x.append(y)

	return np.array(X_train), np.array(X_test)

def get_sorted_rankings(feat_import, append=False):
	sorted_import = feat_import.argsort()
	temp =  [feat for feat in feats for i in range(50)]
	if append:
		temp.append("appended class")					
	features_sorted = np.array(temp)[sorted_import]

	avg_rankings = {}
	for i, feat in enumerate(features_sorted.tolist()):
		if feat in avg_rankings:
			avg_rankings[feat] += i
		else:
			avg_rankings[feat] = i

	avg_rankings = {key:value/50 for key, value in avg_rankings.items()}
	sorted_avg_rankings = dict(sorted(avg_rankings.items(), key=lambda x : x[1]))
	sorted_avg_rankings_ls = [(key, value) for key, value in sorted_avg_rankings.items()]

	return sorted_avg_rankings_ls

def test_append(target, mapping, data, train_index, test_index):
	reshape = False
	if target == "subcategory":
		y_cat_train = fmp.to_category([mapping[i] for i in data["y_train"]])
		y_cat_test = fmp.to_category([mapping[i] for i in data["y_test"]])
		X_train, X_test = append_y_to_x(data["X_train"], data["X_test"], y_cat_train, y_cat_test)
		print(data["X_train"][:10])
		print(X_train[:10])
	elif target == "category":
		y_subcat_train = np.array(y_subcat)[train_index]
		y_subcat_test = np.array(y_subcat)[test_index]
		X_train, X_test = append_y_to_x(data["X_train"], data["X_test"], y_subcat_train, y_subcat_test)

	return X_train, X_test

def test_two_part(target, model, data):
	if target == "category":
		X_train, X_test = fmp.standardize_data(data["X_train"], data["X_test"])
		cat_data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}				
		macro_f1, counts, report, conf_mat, feat_import, y_pred = apply_concat_models(classes, cat_data, model_name=model)

		predictions_mapping_pred.update(dict(zip(test_index, y_pred)))
		with open("predictions_mapping.txt", "wb") as f:
			pickle.dump(predictions_mapping_pred, f)

		# predictions_mapping_true.update(dict(zip(test_index, y_test)))
		# with open("predictions_mapping_true.txt", "wb") as f:
		# 	pickle.dump(predictions_mapping_true, f)

	elif target == "subcategory":
		with open("predictions_mapping.txt", "rb") as f:
			predictions_mapping = pickle.load(f)

		# with open("predictions_mapping_true.txt", "rb") as f:
		# 	predictions_mapping = pickle.load(f)

		X_train, X_test = fmp.standardize_data(data["X_train"], data["X_test"])


		y_pred_test = [predictions_mapping[ind] for ind in test_index]
		y_pred_train = [predictions_mapping[ind] for ind in train_index]

		X_train, X_test = append_y_to_x(data["X_train"], data["X_test"], y_pred_train, y_pred_test)

		subcat_data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
		macro_f1, counts, report, conf_mat, feat_import, y_pred = apply_concat_models(classes, subcat_data, model_name=model, reshape=False)
		
		# if args.model in ["RFC", "XGBC"]:
		# 	sorted_avg_rankings_ls = get_sorted_rankings(feat_import, append=True)

	results = {"macro_f1":macro_f1, "counts":counts, "report":report,
			  "conf_mat":conf_mat, "feat_import":[], "y_pred":y_pred}

	return results


def test_standard(model, params, X_train, X_test, y_train, y_test, train_index, test_index):
	X_train, X_test = fmp.standardize_data(X_train, X_test)
	data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}	

	if model in ["LSTM", "CNN"]:

		if model == "CNN":
			data["X_train"] = data["X_train"][:, None, :]
			data["X_test"] = data["X_test"][:, None, :]

		train_loader, test_loader = get_dataloaders(data)
		input_size = X_train.shape[2]

		if model == "LSTM":
			model = get_model_lstm(input_size, num_classes=len(params["classes"]), num_layers=params["num_layers"])
		else:
			model = get_model_cnn(num_classes=params["num_classes"])

		macro_f1, counts, model, report, conf_mat, y_pred = apply_NNs(params["classes"], model, train_loader, test_loader, params)
		# _, auc_score, macro_f1, counts, model, report, conf_mat = train_two_part(classes, model, train_loader, test_loader, params, cat_subcat[test_index])

	else:
		macro_f1, counts, report, conf_mat, feat_import, y_pred = apply_concat_models(params["classes"], data, model_name=model, reshape=True, dim_reduce=False)
		
		# predictions_mapping_pred.update(dict(zip(test_index, y_pred)))
		# predictions_mapping_true.update(dict(zip(test_index, y_test)))

		# with open("predictions_mapping_pred_cat_orig.txt", "wb") as f:
		# 	pickle.dump(predictions_mapping_pred, f)	

		# with open("predictions_mapping_true_cat_orig.txt", "wb") as f:
		# 	pickle.dump(predictions_mapping_true, f)


		# if args.model in ["RFC", "XGBC"]:
		# 	sorted_avg_rankings_ls = get_sorted_rankings(feat_import, append=args.append)
	results = {"macro_f1":macro_f1, "counts":counts, "report":report,
			  "conf_mat":conf_mat, "feat_import":feat_import, "y_pred":y_pred}

	return results

def preprocess_data(arguments):
	if arguments["equal"]:
		with open("X_data_equal.txt", "rb") as f:
			X = pickle.load(f)

		with open("y_data_equal.txt", "rb") as f:
			y_subcat = pickle.load(f)

		with open("mapping_equal.txt", "rb") as f:
			mapping = pickle.load(f)

	else:
		with open(arguments["x_data"], "rb") as f:
			X = pickle.load(f)
		with open(arguments["y_data"], "rb") as f:
			y_subcat = pickle.load(f)
		with open(arguments["mapping"], "rb") as f:
			mapping = pickle.load(f)



	if arguments["target"] == "category":
		# Convert subcategory back to its original id, then use that to get category
		y_cat = fmp.to_category([mapping[i] for i in y_subcat])
		cat_subcat = [(y_c, y_s) for y_c, y_s in zip(y_cat, y_subcat)]
		y = y_cat

		# Convert category to its labels
		reversed_mapping = {value:key for key, value in mapped_label.items()}
		classes = [reversed_mapping[i] for i in list(Counter(y_cat).keys())]

	elif arguments["target"] == "subcategory":
		y = y_subcat
		classes = [mapped_subcat[mapping[i]] for i in range(len(mapping))]
		X, y, classes, _ = create_other_class(X, y, mapping, classes, drop_other=arguments["drop_other"])

	elif arguments["target"] == "other":
		y = y_subcat
		X, y, classes = get_only_other(X, y, mapping)

	elif arguments["target"] == "two_class":
		y = y_subcat
		classes = [mapped_subcat[mapping[i]] for i in range(len(mapping))]
		X, y, classes = create_two_class(X, y, mapping, classes)

		

	if arguments["indices"]:
		X = np.array(X)[:, arguments["indices"], :]
		y = np.array(y)
	else:
		X = np.array(X)
		y = np.array(y)
		if arguments["drop_other"] and arguments["target"] == "category":
			_, _, _, kept_indices = create_other_class(X, y_subcat, mapping,
									[mapped_subcat[mapping[i]] for i in range(len(mapping))],
									drop_other=arguments["drop_other"])
			X = X[kept_indices, :, :]
			y = y[kept_indices]
			classes = ["guided", "auto"]

	return X, y, classes, mapping

def write_averages(file_name, results, stacks, n_folds, classes):
	means = np.mean(stacks["report_stack"], axis=0)
	stds = np.std(stacks["report_stack"], axis=0)
	macro_mean = np.mean(stacks["macro_f1_stack"])
	macro_std = np.std(stacks["macro_f1_stack"])
	conf_mean = stacks["conf_mat_stack"]/n_folds


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

def write_per_fold(fold, file_name, results, classes, total_conf_mat, report_stack, macro_f1_stack):
	# with open(file_name, 'a', newline='') as csvfile:
	# 	csv_writer = csv.writer(csvfile)
	# 	csv_writer.writerow(["\n"])            
	# 	csv_writer.writerow(["Fold : " + str(fold)])
	# 	csv_writer.writerow([""] + [c for c in classes])
	# 	csv_writer.writerow(["Precision"] + [round(report[c]["precision"], 4) for c in classes])
	# 	csv_writer.writerow(["Recall"] + [round(report[c]["recall"], 4) for c in classes])
	# 	csv_writer.writerow(["F-score"] + [round(report[c]["f1-score"], 4) for c in classes])
	# 	csv_writer.writerow(["Macro F1 Score", round(macro_f1, 4)])
	# 	csv_writer.writerow(["True Counts"] + [round(report[c]["support"], 4) for c in classes])
	# 	csv_writer.writerow(["Pred Counts"] + [counts[i] for i, c in enumerate(classes)])

	# 	# if args.model in ["RFC", "XGBC"]:
	# 	# 	csv_writer.writerow(["\n"])
	# 	# 	csv_writer.writerow(["Feature"] + list(list(zip(*sorted_avg_rankings_ls))[0]))
	# 	# 	csv_writer.writerow(["Average Ranking"] + list(list(zip(*sorted_avg_rankings_ls))[1]))

	# 	csv_writer.writerow(["\n"])
	# 	csv_writer.writerow([""] + [c for c in classes])  
	# 	for i in range(len(conf_mat)):	
	# 		csv_writer.writerow([classes[i]] + conf_mat[i].tolist())

	total_conf_mat = np.add(total_conf_mat, results["conf_mat"])
		

	for i, metric in enumerate(["precision", "recall", "f1-score"]):
		for j, c in enumerate(classes):
			report_stack[fold][i][j] = results["report"][c][metric]

	# auc_stack.append(auc_score)
	macro_f1_stack.append(results["macro_f1"])

	return total_conf_mat, report_stack, macro_f1_stack 

def main():   
	parser = argparse.ArgumentParser()
	parser.add_argument("-k", "--n_folds", type=int, help="number of folds in k-fold cross validation", default=5)
	parser.add_argument("-tk", "--turn_off_kfold", action='store_true', help="turn off kfold", default=False)
	parser.add_argument("-l", "--learning_rate", type=float, help="learning rate", default=0.001)
	parser.add_argument("-e", "--n_epochs", type=int, help="number of epochs (divisible by 4)", default=100)
	parser.add_argument("-nl", "--n_layers", type=int, help="number of layers in LSTM", default=1)
	parser.add_argument('-csv','--csv', type=str, help='csv file name output')
	parser.add_argument("-d", "--description", type=str, help="description of experiment", required=True)
	parser.add_argument("-t", "--target", type=str, help="subcategory or category", required=True)
	parser.add_argument("-m", "--model", type=str, help="model (LSTM, SVM, RFC, XGBC)", default="LSTM")
	parser.add_argument("-r", "--reduce", type=int, help="dimension reduction amount", default=None)
	parser.add_argument('-in', "--indices", nargs="+", type=int, default=None)

	parser.add_argument("-x", "--x_data", type=str, help="x data file")	
	parser.add_argument("-y", "--y_data", type=str, help="y data file")	
	parser.add_argument("-ma", "--mapping", type=str, help="mapped labels file")	
	parser.add_argument("-do", "--drop_other", action='store_true', help="drop other class", default=False)
	# parser.add_argument("-oo", "--only_other", action='store_true', help="test other class", default=False)

	parser.add_argument("-eq", "--equal", action='store_true', help="equal distribution")
	# parser.add_argument("-ch", "--chunked", action='store_true', help="chunked data")
	parser.add_argument("-ap", "--append", action='store_true', help="append category or subcategory", default=False)
	
	parser.add_argument("-tw", "--twopart", action='store_true', help="if two part model is used", default=False)
	# parser.add_argument("-tw-type", "--twotype", type=str, help="two part step (category, subcategory)", default=None)


	args = parser.parse_args()


	arguments = {"equal": args.equal, "x_data": args.x_data, "y_data": args.y_data,
				 "mapping": args.mapping, "target": args.target, "drop_other": args.drop_other,
				 "indices": args.indices}

	X, y, classes, mapping = preprocess_data(arguments)
	print(X.shape)
	# X = X[:1000, :, :]
	# y = y[:1000]

	print("------------------------------------ Parameters ------------------------------------")
	description = f"Description: {args.description}, {args.model}"
	print(description)

	if args.csv:
		file_name = args.csv
		with open(file_name, 'a', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow(["\n"])            
			csv_writer.writerow([description])


	if not args.turn_off_kfold:
		fold = 0
		report_stack = np.zeros((args.n_folds, 3, len(classes)))
		macro_f1_stack = []
		total_conf_mat = np.zeros((len(classes), len(classes)))
		predictions_mapping_pred = {}
		predictions_mapping_true = {}

		kf = StratifiedKFold(n_splits=args.n_folds)
		splits = kf.split(X, y)
		num_classes = len(Counter(y.tolist()))

		for train_index, test_index in splits:
			print("------------------------------------ " + "Fold: " + str(fold) + " ------------------------------------")


			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			# Appending cats to subcats or vice-versa
			if args.append:
				data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}	
				X_train, X_test = test_append(args.target, mapping, data, train_index, test_index)

			# Two part model
			if args.twopart:	
				data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}						
				results = test_two_part(arg.target, args.model, data, train_index, test_index)

			# Standard run
			else:
				params = {"classes": classes, "lr": args.learning_rate, "num_epochs":args.n_epochs, "num_layers":args.n_layers}
				results = test_standard(args.model, params, X_train, X_test, y_train, y_test, train_index, test_index)

			if args.csv:
				file_name = args.csv
				total_conf_mat, report_stack, macro_f1_stack = write_per_fold(fold, file_name, results, classes,
																			  total_conf_mat, report_stack, macro_f1_stack)

			fold += 1


		if args.csv:
			stacks = {"report_stack": report_stack, "macro_f1_stack": macro_f1_stack, "conf_mat_stack": total_conf_mat}
			write_averages(file_name, results, stacks, args.n_folds, classes)


	# else:
	# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
	# 	data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}	
	# 	counts, macro_f1, counts, report, conf_mat, feat_import, y_pred = apply_concat_models(classes, data, model_name=args.model,
	# 																						  reshape=reshape, dim_reduce=args.reduce)










##### Trying two part models from other papers

def get_data():
	with open("../experiment_original/X_data.txt", "rb") as f:
		X = pickle.load(f)
	with open("../experiment_original/y_data.txt", "rb") as f:
		y_subcat = pickle.load(f)
	with open("../experiment_original/mapping.txt", "rb") as f:
		mapping = pickle.load(f)


	# Generate cat labels
	y_cat = fmp.to_category([mapping[i] for i in y_subcat])
	cat_subcat = [(y_c, y_s) for y_c, y_s in zip(y_cat, y_subcat)]
	reversed_mapping = {value:key for key, value in mapped_label.items()}
	cat_classes = [reversed_mapping[i] for i in list(Counter(y_cat).keys())]

	# Generate subcat labels
	classes = [mapped_subcat[mapping[i]] for i in range(len(mapping))]
	_, y_subcat, subcat_classes, _ = create_other_class(X, y_subcat, mapping, classes)

	# with open("../experiment_original/new_mapped_y.txt", "rb") as f:
	# 	mapped_y = pickle.load(f)

	# extended_y = []
	# for key, value in mapped_y.items():
	# 	extended_y += value
	# print(Counter([mapped_subcat[y] for y in extended_y]))
	# print(Counter([mapped_cat[mapped_subcat[y]] for y in extended_y]))
	

	print(mapping)
	print(Counter([mapped_subcat[mapping[y]] for y in y_subcat]))

	return X, y_cat, y_subcat, cat_classes, subcat_classes


def test_paper_2_approach():
	X, y_cat, y_subcat, cat_classes, subcat_classes = get_data()
	# kf = StratifiedKFold(n_splits=5)
	# splits = kf.split(X, y_cat)
	
	# params = {"lr": 0.001, "num_epochs":10, "num_layers":1}

	# for train_index, test_index in splits:
	# 	X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
	# 	X_train, X_test = fmp.standardize_data(X_train, X_test)
	# 	input_size = X_train.shape[2]

	# 	y_train_cat, y_test_cat = np.array(y_cat)[train_index], np.array(y_cat)[test_index]
	# 	data = {"X_train": X_train, "X_test": X_test, "y_train": y_train_cat, "y_test": y_test_cat}	
	# 	train_loader, test_loader = get_dataloaders(data)

	# 	model = get_model_lstm(input_size, num_classes=len(cat_classes), num_layers=params["num_layers"])

	# 	macro_f1, counts, model, report, conf_mat, y_pred = apply_NNs(cat_classes, model, train_loader, test_loader, params, verbose=False)

	# 	model.LSTM.weight_ih_l0.requires_grad = False
	# 	model.LSTM.bias_ih_l0.requires_grad = False

	# 	model.fc = nn.Linear(128, len(subcat_classes))

	# 	y_train_subcat, y_test_subcat = np.array(y_subcat)[train_index], np.array(y_subcat)[test_index]
	# 	data = {"X_train": X_train, "X_test": X_test, "y_train": y_train_subcat, "y_test": y_test_subcat}	
	# 	train_loader, test_loader = get_dataloaders(data)
	# 	macro_f1, counts, model, report, conf_mat, y_pred  = apply_NNs(subcat_classes, model, train_loader, test_loader, params)




	# # X_train, X_test, y_train_cat, y_test_cat, y_train_subcat, y_test_subcat, cat_classes, subcat_classes = prep_data()

	# # ##### WITHOUT TECHNIQUE #####
	# # data = {"X_train": X_train, "X_test": X_test, "y_train": y_train_subcat, "y_test": y_test_subcat}	
	# # train_loader, test_loader = get_dataloaders(data)
	# # model = get_model_lstm(input_size, num_classes=len(subcat_classes), num_layers=params["num_layers"])
	# # _, auc_score, macro_f1, counts, model, report, conf_mat = apply_NNs(subcat_classes, model, train_loader, test_loader, params)


	# ##### WITH TECHNIQUE #####
	# # 1.1 Obtain coarse labels
	# data = {"X_train": X_train, "X_test": X_test, "y_train": y_train_cat, "y_test": y_test_cat}	
	# train_loader, test_loader = get_dataloaders(data)

	# # 1.2 Obtain coarse model 
	# model = get_model_lstm(input_size, num_classes=len(cat_classes), num_layers=params["num_layers"])

	# # 1.3 Train with coarse labels
	# _, auc_score, macro_f1, counts, model, report, conf_mat = apply_NNs(cat_classes, model, train_loader, test_loader, params)

	# # 2.1 Modify trained model
	# # for i, (name, param) in enumerate(model.LSTM.named_parameters()):
	# # 	if name == "weight_hh_l0":
	# # 		print(param)

	# model.LSTM.weight_ih_l0.requires_grad = False
	# model.LSTM.bias_ih_l0.requires_grad = False

	# # model.LSTM.weight_hh_l0.requires_grad = False
	# # model.LSTM.bias_hh_l0.requires_grad = False

	# model.fc = nn.Linear(128, len(subcat_classes))

	# # 2.2 Update data
	# data = {"X_train": X_train, "X_test": X_test, "y_train": y_train_subcat, "y_test": y_test_subcat}	
	# train_loader, test_loader = get_dataloaders(data)

	# # 3.1 Train with fine labels
	# _, auc_score, macro_f1, counts, model, report, conf_mat = apply_NNs(subcat_classes, model, train_loader, test_loader, params)


if __name__ == "__main__":
	main()
	# test_paper_2_approach()