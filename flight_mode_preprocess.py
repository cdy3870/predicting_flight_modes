import UAV as uav
import pandas as pd
import os
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
from collections import Counter
import pickle
import data_processing as dp
import utils


ulog_folder = "../../../../../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"
ulog_folder_hex = "../../../../../../../work/uav-ml/px4-Ulog-Parsers/dataDownloadedHex"

indexable_meta = utils.get_indexable_meta()

def get_filtered_ids(drone_type="Quadrotor"):
    '''
    Get only a specific drone_type

    Returns:
        filtered_ids (list) :  list of ulog ids
    '''

	ulogs_downloaded = os.listdir(ulog_folder)
	drone_ids = [u[:-4] for u in ulogs_downloaded]

	filtered_ids = [u for u in drone_ids if indexable_meta[u]["duration"] != "0:00:00"]
	filtered_ids = [u for u in drone_ids if indexable_meta[u]["type"] == drone_type]

	distribution = [indexable_meta[u]["type"] for u in filtered_ids]
			
	return filtered_ids

def split_data(df, feat_name):
    '''
    Splits the raw data according to each timestamp's assigned mode for a specific feature

    Parameters:
        df (pd.DataFrame) : dataframe of unsplit mode labeled data 
        feat_name (string) : the feature of interest
        
    Returns:
        log_X (list) : mode separated flight
        log_y (list) : corresponding mode labels
    '''

	current_index = 0
	prev_value = df["mode"].iloc[0]
	prev_index = 0
	log_X = []
	log_y = [prev_value]

	for i, row in df.iterrows():
		next_value = int(row["mode"])
		if next_value != prev_value:
			log_X.append((prev_value, df.iloc[prev_index:i].reset_index()[["timestamp", feat_name]]))
			# log_X.append((prev_value, feat_name))
			prev_value = next_value
			prev_index = i
			log_y.append(next_value)

	log_X.append((prev_value, df.iloc[prev_index:len(df)].reset_index()[["timestamp", feat_name]]))
	# log_X.append((prev_value, feat_name))

	return log_X, log_y

def extract_dfs(mUAV, table_name, feat_name):
    '''
	Extracts dataframes from raw px4 data and groups them according to feature 

    Parameters:
        mUAV (object) : ulog parsing and mode separator object
        table_name (string) : the ulog topic
        feat_name (string) : the corresponding feature
        
    Returns:
        df (pd.DataFrame) : parsed ulog data
    '''

	data = mUAV.get_desired_feats(table_name, feat_name)
	modes = mUAV.modes_nearest_indices(table_name)
	timestamps = mUAV.get_time_stamp(table_name)

	tmp_dict = []

	for i, mode in enumerate(modes):
		tmp_dict.append({"timestamp": timestamps[i], "mode": mode, feat_name: data[i]})

	df = pd.DataFrame.from_dict(tmp_dict)

	return df

def generate_data(feats):
	# Only ids that are quad
	ids = get_filtered_ids()
	ids = [u for u in ids if indexable_meta[u]["type"] == "Quadrotor"]

	X = []
	y = []
	mapped_X = {}
	mapped_y = {}
	success_count = 0

	path = ulog_folder

	for i, id in enumerate(ids):

		try:
			mUAV = uav.UAV(id, path)
		except:
			continue

		print(f"Log {i}/{len(ids)}")
		print(f"Success count: {success_count}")
		print(id)


		modes = []
		data = {}

		counter = 0
		for feat in feats:
			strings = feat.split(" ")
			table_name = strings[0]
			feat_name = strings[2]
			try:
				df = extract_dfs(mUAV, table_name, feat_name)
				counter += 1
			except:
				print(feat_name)
				break

		if counter == len(feats):
			for feat in feats:
				strings = feat.split(" ")
				table_name = strings[0]
				feat_name = strings[2]

				df = extract_dfs(mUAV, table_name, feat_name)
				log_X, log_y = split_data(df, feat_name)

				data[feat] = log_X
				modes.append(log_y)


			
			lens = [len(m) for m in modes]
			min_ind = np.argmin(np.array(lens))
			min_list = modes[min_ind]

			mapped_X[id] = [{} for i in min_list]

			for key, value in data.items():
				temp_index = 0
				# mapped_X[id][key] = {}

				for i, x in enumerate(value):		
					if x[0] == min_list[temp_index]:
						mapped_X[id][temp_index][key] = x[1]
						temp_index += 1	

					if temp_index == len(min_list):
						break		

			mapped_y[id] = min_list

			success_count += 1


	return mapped_X, mapped_y


def get_equal_distribution(X, y, min_instances=1000):
	counts = dict(Counter(y))
	# print(counts)
	new_X, new_y = {}, []
	modes = [key for key, value in counts.items() if value > min_instances - 1]
	counter = {m:0 for m in modes}
	keys = list(X.keys())

	i = 0

	while modes:
		current_val = y[i]
		
		if current_val in modes:
			counter[current_val] += 1
			new_X[keys[i]] = X[keys[i]]
			new_y.append(current_val)

			if counter[current_val] == min_instances:
				modes.remove(current_val)
				del counter[current_val]

		i += 1

	return new_X, new_y


def apply_dur_threshold(X, y, is_added):
	print(len(X))
	total_mins = []
	total_maxes = []
	keys = list(X.keys())

	for x in X.values():
		indiv_mins = []
		indiv_maxes = []
		for key, value in x.items():
			indiv_mins.append(value["timestamp"].min())
			indiv_maxes.append(value["timestamp"].max())
		total_mins.append(min(indiv_mins))
		total_maxes.append(max(indiv_maxes))


	durations = list(np.array(total_maxes) - np.array(total_mins))
	durations = list(map(lambda x : utils.micros_to_secs(x), durations))

	threshold_ind = [i for i, d in enumerate(durations) if d >= 10]
	threshold_keys = list(np.array(keys)[threshold_ind])

	# extra_indices = []
	# for i, ind in enumerate(threshold_ind):
	# 	if is_added[ind] == 1:
	# 		extra_indices.append(i)

	# with open("extra_indices.txt", "wb") as f:
	# 	pickle.dump(extra_indices, f)



	new_X = {}
	for key in threshold_keys:

		new_X[key] = X[key]


	new_y = list(np.array(y)[threshold_ind])

	print(len(new_X))
	return new_X, new_y

def to_category(y):
	new_y = [utils.mapped_label[utils.mapped_cat[utils.mapped_subcat[i]]] for i in y]

	return new_y


def remap_y(y):
	counts = dict(Counter(y))
	new_mapping = {}
	new_y = []
	for i, value in enumerate(counts.keys()):
		new_mapping[value] = i

	for i in y:
		new_y.append(new_mapping[i])

	new_mapping = {value:key for key,value in new_mapping.items()}

	return new_y, new_mapping


def extending_lists(mapped_X, mapped_y, feats, extra_ids):
	X = {}
	y = {}
	y_list = []
	is_added = []



	for key, value in mapped_X.items():
		counter = 0
		for i, sample in enumerate(value):
			if len(list(sample.keys())) == len(feats):
				X[f"{key} | {counter}"] = sample
				y[f"{key} | {counter}"] = mapped_y[key][i]
				y_list.append(mapped_y[key][i])

				if key in extra_ids:
					is_added.append(1)
				else:
					is_added.append(0)

				counter += 1


	return X, y, y_list, is_added
	

def preprocess_data(mapped_X, mapped_y, feats, equal_dist=False, chunking=False, num_t_ints=50, derived_feat=False):
# 	with open("extra_ids.txt", "rb") as f:
# 		extra_ids = pickle.load(f)

	X, y, y_list, is_added = extending_lists(mapped_X, mapped_y, feats, [])

	X, y_list = apply_dur_threshold(X, y_list, [])

	if equal_dist:
		X, y_list = get_equal_distribution(X, y_list)

	if chunking:
		X, ids_intervals, y_list = dp.get_x_min_chunks(X, y)


	if derived_feat:
		with open("experiment_derived/new_mapped_X_derived_full.txt", "rb") as f:
			mapped_X = pickle.load(f)
		keys = list(mapped_X.keys())
		X = mapped_X
		print(X[keys[-1]].keys())


	# X = {k: X[k] for k in list(X)[:100]}

	new_X = dp.timestamp_bin(X, num_t_ints=num_t_ints)

	new_y, new_mapping = remap_y(y_list)

	return new_X, new_y, new_mapping


def standardize_data(X_train, X_test):
	scaler = StandardScaler()

	print(X_train.shape)

	num_instances = X_train.shape[0]
	num_features = X_train.shape[1]
	num_instances_test = X_test.shape[0]
	num_features_test = X_test.shape[1]

	if len(X_train.shape) == 3:
		num_times = X_train.shape[2]
		num_times_test = X_test.shape[2]
		scaler = scaler.fit(X_train.reshape(num_instances * num_times, num_features))
		X_train = scaler.transform(X_train.reshape(num_instances * num_times, num_features))
		X_test = scaler.transform(X_test.reshape(num_instances_test * num_times_test, num_features_test))

		X_train = X_train.reshape(num_instances, num_features, num_times)
		X_test = X_test.reshape(num_instances_test, num_features_test, num_times_test)
	else:
		scaler = scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)


	return X_train, X_test


def main():
	print(indexable_meta.keys())

if __name__ == "__main__":
	main()
