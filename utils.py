import json


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

json_file = "../../../../../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"

def micros_to_mins(micros):
	return micros/6e7

def mins_to_micros(micros):
	return micros/6e7

def micros_to_secs(micros):
	return micros/1e6

def secs_to_micros(secs):
	return secs*1e6


def get_dur_in_micros(data):
    return data["timestamp"].max() - data["timestamp"].min()


def get_indexable_meta():
	with open(json_file, 'r') as inputFile:
		meta_json = json.load(inputFile)

	indexable_meta = {}

	for m in meta_json:
		temp_id = m["id"]
		m.pop("id")
		indexable_meta[temp_id] = m
		
	return indexable_meta



	