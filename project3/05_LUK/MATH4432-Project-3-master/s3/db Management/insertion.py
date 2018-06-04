import json
from pymongo import MongoClient

filename = 'raw_plus_preprocessed_multiple_3.txt'
client = MongoClient("mongodb://localhost:27017")
db = client['fyp']

def insertJson(): #this is actually insertion of smoothed value
	with open(filename, 'r') as source:
		data = source.readlines()
		count = 0
		for line in data:
			entry = json.loads(line)
			db.googleChart.insert(entry)
			print("inserting entry : {:d}".format(count))
			count += 1


insertJson()
client.close()
