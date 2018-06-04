from pymongo import MongoClient
import json 

client = MongoClient("mongodb://localhost:27017")
db = client['fyp']


f = db.googleChart.find({}, {'_id':0})

for line in f:
	print(line)
	tag = json.loads(json.dumps(line))
