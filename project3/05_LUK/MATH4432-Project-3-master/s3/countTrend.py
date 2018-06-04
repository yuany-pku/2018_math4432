import json
trendingCount = {} 

filename = 'raw_plus_preprocessed_multiple_3.txt'

with open(filename, 'r') as f: 
	lines = f.readlines()
	for line in lines: 
		#print(line)
		entry =  json.loads(line)
		if (entry['isTrend'] ==1 ): 
			if(entry['ts_start'][:8] not in trendingCount):
				trendingCount[entry['ts_start'][:8]] = 1 
			else:
				trendingCount[entry['ts_start'][:8]] += 1 

print(trendingCount)
sum = 0 
count = 0 

for key, value in trendingCount.items():
	sum += value 
	#print("{:d} value processed".format(count))
	count = count +1 

print("average trending topic per day is {:f}".format(sum / len(trendingCount)) )

