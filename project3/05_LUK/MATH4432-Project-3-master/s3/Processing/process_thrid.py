import json 

def getInfo():
	with open('clusterfyp_withTsne_562clusters_4words.txt') as f, open('z.txt', 'w') as o :
		data = json.loads(f.readline())
		print(type(data))
		for i in range(len(data)): 
			o.write(json.dumps([data[i][0].strip(), data[i][1].strip(), data[i][2].strip() ]) +'\n')



getInfo()
