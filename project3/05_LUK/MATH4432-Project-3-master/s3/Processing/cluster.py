import json 
uniqueCluster = [] 
uniqueText = []
links = []
file1 = 'kmeans.txt'
file2 = 'pca.txt'

nodes = []



def getUnique(filename):
	with open(filename, 'r') as f: 
		data = f.readlines()
		#count = 0
		for line in data:
			entry = json.loads(line)
			cluster = entry['cluster_words']
			#count +=1
			if cluster not in uniqueCluster:
				uniqueCluster.append(cluster)
		"""print('all cluster:', count)
								print('unique cluster:', len(uniqueCluster))
								print(uniqueCluster)
		"""
		
		for i in range(len(uniqueCluster)):
			for word in uniqueCluster[i]:
				if word not in uniqueText:
					uniqueText.append("{:2d}".format(10 + i) + word)
		#print('all words:', count)
		#print('unique word:', len(uniqueText))
		#print(uniqueText)

		"""
		for i in range(len(uniqueCluster)):
			for word in uniqueCluster[i]:
				if word not in uniqueText:
					uniqueText.append(word)
				else: print(word)

		print((uniqueText))"""

def generateLink(source, target):
	links.append({'source': source, 'target': target})

def createJson():
	for word in uniqueText:
		nodes.append({"name":word[2:], "group":word[:2]})

	print(nodes)
	#with open('output.txt', 'w') as o:
		#o.write(nodes)


getUnique(file1)
createJson()