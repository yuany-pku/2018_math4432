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
			#if cluster not in uniqueCluster:
			uniqueCluster.append(cluster)


def generateLink(source, target):
	links.append({'source': source, 'target': target})

def createJson():
	for i in range(len(uniqueCluster)):
		nodes.append({"name":uniqueCluster[i], "group":uniqueCluster.index(uniqueCluster[i])})
		for word in uniqueCluster[i]:
			for j in range(i + 1, len(uniqueCluster)):
				if word in uniqueCluster[j]:
					generateLink(i, j)
	print(nodes)
	print(links)
	#with open('output.txt', 'w') as o:
		#o.write(nodes)


getUnique(file1)
createJson()