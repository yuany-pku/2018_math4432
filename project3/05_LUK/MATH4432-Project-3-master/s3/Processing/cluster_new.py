import json 
uniqueCluster = [] 
uniqueText = []
links = []
file1 = 'kmeans.txt'
file2 = 'pca.txt'
file3 = 'x.txt'
file4 = 'y.txt'
file5 = 'z.txt'


nodes = []
output = {}

linkTable = []

def getUnique(filename):
	with open(filename, 'r') as f: 
		data = f.readlines()
		#count = 0
		for cluster in data:
			cluster = json.loads(cluster)
			#print(cluster)
			#print(cluster.split(','))
			#count +=1
			if cluster not in uniqueCluster:
				uniqueCluster.append(cluster)

def toNodes():
	for cluster in uniqueCluster:
		nodes.append({"name":cluster, "group":uniqueCluster.index(cluster)})

		#print(uniqueCluster)

def initializeLinkTable(): 
	for i in range(len(uniqueCluster)):
		for j in range(i + 1, len(uniqueCluster)):
			linkTable.append([i,j,0]) #source, target, value
	print(linkTable)

def generateLink():
	print()

#helper function 
def ToLink(source, target):
	links.append({'source': source, 'target': target})

def generateLinks():
	#for cluster in uniqueCluster:
		#linkTabl
	print()


def createJson():
	for i in range(len(uniqueCluster)):
		nodes.append({"name":uniqueCluster[i], "group":uniqueCluster.index(uniqueCluster[i])})
		for word in uniqueCluster[i]:
			for j in range(i + 1, len(uniqueCluster)):
				if word in uniqueCluster[j]:
					ToLink(i, j)
	print(nodes)
	#print(links)
	#with open('output.txt', 'w') as o:
		#o.write(nodes)

def makeJson(name): 
	output["nodes"] = nodes
	output["links"] = links
	#print(output)
	with open(name, 'w') as o:
		o.write(json.dumps(output))



getUnique(file5)
toNodes()
createJson()
print(len(uniqueCluster))
#initializeLinkTable()
