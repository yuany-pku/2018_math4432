import json 

with open('clusterfyp_withTsne_562clusters_4words.txt') as f, open('z.txt', 'a') as o:
	for l in f:
		o.write( json.dumps([ json.loads(l)[1][0].strip(), json.loads(l)[1][1].strip(), json.loads(l)[1][2].strip() ]) +'\n')