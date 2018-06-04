import csv
import json


csvfile = open('pca.csv', 'r')
jsonfile = open('pcaResult', 'w')

fieldnames = ("id", "cluster_id", "ts_start", "t1", "t2", "predict", "kr", "dt", "kn", "lp", "rf", "gp", "ls")
reader = csv.DictReader(csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')