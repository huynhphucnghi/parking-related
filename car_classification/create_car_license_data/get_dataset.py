import pandas as pd
import numpy as np
import requests
import os

# data_csv_file = glob.glob('data/*csv')
data = 'data/export-2019-12-10T18_53_25.472Z.csv'
summary = 'data/summary.txt'
data_directory = 'data/'

# data summary co dang x1, y1, x2, y2
lines = []
with open(data, 'r') as df:
	line = df.readline()
	for line in df:
		line = line.strip().split(',')
		temp = ','.join([line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10]])
		
		temp = temp.split('[')[2]
		temp = temp.split(']')[0]
		temp = temp.replace('{', '')
		temp = temp.replace('}', '')
		# print(temp)
		temp = temp.replace('""x"":', '')
		temp = temp.replace('""y"":', '')
		temp = temp.split(',')
		temp = [int(x) for x in temp]
		x1, y1, x2, y2, x3, y3, x4, y4 = temp
		x = [x1, x2, x3, x4]
		y = [y1, y2, y3, y4]
		x = np.sort(x)
		y = np.sort(y)
		box = [x[0], y[0], x[3], y[3]]
		box = [str(x) for x in box]

		needed_data = [line[2], box, line[16]]
		try :
			with open(os.path.join(data_directory, needed_data[2]), 'wb') as f:
				f.write(requests.get(needed_data[0]).content)
			lines.append(needed_data)
		except:
			print('loi cmnr')
with open(summary, 'w') as wf:
	for line in lines:
		wf.write(os.path.join(data_directory, line[2]) + ' ' + ' '.join(line[1]) + '\n')