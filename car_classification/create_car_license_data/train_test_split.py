import numpy as np
import os
import glob
import random
TRAIN_FILE = 'train_summary.txt'
TEST_FILE = 'test_summary.txt'
with open('data/summary.txt' , 'r') as rf:
	result = []
	for line in rf :
		result.append(line)
	random.Random(12345).shuffle(result)

len_data = len(result)
print(len_data)
train_len = int(0.9 * len_data)
tran_data = result[0:train_len]
test_data = result[train_len:-1]

with open(TRAIN_FILE, 'w') as wf:
	for line in tran_data:
		wf.write('{}\n'.format(line.strip()))

with open(TEST_FILE, 'w') as wf:
	for line in test_data:
		wf.write('{}\n'.format(line.strip()))