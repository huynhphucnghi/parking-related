import matplotlib.pyplot as plt
import numpy as np 

LOG_FILE = 'check_point/log.txt'

data = []
with open(LOG_FILE, 'r') as rf:
	for line in rf:
		line = line.strip().split('\t')
		# line = [float(x) for x in line]
		data.append(line)
	data = data[1:]
	data = [[float(x) for x in line] for line in data]
np.array(data)
train_loss = [line[1] for line in data]
train_acc  = [line[2] for line in data]
test_loss = [line[3] for line in data]
test_acc  = [line[4] for line in data]

plt.plot(train_acc, label='Train')
plt.plot(test_acc, label='Test')
plt.ylabel('Accuracy')
# plt.xlabel('Step')
plt.ylim((0.7,1))
plt.legend()
plt.show()

plt.plot(train_loss, label='Train')
plt.plot(test_loss, label='Test')
plt.ylabel('Loss')
# plt.xlabel('Step')
plt.legend()
plt.show()