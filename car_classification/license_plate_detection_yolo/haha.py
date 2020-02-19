


SUMMARY = './data/summary.txt'

result = []
with open(SUMMARY, 'r') as rf:
	for l in rf:
		l = l.strip().split()
		file = l[0]
		box = [int(x) for x in l[1:]]
		box[2] = box[2] - box[0]
		box[3] = box[3] - box[1]
		result.append([file, box])
with open('./data/train_summary.txt', 'w') as wf:
	for f, b in result:
		b = [str(x) for x in b]
		wf.write(f + ' ' + ' '.join(b) + '\n')