# grep "json_stats:" e2e-training-logs.txt > json_stats.txt
import matplotlib.pyplot as plt
import os

stats_dir = '/mnt/fcav/self_training/object_detection/upperbound1'
output_dir = '/mnt/fcav/self_training/object_detection/upperbound1/loss_plot'

stats_file = os.path.join(stats_dir, 'json_stats.txt')
file = open(stats_file, 'r')

iter = []
loss = []
lr = []

for line in file:
    line = line.split()
    num = len(line)
    i = 0
    while i < num:
        if line[i].find('iter') != -1:
            i += 1
            iter.append(int(line[i].strip(',')))
            continue

        if line[i].find('"loss"') != -1:
            i += 1
            loss.append(float(line[i].strip(',')))
            continue

        if line[i].find('"lr"') != -1:
            i += 1
            lr.append(float(line[i].strip(',')))
            continue

        i += 1

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fig = plt.figure()
plt.plot(iter, loss)
plt.title('Loss vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
name = 'Loss_iter{}_base_lr{}.jpg'.format(iter[len(iter) - 1], lr[0])
output_file = os.path.join(output_dir, name)
fig.savefig(output_file)

fig = plt.figure()
plt.plot(iter, lr)
plt.title('Learning rate vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Learning rate')
plt.show()
name = 'Lr_iter{}_base_lr{}.jpg'.format(iter[len(iter) - 1], lr[0])
output_file = os.path.join(output_dir, name)
fig.savefig(output_file)


