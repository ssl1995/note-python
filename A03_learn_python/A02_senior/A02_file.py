'''
文件的写入
'''
import random

epoch = 100  # 遍历的次数
accuracy = 0.5
loss = 0.9

with open('training_log.txt', 'w') as file:
    # 文件头
    file.write('Epoch\tAccuracy\tLoss\n')
    # 文件内容
    for epoch_i in range(1, epoch + 1):
        accuracy += random.uniform(0, 0.005)  # 取一个随机数
        loss -= random.uniform(0, 0.005)

        accuracy = min(1, accuracy)
        loss = max(0, loss)

        file.write(f'{epoch_i}\t{accuracy:.3f}\t{loss:.3f}\n')

        print(f'Epoch:{epoch_i}\t Accuracy:{accuracy:.3f}\t Loss:{loss:.3f}')
