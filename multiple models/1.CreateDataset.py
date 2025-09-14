'''
生成训练集和测试集，保存在txt文件中
'''
import os
import random

# 数据划分比例
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# 数据目录
rootdata = r"data2"

train_list, val_list, test_list = [], [], []  # 三个列表分别存储训练集、验证集和测试集数据
data_list = []

# 生产 train.txt, val.txt, 和 test.txt
class_flag = -1
for a, b, c in os.walk(rootdata):
    print(a)
    for i in range(len(c)):
        data_list.append(os.path.join(a, c[i]))

    # 按比例划分
    total_num = len(c)
    train_end = int(total_num * train_ratio)
    val_end = int(total_num * (train_ratio + val_ratio))

    # 训练集
    for i in range(0, train_end):
        train_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
        train_list.append(train_data)

    # 验证集
    for i in range(train_end, val_end):
        val_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
        val_list.append(val_data)

    # 测试集
    for i in range(val_end, total_num):
        test_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
        test_list.append(test_data)

    class_flag += 1

# 打乱数据
random.shuffle(train_list)
random.shuffle(val_list)
random.shuffle(test_list)

# 保存文件
with open('train.txt', 'w', encoding='UTF-8') as f:
    for train_img in train_list:
        f.write(str(train_img))

with open('val.txt', 'w', encoding='UTF-8') as f:
    for val_img in val_list:
        f.write(str(val_img))

with open('test.txt', 'w', encoding='UTF-8') as f:
    for test_img in test_list:
        f.write(str(test_img))

print(f"训练集数量: {len(train_list)}")
print(f"验证集数量: {len(val_list)}")
print(f"测试集数量: {len(test_list)}")


