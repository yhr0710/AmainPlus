import numpy as np
# import csv
#
# reader = csv.reader(open("GCJ_clone.csv", 'r'))
# print(reader)
#
# for r in reader:
#     print(r)
#     # 对于CSV文件每一行,提取第一列和第二列的内容,分别存储在f1和f2中
#     # 使用.split('.java')[0]从f1和f2中去掉后缀.java,得到文件名
#     f1 = r[0].split('.java')[0]
#     f2 = r[1].split('.java')[0]
#     print(f1)
#     print(f2)

weightorder = []
with open('weight.txt', 'r') as f:
    lines = f.readlines()[:456]
    for line in lines:
        value = line.split()[1]
        weightorder.append(int(value))
print(weightorder)
print(len(weightorder))
