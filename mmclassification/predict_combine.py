#coding: utf - 8
s1 = set(open(r"/root/Code/mmclassification-master/data/index1.txt").readlines())
s2 = set(open(r"/root/Code/mmclassification-master/data/index2.txt").readlines())

f = open("/root/Code/mmclassification-master/data/index3.txt", "w")

# all_union = list(set(s1).union(set(s2)))  # 并集
# # all_intersection = list(set(s1).intersection(set(s2)))  #交集
#
# for i in range(len(all_union)):
#     f.write(all_union[i])
# f.close()


all_difference = list(set(s1).difference(set(s2)))  #差集
for i in range(len(all_difference)):
    f.write(all_difference[i])
f.close()
# 正样本取并集，负样本取并集。然后二者再取差集。负样本数据多，可能包含正样本，所以负样本-正样本数据。