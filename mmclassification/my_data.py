import os

train_path = "/root/Code/mmclassification-master/data/my_dataset/train"
val_path = "/root/Code/mmclassification-master/data/my_dataset/val"

train_1_list = os.listdir(train_path + "/class1")
# print(train_1_list)
train_2_list = os.listdir(train_path + "/class2")

val_1_list = os.listdir(val_path + "/class1")
val_2_list = os.listdir(val_path + "/class2")

f = open("/root/Code/mmclassification-master/data/my_dataset/meta/train.txt", "w")
for i in range(len(train_1_list)):
    tmp = "class1/" + train_1_list[i]
    f.write(tmp)
    f.write(" ")
    f.write("0")
    f.write("\n")
    print("success" + tmp)

for j in range(len(train_2_list)):
    tmp = "class2/" + train_2_list[j]
    f.write(tmp)
    f.write(" ")
    f.write("1")
    f.write("\n")
    print("success" + tmp)

f.close()

f = open("/root/Code/mmclassification-master/data/my_dataset/meta/val.txt", "w")
for i in range(len(val_1_list)):
     tmp = "class1/" + val_1_list[i]
     f.write(tmp)
     f.write(" ")
     f.write("0")
     f.write("\n")
     print("success" + tmp)

for j in range(len(val_2_list)):
     tmp = "class2/" + val_2_list[j]
     f.write(tmp)
     f.write(" ")
     f.write("1")
     f.write("\n")
     print("success" + tmp)

f.close()
