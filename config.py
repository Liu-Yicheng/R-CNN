import os

Image_size = 224
Staircase=True
Summary_iter=50
Save_iter = 5000
Max_iter = 20000

Train_list = r'./train_list.txt'
Finetune_list = './fine_tune_list.txt'

DATA = './FlowerData'
Fineturn_save = './FlowerData/Fineturn'
SVM_and_Reg_save ='./FlowerData/SVM_and_Reg'

Out_put = './output'

T_class_num = 17
T_batch_size =64
T_decay_iter=100
T_learning_rate=0.0001
T_decay_rate=0.99
T_weights_file =r'./output/train_alexnet/save.ckpt-40000'

F_class_num = 3
F_batch_size = 256
F_decay_iter=1000
F_learning_rate=0.001
F_decay_rate=0.99
F_fineturn_threshold =0.3
F_svm_threshold =0.3
F_regression_threshold =0.6
F_weights_file =r'./output/fineturn/save.ckpt-54000'

R_class_num = 5
R_batch_size = 512
R_decay_iter=5000
R_learning_rate=0.0001
R_decay_rate=0.5
R_weights_file =r'./output/Reg_box/save.ckpt-10000'