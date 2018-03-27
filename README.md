# RCNN 
Rich feature hierarchies for accurate object detection and semantic segmentation   

# 工程内容
这个程序是基于tensorflow实现RCNN功能。  

# 开发环境  
windows10 + python3.5 + tensorflow1.2 + tflearn + cv2 + scikit-learn   
i5-7500 + GTX 1070Ti   

# 数据集
采用17flowers据集, 官网下载：http://www.robots.ox.ac.uk/~vgg/data/flowers/17/  

# 程序说明   
1、config.py---网络定义、训练与数据处理所需要用到的参数      
2、Networks.py---用于定义Alexnet_Net模型、fineturn模型、SVM模型、边框回归模型   
4、process_data.py---用于对训练数据集与微调数据集进行处理（选择性搜索、数据存取等）    
5、train_and_test.py---用于各类模型的训练与测试、主函数     
6、selectivesearch.py---选择性搜索源码       


# 文件说明   
1、train_list.txt---预训练数据，数据在17flowers文件夹中         
2、fine_tune_list.txt---微调数据2flowers文件夹中       
3、直接用选择性搜索的区域划分　　　　通过RCNN后的区域划分　　通过SVM与边框回归之后的最终结果                     
　　![selectivesearch_1](https://github.com/liuyicheng007/R-CNN/raw/master/result/1.PNG)　　　![RCNN_1](https://github.com/liuyicheng007/R-CNN/raw/master/result/2.PNG)　　　![RCNN_2](https://github.com/liuyicheng007/R-CNN/raw/master/result/3.PNG)                        


# 程序问题   
1.参考代码给出了R-CNN十分清晰的主体流程，但缺失了最后的边框回归步骤。在实现的部分中使用了tflearn来实现       
　网络的定义与使用，好处是使用十分简单，使得程序易懂，代价是效率与灵活性。    
2.在保留参考代码的主体思想与一些小轮子的基础上，本代码实现了各个网络定义与使用的每一个具体步骤   
　与集成数据处理各个步骤。本次学习的目的:学习RCNN的实现过程，熟悉用tensorflow来构建网络与训练   
3.微调时的数据集存在一定的问题：一张图片经过筛选后一般保留100-200个候选框，当这些候选框与ground_truth的    
　IOU大于0.3时，我们将其认为是目标，小于0.3时我们认为是背景。实际情况是，目标候选框与背景候选框的比例过于   
　悬殊，可能达到几十比一的比例，这就导致了在微调训练时效果并不是特别好。
　解决思路：将目标候选框与背景候选框在数据处理时就分开保存，之后训练时，按1：2的比例调数据进行训练。待续   
4.在训练边框回归网络时，采用ReLU激活函数导致训练失败，许多权重都被变为0, 再一次证明了Relu的脆弱性。因此    
　采用了tanh激活函数，问题得到解决。上一次发生这种情况采用Leaky ReLU激活函数（可以解决ReLU易坏死问题）。    
5.SVM训练过程，没有采用论文所描述的hard negative mining method与NMS，简单化了。

# 论文细节补充：
1.finturn过程：
  计算每个region proposal与人工标注的框的IoU，IoU重叠阈值设为0.5，大于这个阈值的作为正样本，其他作     
　为负样本。然后在训练的每一次迭代中都使用32个正样本（包括所有类别）和96个背景样本组成的128张图片的batch    
　进行训练（正样本图片太少了）      
2.SVM训练过程：     
　对每个类都训练一个线性的SVM分类器，训练SVM需要正负样本文件，这里的正样本就是ground-truth框中的图像作
　为正样本，完全不包含的region proposal应该是负样本，但是对于部分包含某一类物体的region proposal该如  
　何训练作者同样是使用IoU阈值的方法，这次的阈值为0.3，计算每一个region proposal与标准框的IoU，小于0.3   
　的作为负样本，其他的全都丢弃。由于训练样本比较大，作者使用了standard hard negative mining method   
　（具体怎么弄的不清楚）来训练分类器。作者在补充材料中讨论了为什么fine-tuning和训练SVM时所用的正负样本   
　标准不一样，以及为什么不直接用卷积神经网络的输出来分类而要单独训练SVM来分类，作者提到，刚开始时只是用了   
　ImageNet预训练了CNN，并用提取的特征训练了SVMs，此时用正负样本标记方法就是前面所述的0.3,后来刚开始使用   
　fine-tuning时，使用了这个方法但是发现结果很差，于是通过调试选择了0.5这个方法，作者认为这样可以加大样本   
　的数量，从而避免过拟合。然而，IoU大于0.5就作为正样本会导致网络定位准确度的下降，故使用了SVM来做检测，全    
　部使用ground-truth样本作为正样本，且使用非正样本的，且IoU小于0.3的“hard negatives”，提高了定位的准确度。           
 3.hard negatives:    
　在训练过程中会出现 正样本的数量远远小于负样本，这样训练出来的分类器的效果总是有限的，会出现许多false positive。
　采取办法可以是，先将正样本与一部分的负样本投入模型进行训练，然后将训练出来的模型去预测剩下未加入训练过程的负样本，
　当负样本被预测为正样本时，则它就为false positive，就把它加入训练的负样本集，进行下一次训练，知道模型的预测精度不再提升
　这就好比错题集，做错了一道题，把它加入错题集进行学习，学会了这道题，成绩就能得到稍微提升，把自己的错题集都学过去，成绩就达到了相对最优
         

# 参考   
1、论文参考：        
   https://www.computer.org/csdl/proceedings/cvpr/2014/5118/00/5118a580-abs.html          
2、代码参考：     
   https://github.com/yangxue0827/RCNN     
   https://github.com/edwardbi/DeepLearningModels/tree/master/RCNN          
3、博客参考：       
   http://blog.csdn.net/u011534057/article/details/51218218        
   http://blog.csdn.net/u011534057/article/details/51218250        
  
