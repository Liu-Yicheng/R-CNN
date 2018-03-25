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
3、直接用选择性搜索的区域划分　　　　通过RCNN后的区域划分　　　通过SVM与边框回归之后的最终结果                     
　　![selectivesearch_1](https://github.com/liuyicheng007/R-CNN/raw/master/result/1.PNG)　　　![RCNN_1](https://github.com/liuyicheng007/R-CNN/raw/master/result/２.PNG)　　　![RCNN_2](https://github.com/liuyicheng007/R-CNN/raw/master/result/3.PNG)                        
4、2.png---       
      
5、3.png---   
    

# 程序问题   
1.参考代码给出了R-CNN十分清晰的主体流程，但缺失了最后的边框回归步骤。在实现的部分中使用了tflearn来实现       
   网络的定义与使用，好处是使用十分简单，使得程序易懂，代价是效率与灵活性。    
2.在保留参考代码的主体思想与一些小轮子的基础上，本代码实现了各个网络定义与使用的每一个具体步骤   
   与集成数据处理各个步骤。本次学习的目的:学习RCNN的实现过程，熟悉用tensorflow来构建网络与训练   
3.微调时的数据集存在一定的问题：一张图片经过筛选后一般保留100-200个候选框，当这些候选框与ground_truth的    
   IOU大于0.3时，我们将其认为是目标，小于0.3时我们认为是背景。实际情况是，目标候选框与背景候选框的比例过于   
   悬殊，可能达到几十比一的比例，这就导致了在微调训练时效果并不是特别好。（原代码作者认为数据集小也是训练    
   结果不好的因素之一）   
   解决思路：将目标候选框与背景候选框在数据处理时就分开保存，之后训练时，按1：2的比例调数据进行训练。待续   
4.在训练边框回归网络时，采用ReLU激活函数导致训练失败，许多权重都被变为0, 再一次证明了Relu的脆弱性。因此    
   采用了tanh激活函数，问题得到解决。上一次发生这种情况采用Leaky ReLU激活函数（可以解决ReLU易坏死问题）。    
5.对选择的区域是直接进行缩放的，并没有像原论文中进行对图片的处理；      
6.由于数据集合论文采用不一样，但是微调和训练SVM时采用的IOU阈值一样，有待调参。    

# 参考   
1、论文参考：        
   https://www.computer.org/csdl/proceedings/cvpr/2014/5118/00/5118a580-abs.html          
2、代码参考：     
   https://github.com/yangxue0827/RCNN     
   https://github.com/edwardbi/DeepLearningModels/tree/master/RCNN          
3、博客参考：       
   http://blog.csdn.net/u011534057/article/details/51218218        
   http://blog.csdn.net/u011534057/article/details/51218250        
  
