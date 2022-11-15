import os
import csv
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from prettytable import PrettyTable

# 定义混淆矩阵类，方便用于计算各种损失
class ConfusionMatrix(object):
    def __init__(self,num_classes:int,labels:list):
        # self.matrix中存储混淆矩阵取值
        self.matrix = np.zeros((num_classes,num_classes))
        self.num_classes = num_classes
        self.labels = labels
       
    
    # 注意一下：这里传入的都是numpy数据格式
    def update(self,preds,labels):
        for p,t in zip(preds,labels):
            self.matrix[p,t] += 1
    
    def summary(self,log_dir='',is_print=False):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i,i]
        acc = sum_TP / np.sum(self.matrix)
        
        precisions = []
        recalls = []
        specificitys = []
        f1s = []

        tabel = PrettyTable()
        tabel.field_names = ['','Precision','Recall','Specificity','F1-score']
        # 这个表存储的类别计算得到的平均值
        mtabel = PrettyTable()
        mtabel.field_names = ['','Accuracy','mPrecision','mRecall','mSpecificity','mF1-score']
        for i in range(self.num_classes):
            TP = self.matrix[i,i]
            FP = np.sum(self.matrix[i,:]) - TP
            FN = np.sum(self.matrix[:,i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            # round(num,dig) 四舍五入到指定的小数位数 dig
            precision = round(TP / (TP + FP),4) if TP + FP != 0 else 0.
            recall = round(TP / (TP + FN), 4) if TP + FN != 0 else 0.
            specificity = round(TN / (TN + FP), 4) if TN + FP !=0 else 0.
            f1 = round( 2 * (precision * recall) / ( precision + recall) ,4) if precision + recall !=0 else 0.

            precisions.append(precision)
            recalls.append(recall)
            specificitys.append(specificity)
            f1s.append(f1)

            tabel.add_row([self.labels[i],precision,recall,specificity,f1])
        

        mprecision = np.mean(precisions)
        mrecall = np.mean(recalls)
        mspecificity = np.mean(specificitys)
        mf1 = np.mean(f1s)
        mtabel.add_row(['',acc,mprecision,mrecall,mspecificity,mf1])

        if is_print:
            print(tabel)
            print(mtabel)
            with open(os.path.join(log_dir,"metric.txt"),'a+') as f:
                f.write(tabel.get_string())
                f.write('\n')
                f.write(mtabel.get_string())
            with open(os.path.join(log_dir,"metrics.csv"),'w') as f:
                epoch_outputs = [['Test Acc', 
            'Precision(0)', 'Precision(1)', 'Recall(0)', 'Recall(1)',
             'Specificity(0)', 'Specificity(1)','F1 Score(0)','F1 Score(1)']]
                epoch_outputs.append([acc,precisions[0],precisions[1],recalls[0],recalls[1],
                                        specificitys[0],specificitys[1],f1s[0],f1s[1]])
                csvwriter = csv.writer(f)
                csvwriter.writerows(epoch_outputs)
            


        return precisions,recalls,specificitys,f1s,acc,mprecision,mrecall,mspecificity,mf1

    def plot(self,log_dir):
        matrix = self.matrix
        plt.imshow(matrix,cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), ["Be","Ma"], rotation=45)
        plt.xlabel('True Labels')
        plt.yticks(range(self.num_classes), ["Be","Ma"])
        plt.ylabel('Predict Labels')
        plt.title('Confusion matrix')

        # 显示colorbar
        plt.colorbar()

        # 在图中标注数量信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里是[y,x]而非[x,y]
                info = int(matrix[y,x])
                plt.text(x,y,info,
                        verticalalignment='center',
                        horizontalalignment='center',
                        color="white" if info > thresh else "black")
        
        plt.tight_layout() # 自动调节子图区域，使之充满整个区域
        plt.savefig(os.path.join(log_dir,"ConfusionMatrix.png"))




