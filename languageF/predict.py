from tkinter import *
import numpy as np
import jieba as jb
import joblib
from gensim.models.word2vec import Word2Vec

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )


class core():
    def __init__(self,str):
        self.string=str

    def build_vector(self,text,size,wv):
        #创建一个指定大小的数据空间
        vec = np.zeros(size).reshape((1,size))
        #count是统计有多少词向量
        count = 0
        #循环所有的词向量进行求和
        for w in text:
            try:
                vec +=  wv[w].reshape((1,size))
                count +=1
            except:
                continue
        #循环完成后求均值
        if count!=0:
            vec/=count
        return vec
    def get_predict_vecs(self,words):
        # 加载模型
        twitter_w2v = Word2Vec.load("data/model3.pkl")
        #将新的词转换为向量
        train_vecs = self.build_vector(words,300,twitter_w2v.wv)
        return train_vecs
    def svm_predict(self,string):
        # 对语句进行分词
        words = jb.cut(string)
        # 将分词结果转换为词向量
        word_vecs = self.get_predict_vecs(words)
        #加载模型
        cls = joblib.load("data/svcmodel.pkl")
        #预测得到结果
        result = cls.predict(word_vecs)
        #输出结果
        if result[0]==1:
            return "好感"
        else:
            return "反感"
    def main(self):
        s=self.svm_predict(self.string)
        return s

root=Tk()
root.title("情感分析")
sw = root.winfo_screenwidth()
#得到屏幕宽度
sh = root.winfo_screenheight()
#得到屏幕高度
ww = 500
wh = 300
x = (sw-ww) / 2
y = (sh-wh) / 2-50
root.geometry("%dx%d+%d+%d" %(ww,wh,x,y))
# root.iconbitmap('tb.ico')

lb2=Label(root,text="输入内容，按回车键分析")
lb2.place(relx=0, rely=0.05)

txt = Text(root,font=("宋体",20))
txt.place(rely=0.7, relheight=0.3,relwidth=1)

inp1 = Text(root, height=15, width=65,font=("宋体",18))
inp1.place(relx=0, rely=0.2, relwidth=1, relheight=0.4)

def run1():
    txt.delete("0.0",END)
    a = inp1.get('0.0',(END))
    p=core(a)
    s=p.main()
    print(s)
    txt.insert(END, s)   # 追加显示运算结果

def button1(event):
    btn1 = Button(root, text='分析', font=("",12),command=run1) #鼠标响应
    btn1.place(relx=0.35, rely=0.6, relwidth=0.15, relheight=0.1)
    # inp1.bind("<Return>",run2) #键盘响应

button1(1)
root.mainloop()



