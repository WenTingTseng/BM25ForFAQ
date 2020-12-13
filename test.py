
import codecs
import similarity
import jieba
import json
import time
import sys
import torch
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from collections import Counter

def BM25_readfile():
    #讀取BM25的訓練資料(標準問句斷過詞)
    BM25_train_data=[]
    with open("Corpus/segmentData/CutTrain.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            BM25_train_data.append(line.replace("\r", "").replace("\n", ""))
  
    #讀取stop word
    stopWords=[]
    with open("OtherData/stopword.txt", 'r', encoding='UTF-8') as file:
        for data in file.readlines():
            data = data.strip()
            stopWords.append(data)
    return stopWords,BM25_train_data
    
def BM25():
    stopWords,BM25_train_data=BM25_readfile()  
    traindict=type_dict()

    fp = open("Corpus/segmentData/CutTest.txt", "r")
    query = fp.readlines()
    fp.close()

    fp = open("Corpus/segmentData/CutTrain.txt", "r")
    Question = fp.readlines()
    fp.close()

    AllRank=[]
    Result=[]
    for (idx,q_input) in enumerate(query):
        words = [w for w in q_input.split() if w not in stopWords]
        q_input="".join(words)
        ##################### BM25 ##############################
        seg_list_query = " ".join(jieba.cut(q_input, cut_all=False))
        dataset=dict()
        dataset["".join(seg_list_query)]=dict()
        #result=dict()
        for train in Question:
            sim=similarity.ssim(seg_list_query,train,model='bm25')  
            dataset["".join(seg_list_query)][train.replace('\n','')]=dataset["".join(seg_list_query)].get(train,0)+sim

        for r in dataset:
            top=sorted(dataset[r].items(),key=lambda x:x[1],reverse=True)
        statistics=[]
        
        for t in top[1:4]:
            statistics.append(traindict[t[0]])
        result = Counter(statistics)
       # Result.append(result.most_common(1)[0][0])
        if(result.most_common(1)[0][1]>3):
            Result.append(result.most_common(1)[0][0])
        else:
            print(traindict[top[0][0]])
            Result.append(traindict[top[0][0]])
    return Result

def type_dict():
    fp = open("Corpus/segmentData/CutTrain.txt", "r")
    data = fp.readlines()
    fp.close()

    fp = open("CorpusType/AllType.txt", "r")
    alltype = fp.readlines()
    fp.close()

    train_Dict=dict()
    for (d,t) in zip(data,alltype):
        train_Dict[d.replace('\n','')]=int(t.replace('\n',''))
    return train_Dict

def gold_readfile():
    fp = open("CorpusType/CuttestType.txt", "r")
    gold = fp.readlines()
    fp.close()

    for (idx,g) in enumerate(gold):
        gold[idx]=int(g.replace('\n',''))
    return gold

def to_class():
    # traindict=type_dict()
    Predict=BM25()
    # Predict=[]
    # for p in Predict_result:
    #     Predict.append(traindict[p])
    return Predict        

if __name__ == '__main__':
    Predict=to_class()
    gold=gold_readfile()

    Correct=0
    for (p,g) in zip(Predict,gold):
        print(p)
        if(p==g):
            Correct+=1
    print("準確率")
    print(Correct/len(Predict))
    print(accuracy_score(gold, Predict))
    f1 = f1_score( gold, Predict, average='macro' )
    p = precision_score(gold, Predict, average='macro')
    r = recall_score(gold, Predict, average='macro')
    print("F1 Score:")
    print(f1)
    print("Precision:")
    print(p)
    print("Recall:")
    print(r)