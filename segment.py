import jieba
import sys

CutData=[]     #儲存Questions資料
CutDataType=[] #儲存Ans Label
# train data
with open("Corpus/RawData/train.tsv", mode="r", encoding="utf-8") as f:
    for line_id, line in enumerate(f):
        line = line.strip().split("\t")
        CutDataType.append(line[0])
        CutData.append(" ".join(jieba.cut(line[1]))) #讀到每一筆斷詞
# dev data
# with open("Corpus/RawData/dev.tsv", mode="r", encoding="utf-8") as f:
#     for line_id, line in enumerate(f):
#         line = line.strip().split("\t")
#         CutDataType.append(line[0])
#         CutData.append(" ".join(jieba.cut(line[1]))) #讀到每一筆斷詞

# train data+ dev data=all Questions
# 寫檔 all Questions
fp = open("Corpus/segmentData/CutTrain.txt", "a")
for cutdata in CutData:
    fp.write(cutdata+'\n')
fp.close()

# test data
with open("Corpus/RawData/test.tsv", mode="r", encoding="utf-8") as f:
    for line_id, line in enumerate(f):
        line = line.strip().split("\t")
        CutDataType.append(line[0])
        CutData.append(" ".join(jieba.cut(line[1]))) #讀到每一筆斷詞

# test data(user query)
# 寫檔 all Query
fp = open("Corpus/segmentData/CutTest.txt", "a")
for cutdata in CutData:
    fp.write(cutdata+'\n')
fp.close()