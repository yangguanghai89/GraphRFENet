import codecs, datetime
import numpy as np

def ACCURACY_func (real,pred):
    real = real.cpu().numpy()
    pred = pred.cpu().numpy()

    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1

    acc = (pred == real).sum().astype('float32')

    return acc

def readQRELS(fname):           #从文件中读取数据，并将主专利作为键并且与他相似的专利作为值组成集合储存在字典中
    result = {}
    reader = codecs.open(filename=fname, mode='r', encoding='utf-8')
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        ss = line.split('\t')
        if ss[0] not in result.keys():
            result[ss[0]] = []
        result[ss[0]].append(ss[1])
    reader.close()
    return result

def computePerformance(results, QRELS):     #results自己的   计算各种指标
    n = 0
    sum_recall = 0.0    #召回率
    sum_accuracy = 0.0  #准确率
    sum_map = 0.0       #平均精度均值
    sum_pres = 0.0      #精确度
    for tid, sids_ret in results.items():
        if tid in QRELS.keys():
            n += 1
        else:
            continue

        sids_qrel = QRELS[tid]

        recall, accuracy, map, pres = computePerformanceForOnePatent(sids_ret, sids_qrel)
        sum_recall += recall
        sum_accuracy += accuracy
        sum_map += map
        sum_pres += pres

    if n == 0:
        print(123)

    return sum_recall/n, sum_accuracy/n, sum_map/n, sum_pres/n


def computePerformanceForOnePatent(sids_ret, sids_qrel):
    nCount = 0
    map = 0.0
    sumRank = 0.0                  # 用于计算PRES
    for i in range(len(sids_ret)):
        sid_ret = sids_ret[i]
        if sid_ret in sids_qrel:
            nCount += 1
            map += nCount/(i+1)
            sumRank += i+1

    # 计算召回率
    recall = (float(nCount))/len(sids_qrel)

    # 计算准确率
    accuracy = (float(nCount))/len(sids_ret)

    # 计算map值
    if nCount == 0:
        map = 0.0
    else:
        #map = map/len(sids_qrel)
        map = map / nCount

    # 计算PRES值
    n = len(sids_qrel)
    nMax = len(sids_ret)
    pres = 0.0
    if n*nMax != 0:
        nCollection = nMax + n
        remain = n - nCount
        sumRank += remain * (nCollection - (remain-1)/(float(2)))
        pres = 1 - (sumRank-(n*(n+1)/float(2)))/(n*(nCollection-n))
        if (pres < 0.0) | (pres > 1.0):
            print('Error:PRES-->' + str(pres) + '不符合规范！')

    return recall, accuracy, map, pres

# TOP1000
def mergeResult(tid, sids, labels, weights, address, topN=1000, bSave=True):     #找出其中最相关的前1000个数据
    weights = [float(x) for x in weights]
    sid2weight = {}
    sid2label = {}
    for i in range(len(sids)):
        sid2weight[sids[i]] = weights[i]
        sid2label[sids[i]] = labels[i]

    sid2weight_tuple = sorted(sid2weight.items(), key=lambda x: x[1], reverse=True)
    predicts_real = list()

    outstr = ''
    nPos = 0
    for i in range(len(sid2weight_tuple)):
        # 信息输出
        sid = sid2weight_tuple[i][0]
        outstr += str(i) + '\t' + sid + '\t' + str(sid2weight_tuple[i][1]) + '\t' + str(sid2label[sid]) + '\n'

        # TOP N
        if i < topN:
            predicts_real.append(sid)
            if sid2label[sid] == 1:
                nPos += 1

    if bSave:
        writer = codecs.open(filename = address + tid, mode='w', encoding='utf-8')
        # writer.write('{}：召回率{}\n'.format(tid, float(nPos)/len(QRELS[tid])))
        writer.write(outstr)
        writer.close()

    return predicts_real

def evalute(ret_results, QRELS, tid='All Patents'):

    Recall, Accuracy, MAP, PRES = computePerformance(ret_results, QRELS)

    datetime_object = datetime.datetime.now()
    outstr = 'Current Time:{}\n'.format(str(datetime_object))
    #outstr += 'hinge_loss_bd:{}\n'.format(str(bd))
    outstr += 'adding patent:{}\n'.format(tid)
    outstr += 'Average Recall of {} topic patents:{}\n'.format(len(ret_results), str(Recall))
    outstr += 'Average Accuracy of {} topic patents:{}\n'.format(len(ret_results), str(Accuracy))
    outstr += 'Average MAP of {} topic patents:{}\n'.format(len(ret_results), str(MAP))
    outstr += 'Average PRES of {} topic patents:{}\n\n'.format(len(ret_results), str(PRES))
    print(outstr)

    dir = 'save/'
    path = dir + 'result.txt'
    writer = codecs.open(filename=path, mode='a+', encoding='utf-8')
    writer.write(outstr)
    writer.close()

    return Recall, Accuracy, MAP, PRES





