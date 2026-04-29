import codecs
import os
from tool import my_evalution

def evaluate_full(i):
    # dir = 'save/先分解后门控加cls/za/'
    # dir = 'save/model_text/za/'
    # dir = 'save/model_t_y_dis/model_t_y_dis/五层/za/'
    dir = 'za_2/'
    fnames = os.listdir(dir)

    result = {}
    num = 0
    for fname in fnames:
        if not fname.startswith('EP'):
            continue

        reader = codecs.open(filename=dir+fname, mode='r', encoding='utf-8')
        sids = []
        while True:
            line = reader.readline()
            if len(line) == 0:
                break

            ss = line.strip().split('\t')
            sids.append(ss[1])
            if len(sids) == i:
                 break
        result[fname] = sids
        reader.close()
        num += 1
        # if num == 103:
        #     break

    print('共读取主题专利数目：' + str(num))

    # 读取评估数据
    data_dir = '../data'
    qrels = data_dir + '/test_qrels.txt'
    QRELS = my_evalution.readQRELS(qrels)

    # 进行数据评估
    my_evalution.evalute(result, QRELS)

evaluate_full(1000)
# evaluate_full(800)
# evaluate_full(600)
# evaluate_full(400)
# evaluate_full(300)
# evaluate_full(200)
# evaluate_full(100)
# evaluate_full(10)