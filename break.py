import torch
from tqdm import tqdm
import argparse
from model import ref_1
from tool import my_evalution, utils
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import logging

# 加载BERT模型和分词器
logging.set_verbosity_error()

def train(path, args, address):
    test_data = utils.load_data_withopen(args.test_path, args)
    test_data_size = len(test_data)
    print('测试集的大小是{}'.format(test_data_size))
    test_dataloader = DataLoader(test_data, batch_size=100, shuffle=False)

    # 初始化模型结构
    checkpoint = torch.load(path)
    net = ref_1.net(args).to(args.device)

    # 加载模型参数
    net.load_state_dict(checkpoint)


    total_test_step, i = 0, 0
    net.eval()
    print("----------测试开始----------")
    lst, my_result = [], {}
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="testing", unit="batch"):
            i = i + 1
            if i > 0:
                input_data = utils.batch_data(args, data)
                input_data['tr'] = False
                output_data = net(input_data)
                # output_data = net(input_data, False)
                pred_y = output_data['pred_y']

                combined = np.concatenate((np.expand_dims(np.array(input_data['patentA']), axis=1),
                                           np.expand_dims(np.array(input_data['patentB']), axis=1),
                                           input_data['label'].unsqueeze(1).to('cpu').numpy(),
                                           pred_y.unsqueeze(1).to('cpu').numpy()), axis=1)
                lst.extend(combined)
                if len(lst) == 1000:
                    total_test_step += 1
                    result_list = []
                    result_list.extend(my_evalution.mergeResult(tid=lst[0][0],
                                                                sids=[row[1] for row in lst],
                                                                labels=[row[2] for row in lst],
                                                                weights=[row[3] for row in lst],
                                                                address=address))
                    my_result[lst[0][0]] = result_list
                    print(total_test_step)
                    lst.clear()
        actual = my_evalution.readQRELS('../data/test_qrels.txt')
        Recall, Accuracy, MAP, PRES = my_evalution.evalute(my_result, actual)

if __name__ == '__main__':
    time_start = utils.print_time('程序开始时间：')
    parser = argparse.ArgumentParser()
    args = utils.get_parsere(parser)
    train(path ='save/850_0.020237310189306253_0.020237310189306253.pth', args=args, address='za/')
    time_end = utils.print_time('程序结束时间：')
    print('本程序一共执行了:{}'.format(time_end - time_start))