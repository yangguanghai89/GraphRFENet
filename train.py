import torch
import argparse
from tqdm import tqdm
from tool import utils
from model import model_t_y_rep_3_sage_ref
from torch.utils.data import DataLoader
from transformers import logging
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

logging.set_verbosity_error()

def train(args):
    #load dataset
    train_data = utils.load_data_withopen(args.train_path, args)
    valid_data = utils.load_data_withopen(args.valid_path, args)
    utils.setting(args, train_data, valid_data)
    # g = torch.Generator()
    # g.manual_seed(args.seed)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) #, generator=g
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    batch_number, best_totalperformance, best_yperformance = 0, 1000000, 1000000

    # initialize model and optimizer
    net = model_t_y_rep_3_sage_ref.net(args).to(args.device)
    optimizer = AdamW(net.parameters(), lr=args.learning_rate, weight_decay=0.0005)
    # optimizer = utils.set_lr(args, net)

    #visualization
    writer = SummaryWriter('log_train')

    # start train
    total_train_loss = 0
    for i in range(args.epoch):
        net.train()
        for data in tqdm(train_dataloader, desc="Training", unit="batch"):
            input_data = utils.batch_data(args, data)
            input_data['tr'] = True
            batch_number += 1
            # output_data = net(input_data, True)
            output_data = net(input_data)
            train_loss, loss_train = net.loss_func()
            total_train_loss = total_train_loss + train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if batch_number % 50 == 0:
                utils.print_time('第{}个batch的验证的时间：'.format(batch_number))
                # print("......第{}个batch的损失值为:{}".format(batch_number, loss.item()))
                writer.add_scalar("train_loss", total_train_loss / batch_number, batch_number)

                #start valid
                net.eval()
                valid_step, total_valid_loss, total_valid_y_loss= 0, 0, 0
                with torch.no_grad():
                    for data in tqdm(valid_dataloader, desc="validing", unit="batch"):
                        input_data = utils.batch_data(args, data)
                        input_data['tr'] = False
                        output_data = net(input_data)
                        # output_data = net(input_data, False)
                        valid_loss, loss_valid = net.loss_func()

                        total_valid_loss += valid_loss.item()
                        total_valid_y_loss += loss_valid['loss_y'].item()
                        valid_step += 1

                writer.add_scalar("valid_loss", total_valid_loss / valid_step, batch_number)
                writer.add_scalar("valid_y", total_valid_y_loss / valid_step, batch_number)
                writer.flush()
                # figure_loss.create_list(total_train_loss / batch_number, total_valid_loss / valid_step, total_valid_y_loss / valid_step, batch_number)
                # if batch_number == 2500:
                #     figure_loss.figure("loss_t/model_text.tsv")
                #     sys.exit(0)

                if (best_totalperformance > (total_valid_loss / valid_step)):
                    valid_average_loss = total_valid_loss / valid_step
                    best_totalperformance = valid_average_loss
                    torch.save(net.state_dict(), 'save/{}_{}_{}.pth'.format(batch_number, total_valid_y_loss / valid_step, total_valid_loss / valid_step))

    writer.close()

if __name__ == '__main__':
    time_start = utils.print_time('程序开始时间：')
    parser = argparse.ArgumentParser()
    args = utils.get_parsere(parser)
    train(args = args)
    time_end = utils.print_time('程序结束时间：')
    print('本程序一共执行了:{}'.format(time_end - time_start))