#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time
import random

from progress.bar import Bar
from collections import OrderedDict
from util import *
from data import Tno_Dataset, Train_TNODataset
from self import test_model
import torch
from torch.nn import utils
from base.framework_factory import load_framework
import torch.nn.functional as F
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'


def main():
    net_name = "smnet"
    # Loading model
    config, model, _, _, optim, _, _, sche, model_loss, saver = load_framework(net_name)
    config['stage'] = 2

    # Loading datasets
    train_loader = Train_TNODataset()  # get_loader(config)
    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Tno_Dataset(name=set_name, config=config)

    debug = config['debug']
    num_epoch = 100
    num_iter = train_loader.size
    ave_batch = config['ave_batch']
    batch_idx = 0

    model.zero_grad()
    rank_criterion = torch.nn.MarginRankingLoss(0.)
    best_iou = 0.0
    for epoch in range(1, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()

        st = time.time()
        loss_count = 0
        optim.zero_grad()
        sche.step()
        iter_per_epoch = num_iter // config['batch']
        index_list = np.array(range(num_iter))
        random.shuffle(index_list)
        index_list = index_list[:iter_per_epoch * config['batch']]
        index_list = np.array(index_list).reshape((iter_per_epoch, config['batch']))

        print('Current LR: {:.6f}.'.format(optim.param_groups[1]['lr']))
        bar = Bar('{:10}-{:8} | epoch {:2}:'.format(net_name, config['sub'], epoch), max=iter_per_epoch)

        lamda = config['resdual']
        for i, idx_list in enumerate(index_list):

            name, images, inf_images, gts, _, var = train_loader[idx_list[0]]

            if config['multi']:
                scales = [-1, 0, 1]
                # scales = [-2, -1, 0, 1, 2]
                input_size = config['size']
                input_size += int(np.random.choice(scales, 1) * 64)
                images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')
            fused_image, vis_rec, inf_rec, out, Y, v_pos, v_neg, i_pos, i_neg\
             , depth_pos_i, depth_neg_i, depth_pos_v, depth_neg_v = model(images, inf_images, 'train')

            loss = model_loss(Y, gts, var, config) / ave_batch

            v_pos = torch.softmax(v_pos, dim=1)
            r1 = (images * v_pos).mean([-1, -2]).squeeze(1)  # 求平均值
            v_neg = torch.softmax(v_neg, dim=1)
            r2 = (images * v_neg).mean([-1, -2]).squeeze(1)
            y = torch.full((1 * 2,), -1)  #.cuda()
            rank_loss1 = 5 * rank_criterion(r1, r2, y)  # 排序损失

            i_pos = torch.softmax(i_pos, dim=1)
            r1 = (images * i_pos).mean([-1, -2]).squeeze(1)  # 求平均值
            i_neg = torch.softmax(i_neg, dim=1)
            r2 = (images * i_neg).mean([-1, -2]).squeeze(1)
            y = torch.full((1 * 2,), -1)  #.cuda()
            rank_loss2 = 5 * rank_criterion(r1, r2, y)

            entropy_loss = 5 * -(v_pos * torch.log(v_pos + 1e-5)).sum(dim=1).mean()  # 熵损失
            entropy_loss2 = 5 * -(i_pos * torch.log(i_pos + 1e-5)).sum(dim=1).mean()

            loss_count += loss.data
            total_loss = loss + (rank_loss1 + rank_loss2) / 2 + (entropy_loss+entropy_loss2)/2 #+ (loss_depth+loss_consistency)/2

            total_loss.backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0

            Bar.suffix = '{:4}/{:4} | loss: {:1.5f}, time: {}.'.format(i, iter_per_epoch,
                                                                       round(float(loss_count / i), 5),
                                                                       round(time.time() - st, 3))
            bar.next()
            # 更新伪标签
            if epoch > 3:
                train_loader.save(name, gts * lamda + torch.sigmoid(Y['final'].detach()) * (1 - lamda))

        if epoch > 3:
            train_loader.update()  # 伪标签地址更换

        bar.finish()
        best_iou = test_model(model, test_sets, config, epoch, best_iou)


if __name__ == "__main__":
    main()