import sys
import argparse  
import os
from util import *


def base_config(cfg_dict):
    parser = argparse.ArgumentParser()

    # Training schedule
    parser.add_argument('--model_name', default='cornet', help='Training model')
    parser.add_argument('--optim', default=cfg_dict['optim'], help='set the optimizer of model [Adam or SGD]')
    parser.add_argument('--schedule', default=cfg_dict['schedule'], help='set the scheduler')
    parser.add_argument('--lr', default=cfg_dict['lr'], type=float, help='set base learning rate')
    parser.add_argument('--batch', default=cfg_dict['batch'], type=int, help='Batch Size for dataloader')
    parser.add_argument('--ave_batch', default=cfg_dict['ave_batch'], type=int, help='Number of batches for Backpropagation')
    parser.add_argument('--epoch', default=cfg_dict['epoch'], type=int, help='Training epoch')
    parser.add_argument('--step_size', default=cfg_dict['step_size'], type=str, help='Lr decrease steps')
    parser.add_argument('--gamma', default=cfg_dict['gamma'], type=float)
    parser.add_argument('--clip_gradient', default=cfg_dict['clip_gradient'], type=float, help='Max gradient')
    parser.add_argument('--test_batch', default=cfg_dict['test_batch'], type=int, help='Batch Size for Testing')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--size', default=320, type=int, help='Input size')
    parser.add_argument('--consist', action='store_true')
    parser.add_argument('--resdual', default=0.4, type=float)
    
    parser.add_argument('--weight_path', default='./weight/', help='Weight path of network')
    
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--stage', default=2, type=int, help='Stage')
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--orig_size', action='store_true')
    
    # Dataset setting
    parser.add_argument('--trset', default='tno', help='Set the traing set')
    parser.add_argument('--vals', default='tno', help='Set the testing sets')  # SOD,PASCAL-S,ECSSD,DUTS-TE,HKU-IS,DUT-OMRON
    parser.add_argument('--data_path', default='./dataset', help='The name of network')
    
    parser.add_argument('--backbone', default='resnet', help='Set the backbone of the model')
    parser.add_argument('--gpus', default=None, type=str, help='Set the cuda devices')
    parser.add_argument('--sub', default='base', help='Job name')
    
    parser.add_argument('--debug', action='store_true')

    # testing
    parser.add_argument('--weight', default='', help='Trained weight path')
    parser.add_argument('--save', default=True, action='store_true')
    parser.add_argument('--save_path', default='./result/', help='Save path of network')
    parser.add_argument('--crf', action='store_true')
    
    # Construct loss by loss_factory. More details in methods\base\loss.py.
    # loss: 'b': BCE, 's': SSIM, 'i': IOU, 'd': DICE, 'e': Edge, 'c': CTLoss
    parser.add_argument('--loss', default='', type=str, help='Losses for networks')
    parser.add_argument('--lw', default='', type=str, help='Weights for losses')
    
    return parser


def cfg_convert(config):
    if config['vals'] == 'all':
        # config['vals'] = ['SOD', 'PASCAL-S', 'ECSSD', 'HKU-IS', 'DUTS-TE', 'DUT-OMRON', 'MSB-TE']
        config['vals'] = ['ECSSD', 'MSB-TE', 'DUT-OMRON']
    else:
        config['vals'] = config['vals'].split(',')
    
    if config['step_size'] != '':
        step_sizes = config['step_size'].split(',')
        config['step_size'] = [int(ss) for ss in step_sizes]

    save_path = os.path.join(config['save_path'], config['model_name'], config['backbone'], config['sub'])
    check_path(save_path)
    config['save_path'] = save_path
    
    weight_path = os.path.join(config['weight_path'], config['model_name'], config['backbone'], config['sub'])
    check_path(weight_path)
    config['weight_path'] = weight_path
