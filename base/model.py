import importlib

from .encoder.cy2net import CyFusion, TyFusion
import torch


def Network(net_name, config):

    encoder = CyFusion().to('cpu')
    temp = TyFusion().to('cpu')
    # 导入预训练模型，得到结构和参数
    temp.load_state_dict(torch.load('results/road_model_epoch_29.pth'))
    pretrained_dict = temp.state_dict()
    # 调用自己设置的模型，也得到结构即相应参数
    model_conv_dict = encoder.state_dict()
    # 当模型中的某层是同时在两个模型中共有时才取出，即得到除了全连接层以外的所有层次对应的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_conv_dict}
    # 然后用该新参数的值取更新你自己的模型的参数
    # 这样，除了你修改的全连接层外，其他层次的参数就都是预训练模型的参数了
    model_conv_dict.update(pretrained_dict)
    # 然后将参数导入你的模型即可
    encoder.load_state_dict(model_conv_dict)

    fl = [16, 16, 32, 64, 128]  # PIA模型
    # fl = [16, 64, 112, 160, 208]  # SOS模型
    model = importlib.import_module('methods.{}.collamodel'.format(net_name)).Network(config, encoder, fl)
    dlm_model = importlib.import_module('methods.{}.CFmodel'.format(net_name)).Network(config, encoder, fl)  # 第二阶段模型

    return encoder, model, dlm_model


def ReNetwork(net_name, config):
    input_nc = 1
    output_nc = 1

    # nb_filter = [64, 112, 160, 208, 256]
    # # encoder的输出包括四个：[1, 64, 450, 620]，[1, 112, 225, 310], [1, 160, 112, 155], [1, 208, 56, 77]
    # encoder = SOSMaskFuse_autoencoder(nb_filter, input_nc, output_nc).to('cpu')
    # temp = SOSMaskFuse(nb_filter, input_nc, output_nc).to('cpu')
    # # 导入预训练模型，得到结构和参数
    # temp.load_state_dict(torch.load('results/Epoch_model_37.model'))
    # pretrained_dict = temp.state_dict()
    # # 调用自己设置的模型，也得到结构即相应参数
    # model_conv_dict = encoder.state_dict()
    # # 当模型中的某层是同时在两个模型中共有时才取出，即得到除了全连接层以外的所有层次对应的参数
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_conv_dict}
    # # 然后用该新参数的值取更新你自己的模型的参数
    # model_conv_dict.update(pretrained_dict)
    # # 然后将参数导入你的模型即可
    # encoder.load_state_dict(model_conv_dict)

    # 导入PIA模型
    encoder = CyFusion().to('cpu')
    temp = TyFusion().to('cpu')
    temp.load_state_dict(torch.load('results/road_model_epoch_29.pth'))
    pretrained_dict = temp.state_dict()
    model_conv_dict = encoder.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_conv_dict}
    model_conv_dict.update(pretrained_dict)
    encoder.load_state_dict(model_conv_dict)

    # nb_filter = [64, 112, 160, 208, 256]
    # encoder = NestFuse_light2(nb_filter, input_nc, output_nc, deepsupervision=False).to('cpu')
    # temp = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision=False).to('cpu')
    # temp.load_state_dict(torch.load('results/nestfuse_gray_1e2.model'))
    # pretrained_dict = temp.state_dict()
    # model_conv_dict = encoder.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_conv_dict}
    # model_conv_dict.update(pretrained_dict)
    # encoder.load_state_dict(model_conv_dict)

    # encoder = encoder.cuda()
    # fl = [16, 16, 112, 160, 208]  # SOS网络参数和nest网络参数
    fl = [16, 16, 32, 64, 128]  # PIA网络参数
    model = importlib.import_module('methods.{}.model'.format(net_name)).Network(config, encoder, fl)
    # model.load_state_dict(torch.load('results/next_model.pth'))

    return model
