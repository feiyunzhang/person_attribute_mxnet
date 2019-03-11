# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet
from common import data, fit, modelzoo
from common import  config
import mxnet as mx

def feature_transform(net,num_hidden = 256, drop_out_rate=0.5):
    net = mx.symbol.Convolution(net,kernel=(1,1),stride=(1, 1),num_filter = num_hidden *2)
    net = mx.symbol.BatchNorm(data=net, use_global_stats=False, fix_gamma=False, eps=1e-5)
    net = mx.symbol.Activation(net,act_type='relu')
    net = mx.symbol.Pooling(data=net, global_pool=True, kernel=(7, 7), pool_type='avg')
    net = mx.symbol.Dropout(net, p = drop_out_rate)
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_hidden)
    net = mx.symbol.Activation(net,act_type='relu')
    return net

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """

    # downdress_label = mx.sym.Variable('downdress_label')

    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    num_hidden = 1024
    drop_out_rate = 0.5
    net_gender_split = net
    lam =config.get_lam_value()
    print("lam",lam)
    # # net = mx.symbol.Convolution(net_gender_split, kernel=(1, 1), stride=(1, 1), num_filter=num_hidden * 2)
    # # net = mx.symbol.BatchNorm(data=net, use_global_stats=False, fix_gamma=False, eps=1e-5)
    # # net = mx.symbol.Activation(net, act_type='relu')
    # #net = mx.symbol.Pooling(data=net, global_pool=True, kernel=(7, 7), pool_type='avg')
    # net = mx.symbol.Dropout(net, p=drop_out_rate)
    # net = mx.symbol.FullyConnected(data=net, num_hidden=num_hidden)
    # net_gender_hidden = mx.symbol.Activation(net, act_type='relu')

    net_gender_drop = mx.symbol.Dropout(net_gender_split, p=0.5)
    net_gender_hidden = mx.symbol.FullyConnected(data=net_gender_drop, num_hidden=num_hidden, name='fc-gender-hidden')
    net_gender_hidden = mx.symbol.Activation(net_gender_hidden, act_type='relu')
    net_gender_fc = mx.symbol.FullyConnected(data=net_gender_hidden, num_hidden=2, name='fc-gender')
    # net_gender_fc = mx.symbol.BlockGrad(net_gender_fc)
    net_gender = mx.symbol.SoftmaxOutput(data=net_gender_fc, name='gender', grad_scale=1,ignore_label= -1,use_ignore=True)
    net_gender_mix = mx.symbol.SoftmaxOutput(data=net_gender_fc,name='gender_mix',grad_scale=1,ignore_label= -1,use_ignore=True)
    net_gender = lam * net_gender + (1 - lam) * net_gender_mix
    # net_hat_split = net
    # net_hat = mx.symbol.Dropout(net_hat_split, p=0.5)
    # net_hat_hidden = mx.symbol.FullyConnected(data=net_hat, num_hidden=num_hidden, name='fc-hat-hidden')
    # net_hat_fc = mx.symbol.FullyConnected(data=net_hat_hidden, num_hidden=2, name='fc-hat')
    # # net_hat_fc = mx.symbol.BlockGrad(net_hat_fc)
    # net_hat = mx.symbol.SoftmaxOutput(data=net_hat_fc, name='hat', grad_scale=0.1)

    net_bag_split = net
    net_bag_drop = mx.symbol.Dropout(net_bag_split, p=0.5)
    net_bag_hidden = mx.symbol.FullyConnected(data=net_bag_drop, num_hidden=num_hidden, name='fc-bag-hidden')
    net_bag_fc = mx.symbol.FullyConnected(data=net_bag_hidden, num_hidden=2, name='fc-bag')
    # net_bag_fc = mx.symbol.BlockGrad(net_bag_fc)
    net_bag = mx.symbol.SoftmaxOutput(data=net_bag_fc, name='bag', grad_scale=1,ignore_label= -1,use_ignore=True)
    net_bag_mix = mx.symbol.SoftmaxOutput(data=net_bag_fc,name='bag_mix',grad_scale=1,ignore_label= -1,use_ignore=True)
    net_bag = lam * net_bag + (1-lam) * net_bag_mix

    net_handbag_split = net
    net_handbag_drop = mx.symbol.Dropout(net_handbag_split, p=0.5)
    net_handbag_hidden = mx.symbol.FullyConnected(data=net_handbag_drop, num_hidden=num_hidden, name='fc-handbag-hidden')
    net_handbag_fc = mx.symbol.FullyConnected(data=net_handbag_hidden, num_hidden=2, name='fc-handbag')
    # net_handbag_fc = mx.symbol.BlockGrad(net_handbag_fc)
    net_handbag = mx.symbol.SoftmaxOutput(data=net_handbag_fc, name='handbag', grad_scale=1,ignore_label= -1,use_ignore=True)
    net_handbag_mix = mx.symbol.SoftmaxOutput(data=net_handbag_fc,name='handbag_mix',grad_scale=1,ignore_label= -1,use_ignore=True)
    net_handbag = lam * net_handbag + (1-lam) * net_handbag_mix

    net_backpack_split = net
    net_backpack_drop = mx.symbol.Dropout(net_backpack_split, p=0.5)
    net_backpack_hidden = mx.symbol.FullyConnected(data=net_backpack_drop, num_hidden=num_hidden, name='fc-backpack-hidden')
    net_backpack_fc = mx.symbol.FullyConnected(data=net_backpack_hidden, num_hidden=2, name='fc-backpack')
    # net_backpack_fc = mx.symbol.BlockGrad(net_backpack_fc)
    net_backpack = mx.symbol.SoftmaxOutput(data=net_backpack_fc, name='backpack', grad_scale=1,ignore_label= -1,use_ignore=True)
    net_backpack_mix = mx.symbol.SoftmaxOutput(data=net_backpack_fc, name='backpack_mix',grad_scale=1,ignore_label= -1,use_ignore=True)
    net_backpack = lam * net_backpack + (1- lam) * net_backpack_mix

    net_updress_split = net
    net_updress_drop = mx.symbol.Dropout(net_updress_split, p=0.5)
    net_updress_hidden = mx.symbol.FullyConnected(data=net_updress_drop, num_hidden=num_hidden, name='fc-updress-hidden')
    net_updress_fc = mx.symbol.FullyConnected(data=net_updress_hidden, num_hidden=11, name='fc-updress')
    # net_updress_fc = mx.symbol.BlockGrad(net_updress_fc)
    net_updress = mx.symbol.SoftmaxOutput(data=net_updress_fc, name='updress',grad_scale=1,ignore_label= -1,use_ignore=True)
    net_updress_mix = mx.symbol.SoftmaxOutput(data=net_updress_fc,name='updress_mix',grad_scale=1,ignore_label= -1,use_ignore=True)
    net_updress = lam * net_updress + (1-lam) * net_updress_mix

    net_downdress_split = net
    net_downdress_drop = mx.symbol.Dropout(net_downdress_split, p=0.5)
    net_downdress_hidden = mx.symbol.FullyConnected(data=net_downdress_drop, num_hidden=num_hidden,
                                                    name='fc-downdress-hidden')
    net_downdress_fc = mx.symbol.FullyConnected(data=net_downdress_hidden, num_hidden=12, name='fc-downdress')
    # net_downdress_fc = mx.symbol.BlockGrad(net_downdress_fc)
    net_downdress = mx.symbol.SoftmaxOutput(data=net_downdress_fc, name='downdress', grad_scale=1,ignore_label= -1,use_ignore=True)
    net_downdress_mix =mx.symbol.SoftmaxOutput(data=net_downdress_fc, name='downdress_mix',grad_scale=1,ignore_label= -1,use_ignore=True)
    net_downdress = lam * net_downdress + (1-lam) * net_downdress_mix

    net_hatclolor_split = net
    net_hatclolor_drop = mx.symbol.Dropout(net_hatclolor_split, p=0.5)
    net_hatclolor_hidden = mx.symbol.FullyConnected(data=net_hatclolor_drop, num_hidden=num_hidden,
                                                    name='fc-hatclolor-hidden')
    net_hatclolor_fc = mx.symbol.FullyConnected(data=net_hatclolor_hidden, num_hidden=10, name='fc-hatcolor')
    # net_downdress_fc = mx.symbol.BlockGrad(net_downdress_fc)
    net_hatclolor = mx.symbol.SoftmaxOutput(data=net_hatclolor_fc, name='hatcolor', grad_scale=1,ignore_label= -1,use_ignore=True)
    net_hatclolor_mix =mx.symbol.SoftmaxOutput(data=net_hatclolor_fc, name='hatcolor_mix',grad_scale=1,ignore_label= -1,use_ignore=True)
    net_hatcolor = lam * net_hatclolor + (1-lam) * net_hatclolor_mix

    net_umbrella_split = net
    net_umbrella_drop = mx.symbol.Dropout(net_umbrella_split, p=0.5)
    net_umbrella_hidden = mx.symbol.FullyConnected(data=net_umbrella_drop, num_hidden=num_hidden,
                                                    name='fc-umbrella-hidden')
    net_umbrella_fc = mx.symbol.FullyConnected(data=net_umbrella_hidden, num_hidden=12, name='fc-umbrella')
    # net_downdress_fc = mx.symbol.BlockGrad(net_downdress_fc)
    net_umbrella = mx.symbol.SoftmaxOutput(data=net_umbrella_fc, name='umbrella', grad_scale=1,ignore_label=-1, use_ignore=True)
    net_umbrella_mix = mx.symbol.SoftmaxOutput(data=net_umbrella_fc, name='umbrella_mix',grad_scale=1,ignore_label=-1,use_ignore=True)
    net_umbrella = lam* net_umbrella + (1-lam) * net_umbrella_mix

    net_shoes_split = net
    net_shoes_drop = mx.symbol.Dropout(net_shoes_split, p=0.5)
    net_shoes_hidden = mx.symbol.FullyConnected(data=net_shoes_drop, num_hidden=num_hidden,
                                                    name='fc-shoes-hidden')
    net_shoes_fc = mx.symbol.FullyConnected(data=net_shoes_hidden, num_hidden=8, name='fc-shoes')
    # net_downdress_fc = mx.symbol.BlockGrad(net_downdress_fc)
    net_shoes = mx.symbol.SoftmaxOutput(data=net_shoes_fc, name='shoes', grad_scale=1,ignore_label=-1,use_ignore=True)
    net_shoes_mix = mx.symbol.SoftmaxOutput(data=net_shoes_fc, name= 'shoes_mix',grad_scale=1,ignore_label=-1,use_ignore=True)
    net_shoes = lam * net_shoes + (1-lam) * net_shoes_mix

    new_args = dict({k: arg_params[k] for k in arg_params if 'fc' not in k})

    # net = mx.symbol.Group([mx.symbol.BlockGrad(net_gender), mx.symbol.BlockGrad(net_hat), mx.symbol.BlockGrad(net_bag),
    #                       mx.symbol.BlockGrad(net_handbag), mx.symbol.BlockGrad(net_backpack)
    #                       ,mx.symbol.BlockGrad(net_updress), mx.symbol.BlockGrad(net_downdress)])
    net = mx.symbol.Group([net_gender, net_bag, net_handbag, net_backpack, net_updress, net_downdress, net_hatcolor, net_umbrella, net_shoes])

    return (net, new_args)

def set_imagenet_aug(aug):
    # standard data augmentation setting for imagenet training
    aug.set_defaults(rgb_mean='123.68,116.779,103.939', rgb_std='58.393,57.12,57.375')
    aug.set_defaults(random_crop=0, random_resized_crop=0, random_mirror=1)
    aug.set_defaults(min_random_area=0.08)
    aug.set_defaults(max_random_aspect_ratio=4./3., min_random_aspect_ratio=3./4.)
    aug.set_defaults(brightness=0.4, contrast=0.4, saturation=0.4, pca_noise=0.1)



if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    parser.add_argument('--pretrained-model', type=str,
                        help='the pre-trained model')
    parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    # use less augmentations for fine-tune
    data.set_data_aug_level(parser, 1)
    # use a small learning rate and less regularizations
    parser.set_defaults(image_shape='3,224,224', num_epochs=30,
                        lr=.01, lr_step_epochs='20', wd=0, mom=0)

    args = parser.parse_args()

    # load pretrained model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    (prefix, epoch) = modelzoo.download_model(
        args.pretrained_model, os.path.join(dir_path, 'model'))
    if args.load_epoch is not None:
        (prefix, epoch) = (args.model_prefix, args.load_epoch)
    logging.info(prefix)
    logging.info(epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # remove the last fullc layer
    (new_sym, new_args) = get_fine_tune_model(
        sym, arg_params, args.num_classes, args.layer_before_fullc)
    print(new_sym)
    arg_shape, out_shape, aux_shape = new_sym.infer_shape(data=(64, 3, 224, 112))
    print("out-shape")
    print(out_shape)

    print("arg_shape")
    print(arg_shape)

    # train
    fit.fit(args=args,
            network=new_sym,
            data_loader=data.get_rec_iter_mutil_mixup,
            arg_params=new_args,
            aux_params=aux_params)

