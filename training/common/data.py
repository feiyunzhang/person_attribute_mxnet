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

import mxnet as mx
import random
from mxnet.io import DataBatch, DataIter
import numpy as np
import config

def add_data_args(parser):
    data = parser.add_argument_group('Data', 'the input images')
    data.add_argument('--data-train', type=str, help='the training data')
    data.add_argument('--data-train-idx', type=str, default='', help='the index of training data')
    data.add_argument('--data-val', type=str, help='the validation data')
    data.add_argument('--data-val-idx', type=str, default='', help='the index of validation data')
    data.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939',
                      help='a tuple of size 3 for the mean rgb')
    data.add_argument('--rgb-std', type=str, default='1,1,1',
                      help='a tuple of size 3 for the std rgb')
    data.add_argument('--pad-size', type=int, default=0,
                      help='padding the input image')
    data.add_argument('--fill-value', type=int, default=127,
                      help='Set the padding pixels value to fill_value')
    data.add_argument('--image-shape', type=str,
                      help='the image shape feed into the network, e.g. (3,224,224)')
    data.add_argument('--num-classes', type=int, help='the number of classes')
    data.add_argument('--num-examples', type=int, help='the number of training examples')
    data.add_argument('--resize-train', type=int, help='resize size of train data')
    data.add_argument('--resize-val', type=int, help='rezize size of val data')
    data.add_argument('--data-nthreads', type=int, default=4,
                      help='number of threads for data decoding')
    data.add_argument('--benchmark', type=int, default=0,
                      help='if 1, then feed the network with synthetic data')
    return data

def add_data_aug_args(parser):
    aug = parser.add_argument_group(
        'Image augmentations', 'implemented in src/io/image_aug_default.cc')
    aug.add_argument('--random-crop', type=int, default=0,
                     help='if or not randomly crop the image')
    aug.add_argument('--random-mirror', type=int, default=0,
                     help='if or not randomly flip horizontally')
    aug.add_argument('--max-random-h', type=int, default=0,
                     help='max change of hue, whose range is [0, 180]')
    aug.add_argument('--max-random-s', type=int, default=0,
                     help='max change of saturation, whose range is [0, 255]')
    aug.add_argument('--max-random-l', type=int, default=0,
                     help='max change of intensity, whose range is [0, 255]')
    aug.add_argument('--min-random-aspect-ratio', type=float, default=None,
                     help='min value of aspect ratio, whose value is either None or a positive value.')
    aug.add_argument('--max-random-aspect-ratio', type=float, default=0,
                     help='max value of aspect ratio. If min_random_aspect_ratio is None, '
                          'the aspect ratio range is [1-max_random_aspect_ratio, '
                          '1+max_random_aspect_ratio], otherwise it is '
                          '[min_random_aspect_ratio, max_random_aspect_ratio].')
    aug.add_argument('--max-random-rotate-angle', type=int, default=0,
                     help='max angle to rotate, whose range is [0, 360]')
    aug.add_argument('--max-random-shear-ratio', type=float, default=0,
                     help='max ratio to shear, whose range is [0, 1]')
    aug.add_argument('--max-random-scale', type=float, default=1,
                     help='max ratio to scale')
    aug.add_argument('--min-random-scale', type=float, default=1,
                     help='min ratio to scale, should >= img_size/input_shape. '
                          'otherwise use --pad-size')
    aug.add_argument('--max-random-area', type=float, default=1,
                     help='max area to crop in random resized crop, whose range is [0, 1]')
    aug.add_argument('--min-random-area', type=float, default=1,
                     help='min area to crop in random resized crop, whose range is [0, 1]')
    aug.add_argument('--min-crop-size', type=int, default=-1,
                     help='Crop both width and height into a random size in '
                          '[min_crop_size, max_crop_size]')
    aug.add_argument('--max-crop-size', type=int, default=-1,
                     help='Crop both width and height into a random size in '
                          '[min_crop_size, max_crop_size]')
    aug.add_argument('--brightness', type=float, default=0,
                     help='brightness jittering, whose range is [0, 1]')
    aug.add_argument('--contrast', type=float, default=0,
                     help='contrast jittering, whose range is [0, 1]')
    aug.add_argument('--saturation', type=float, default=0,
                     help='saturation jittering, whose range is [0, 1]')
    aug.add_argument('--pca-noise', type=float, default=0,
                     help='pca noise, whose range is [0, 1]')
    aug.add_argument('--random-resized-crop', type=int, default=0,
                     help='whether to use random resized crop')
    return aug

def set_data_aug_level(aug, level):
    if level >= 1:
        aug.set_defaults(random_crop=1, random_mirror=1)
    if level >= 2:
        aug.set_defaults(max_random_h=36, max_random_s=50, max_random_l=50)
    if level >= 3:
        aug.set_defaults(max_random_rotate_angle=10, max_random_shear_ratio=0.1, max_random_aspect_ratio=0.25)


class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter, dtype):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        self.dtype = dtype
        label = np.random.randint(0, num_classes, [self.batch_size,])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = mx.nd.array(data, dtype=self.dtype, ctx=mx.Context('cpu_pinned', 0))
        self.label = mx.nd.array(label, dtype=self.dtype, ctx=mx.Context('cpu_pinned', 0))
    def __iter__(self):
        return self
    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, self.dtype)]
    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label', (self.batch_size,), self.dtype)]
    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return DataBatch(data=(self.data,),
                             label=(self.label,),
                             pad=0,
                             index=None,
                             provide_data=self.provide_data,
                             provide_label=self.provide_label)
        else:
            raise StopIteration
    def __next__(self):
        return self.next()
    def reset(self):
        self.cur_iter = 0

def get_rec_iter(args, kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    if 'benchmark' in args and args.benchmark:
        data_shape = (args.batch_size,) + image_shape
        train = SyntheticDataIter(args.num_classes, data_shape, 500, np.float32)
        return (train, None)
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    train = mx.io.ImageRecordIter(
        path_imgrec         = args.data_train,
        path_imgidx         = args.data_train_idx,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = image_shape,
        batch_size          = args.batch_size,
        rand_crop           = args.random_crop,
        max_random_scale    = args.max_random_scale,
        pad                 = args.pad_size,
        fill_value          = 127,
        resize              = args.resize_train,
        min_random_scale    = args.min_random_scale,
        max_aspect_ratio    = args.max_random_aspect_ratio,
        random_h            = args.max_random_h,
        random_s            = args.max_random_s,
        random_l            = args.max_random_l,
        max_rotate_angle    = args.max_random_rotate_angle,
        max_shear_ratio     = args.max_random_shear_ratio,
        rand_mirror         = args.random_mirror,
        preprocess_threads  = args.data_nthreads,
        shuffle             = True,
        num_parts           = nworker,
        part_index          = rank)
    if args.data_val is None:
        return (train, None)
    val = mx.io.ImageRecordIter(
        path_imgrec         = args.data_val,
        path_imgidx         = args.data_val_idx,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = image_shape,
        resize              = args.resize_val,
        preprocess_threads  = args.data_nthreads,
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = nworker,
        part_index          = rank)
    return (train, val)

class MultiIter:
    def __init__(self, iter_list,summary_writer= None):
        self.iters = iter_list
        self.label_list = ['gender_label','hat_label','umbrella_label','bag_label','handbag_label','backpack_label','updress_label','downdress_label','hatcolor_label','umbrellacolor_label','shoes_label']
        self.split_label = None
        self.summary_writer = summary_writer
    def next(self):
        batches = self.iters.next()
        #for idx,label in enumerate(self.label_list):
        #    print(batches.label[0].shape)
        split_label_reshape =[]
        split_label = mx.nd.split(batches.label[0], axis=1, num_outputs=len(self.label_list)) 
        
        split_label_reshape = [split.reshape([0,]) for split in split_label]
        #print(batches.data[0].shape)
        if self.summary_writer is not None:
            self.summary_writer.add_image('image',(batches.data[0]+123.68)/274.740997314)
            
        return DataBatch(data=batches.data,
                         label=split_label_reshape)
            
        #for y_slice in y:
        #    print(y_slice.shape)
        # split_label = [split_label,batches.label[:,idx]]
        
        #return DataBatch(data=batches.data,
        #                 label=split_label)
    def __iter__(self):
		return self
    
    def reset(self):
        #for i in self.iters:
        self.iters.reset()
    @property
    def provide_data(self):
        print(self.iters.provide_data)
        return self.iters.provide_data
    @property
    def provide_label(self):
        batchsize , label_width = (self.iters.provide_label[0].shape)
        print([i for i in self.label_list])
        return [mx.io.DataDesc(str(i), [batchsize,] , self.iters.provide_label[0].dtype) for i in self.label_list ]
    
        #return [i for i in self.label_list]


class MultiIter_Mixup:
    def __init__(self, iter_list,summary_writer= None):
        self.iters = iter_list
        self.label_list_origin =['gender_label','hat_label','umbrella_label','bag_label','handbag_label','backpack_label',
                                'updress_label','downdress_label','hatcolor_label','umbrellacolor_label','shoes_label']
        self.label_list = ['gender_label','hat_label','umbrella_label','bag_label','handbag_label',
                           'backpack_label','updress_label','downdress_label',
                           'hatcolor_label','umbrellacolor_label','shoes_label',
                           'gender_mix_label','hat_mix_label','umbrella_mix_label','bag_mix_label','handbag_mix_label',
                            'backpack_mix_label','updress_mix_label','downdress_mix_label',
                           'hatcolor_mix_label','umbrellacolor_mix_label','shoes_mix_label']
        self.split_label = None
        self.summary_writer = summary_writer



    def next(self):
        batches = self.iters.next()
        #for idx,label in enumerate(self.label_list):
        #    print(batches.label[0].shape)

        split_label_reshape =[]
        split_label = mx.nd.split(batches.label[0], axis=1, num_outputs=len(self.label_list_origin)) 

        batch_size = split_label[0].size
        #for split in split_label:
        index  = np.random.permutation(batch_size)


        lam =config.set_lam_value()

        mixed_x = batches.data[0] * lam+ (1.0 - lam) * batches.data[0][index, :]
        batches.data[0] = mixed_x


        split_label_reshape = [split.reshape([0,]) for split in split_label]
        split_label_reshape_mixup = [split.reshape([0,])[index] for split in split_label]
        split_label_reshape = split_label_reshape+ split_label_reshape_mixup

        #print(batches.data[0].shape)
        if self.summary_writer is not None:
            self.summary_writer.add_image('image',(batches.data[0]+123.68)/274.740997314)
            
        return DataBatch(data=batches.data,
                         label=split_label_reshape)
            
        #for y_slice in y:
        #    print(y_slice.shape)
        # split_label = [split_label,batches.label[:,idx]]
        
        #return DataBatch(data=batches.data,
        #                 label=split_label)
    def __iter__(self):
		return self
    
    def reset(self):
        #for i in self.iters:
        self.iters.reset()
    @property
    def provide_data(self):
        print(self.iters.provide_data)
        return self.iters.provide_data
    @property
    def provide_label(self):
        batchsize , label_width = (self.iters.provide_label[0].shape)
        print([i for i in self.label_list])
        return [mx.io.DataDesc(str(i), [batchsize,] , self.iters.provide_label[0].dtype) for i in self.label_list ]
    
        #return [i for i in self.label_list]
    

def get_rec_iter_mutil(args, kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    if 'benchmark' in args and args.benchmark:
        data_shape = (args.batch_size,) + image_shape
        train = SyntheticDataIter(args.num_classes, data_shape, 500, np.float32)
        return (train, None)
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    rgb_std = [float(i) for i in args.rgb_std.split(',')]
    train = mx.io.ImageRecordIter(
        path_imgrec         = args.data_train,
        path_imgidx         = args.data_train_idx,
        label_width         = 11,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        std_r               = rgb_std[0],
        std_g               = rgb_std[1],
        std_b               = rgb_std[2],
        data_name           = 'data',
        label_name          = ['gender_label','hat_label','umbrella_label','bag_label','handbag_label','backpack_label','updress_label','downdress_label','hatcolor_label','umbrellacolor_label','shoes_label'],
        data_shape          = image_shape,
        batch_size          = args.batch_size,
        rand_crop           = args.random_crop,
        max_random_scale    = args.max_random_scale,
        pad                 = args.pad_size,
        fill_value          = args.fill_value,
        random_resized_crop = args.random_resized_crop,
        min_random_scale    = args.min_random_scale,
        max_aspect_ratio    = args.max_random_aspect_ratio,
        min_aspect_ratio    = args.min_random_aspect_ratio,
        max_random_area     = args.max_random_area,
        min_random_area     = args.min_random_area,
        min_crop_size       = args.min_crop_size,
        max_crop_size       = args.max_crop_size,
        brightness          = args.brightness,
        contrast            = args.contrast,
        saturation          = args.saturation,
        pca_noise           = args.pca_noise,
        random_h            = args.max_random_h,
        random_s            = args.max_random_s,
        random_l            = args.max_random_l,
        max_rotate_angle    = args.max_random_rotate_angle,
        max_shear_ratio     = args.max_random_shear_ratio,
        rand_mirror         = args.random_mirror,
        preprocess_threads  = args.data_nthreads,
        shuffle             = True,
        num_parts           = nworker,
        part_index          = rank)
    #print(train.provide_label)
    if args.summary_writer_image:
        train = MultiIter(train,args.summary_writer_image)
    else:
        train = MultiIter(train)
        #train = MultiIter_Mixup(train)
        
    #print(train.provide_label)
    
    if args.data_val is None:
        return (train, None)
    
    val = mx.io.ImageRecordIter(
        path_imgrec         = args.data_val,
        path_imgidx         = args.data_val_idx,
        label_width         = 11,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        std_r               = rgb_std[0],
        std_g               = rgb_std[1],
        std_b               = rgb_std[2],
        data_name           = 'data',
        label_name          = ['gender_label','hat_label','umbrella_label','bag_label','handbag_label','backpack_label','updress_label','downdress_label','hatcolor_label','umbrellacolor_label','shoes_label'],
        batch_size          = args.batch_size,
        data_shape          = image_shape,
        preprocess_threads  = args.data_nthreads,
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = nworker,
        part_index          = rank)
    
    if args.summary_writer_image:
        val = MultiIter(val,args.summary_writer_image)
    else:
        val = MultiIter(val)
        
    return (train, val)

def get_rec_iter_mutil_mixup(args, kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    if 'benchmark' in args and args.benchmark:
        data_shape = (args.batch_size,) + image_shape
        train = SyntheticDataIter(args.num_classes, data_shape, 500, np.float32)
        return (train, None)
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    rgb_std = [float(i) for i in args.rgb_std.split(',')]
    train = mx.io.ImageRecordIter(
        path_imgrec         = args.data_train,
        path_imgidx         = args.data_train_idx,
        label_width         = 11,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        std_r               = rgb_std[0],
        std_g               = rgb_std[1],
        std_b               = rgb_std[2],
        data_name           = 'data',
        label_name          = ['gender_label','hat_label','umbrella_label','bag_label','handbag_label','backpack_label','updress_label','downdress_label','hatcolor_label','umbrellacolor_label','shoes_label'],
        data_shape          = image_shape,
        batch_size          = args.batch_size,
        rand_crop           = args.random_crop,
        max_random_scale    = args.max_random_scale,
        pad                 = args.pad_size,
        fill_value          = args.fill_value,
        random_resized_crop = args.random_resized_crop,
        min_random_scale    = args.min_random_scale,
        max_aspect_ratio    = args.max_random_aspect_ratio,
        min_aspect_ratio    = args.min_random_aspect_ratio,
        max_random_area     = args.max_random_area,
        min_random_area     = args.min_random_area,
        min_crop_size       = args.min_crop_size,
        max_crop_size       = args.max_crop_size,
        brightness          = args.brightness,
        contrast            = args.contrast,
        saturation          = args.saturation,
        pca_noise           = args.pca_noise,
        random_h            = args.max_random_h,
        random_s            = args.max_random_s,
        random_l            = args.max_random_l,
        max_rotate_angle    = args.max_random_rotate_angle,
        max_shear_ratio     = args.max_random_shear_ratio,
        rand_mirror         = args.random_mirror,
        preprocess_threads  = args.data_nthreads,
        shuffle             = True,
        num_parts           = nworker,
        part_index          = rank)
    #print(train.provide_label)
    if args.summary_writer_image:
        train = MultiIter_Mixup(train,args.summary_writer_image)
    else:
       # train = MultiIter(train)
        train = MultiIter_Mixup(train)
        
    #print(train.provide_label)
    
    if args.data_val is None:
        return (train, None)
    
    val = mx.io.ImageRecordIter(
        path_imgrec         = args.data_val,
        path_imgidx         = args.data_val_idx,
        label_width         = 11,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        std_r               = rgb_std[0],
        std_g               = rgb_std[1],
        std_b               = rgb_std[2],
        data_name           = 'data',
        label_name          = ['gender_label','hat_label','umbrella_label','bag_label','handbag_label','backpack_label','updress_label','downdress_label','hatcolor_label','umbrellacolor_label','shoes_label'],
        batch_size          = args.batch_size,
        data_shape          = image_shape,
        preprocess_threads  = args.data_nthreads,
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = nworker,
        part_index          = rank)
    
    if args.summary_writer_image:
       # val = MultiIter_Mixup(val,args.summary_writer_image)
        val = MultiIter(val, args.summary_writer_image)
    else:
        val = MultiIter(val)
        #val = MultiIter_Mixup(val)
        
    return (train, val)