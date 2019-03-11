#!/bin/bash
/usr/bin/python /workspace/mnt/group/video-det/zhangfeiyun/projects/person/person_attribute_pure/prepare_data/create_list.py
/usr/bin/python /workspace/mnt/group/video-det/zhangfeiyun/projects/person/person_attribute_pure/prepare_data/create_list.py --dataset val --json attr_label_val.json

/usr/bin/python /workspace/mnt/group/video-det/zhangfeiyun/projects/person/person_attribute_pure/prepare_data/im2rec.py --resize 128 --pack-label /workspace/mnt/group/video-det/zhangfeiyun/data/person_attribute_data/attribute_duke_M_P_peta_old_train /workspace/mnt/group/video-det/zhangfeiyun/data/person_attribute_data/
/usr/bin/python /workspace/mnt/group/video-det/zhangfeiyun/projects/person/person_attribute_pure/prepare_data/im2rec.py --resize 128 --pack-label /workspace/mnt/group/video-det/zhangfeiyun/data/person_attribute_data/attribute_duke_M_P_peta_old_val /workspace/mnt/group/video-det/zhangfeiyun/data/person_attribute_data/

