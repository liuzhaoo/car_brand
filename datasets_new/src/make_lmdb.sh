#!/bin/bash
# convert images to lmdb

# DATA=/home/meringue/DataBase/cifar-10-batches-py
IMGDIRNAME=/
# IMGLIST=/home/zhaoliu/car_data/训练数据/4.9新加测试集/val.txt
# LMDBNAME=/home/zhaoliu/car_data/训练数据/4.9新加测试集/valval
IMGLIST=/home/zhaoliu/car_brand/lmdb_data_new/youzhan_test/val.txt
LMDBNAME=/mnt/disk/zhaoliu_data/small_car_lmdb/youzhan_test




rm -rf $LMDBNAME
echo 'converting images...'
/home/zhaoliu/caffe/build/tools/convert_imageset --shuffle=true --resize_height=256 --resize_width=256 \
$IMGDIRNAME $IMGLIST $LMDBNAME