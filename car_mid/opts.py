import argparse
from pathlib import Path


def parse_opts():

    parser = argparse.ArgumentParser()

    ### path ###
# /mnt/disk/zhaoliu_data/carlogo/lmdb/carlogo_test/allnewdata_img_test_lmdb
# /mnt/disk/zhaoliu_data/carlogo/lmdb/carlogo_test/car_256_test_lmdb
# /mnt/disk/zhaoliu_data/carlogo/lmdb/carlogo_train_new/car_256_all_lmdb
# /home/zhaoliu/car_class/dataset/qczj_train_all_lmdb.npy
    parser.add_argument('--weight_txt',
                        default='/home/zhaoliu/car_brand/lmdb_data/weight.txt',
                        type=str,
                        help='path of train data weight')

    parser.add_argument('--label_npy',
                        default='/home/zhaoliu/car_brand/datasets_new/tongji/train_tongji.npy',
                        type=str,
                        help='path of train data')
    parser.add_argument('--val_txt',
                        default='/home/zhaoliu/car_brand/datasets/val.txt',
                        type=str,
                        help='path of val data')

    parser.add_argument('--train_data',
                        default="/mnt/disk/zhaoliu_data/small_car_lmdb/train_val_lmdb",
                        type=str,
                        help='path of train data')
    parser.add_argument('--keys_path_train',
                        default='/home/zhaoliu/car_brand/lmdb_data_new/train.npy',                  
                        type=str,
                        help='path of train data')
    parser.add_argument('--val_data',
                        default="/mnt/disk/zhaoliu_data/small_car_lmdb/train_val_lmdb",
                        type=str,
                        help='path of val data')
    parser.add_argument('--keys_path_val',
                        default='/home/zhaoliu/car_brand/lmdb_data_new/val.npy',
                        type=str,
                        help='path of val data')

    parser.add_argument('--result_path',
                        default='/home/zhaoliu/car_brand/car_mid/results_newdata/rotation',
                        type=Path,
                        help='Result directory path')
    
    ####  model para and train para

    parser.add_argument('--fb_cls',
                        default=1189,
                        type=int,
                        help='class number of full brand')
    # parser.add_argument('--model_cls',
    #                     default=21,
    #                     type=int,
    #                     help='class number of model(车型数量)')
    parser.add_argument('--size',
                        default=240,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--learning_rate',
                        default=0.01,
                        type=float,
                        help=('Initial learning rate'
                              '(divided by 10 while training by lr scheduler)'))
    parser.add_argument('--momentum', 
                        default=0.9,
                        type=float, 
                        help='Momentum')
    parser.add_argument('--dampening',
                        default=0.0,
                        type=float,
                        help='dampening of SGD')
    parser.add_argument('--weight_decay',
                        default=0.0005,
                        type=float,
                        help='Weight Decay')

    parser.add_argument('--nesterov',
                        action='store_true',
                        help='Nesterov momentum')
    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        help='sgd|adam')
    parser.add_argument('--lr_scheduler',
                        default='multistep',
                        type=str,
                        help='Type of LR scheduler (multistep | plateau)')
    parser.add_argument(
                        '--multistep_milestones',
                        default=[20, 30, 40],
                        type=int,
                        nargs='+',
                        help='Milestones of LR scheduler. See documentation of MultistepLR.')
    parser.add_argument('--plateau_patience',
                        default=10,
                        type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--batch_size',
                        default=512,
                        type=int,
                        help='Batch Size')
    parser.add_argument('--n_epochs',
                        default=50,
                        type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--n_threads',
                        default=16,
                        type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint',
                        default=10,
                        type=int,
                        help='Trained model is saved at every this epochs.')
    parser.add_argument('--model',
                        default='resnet18',
                        type=str,
                        help='inceptionv4 | resnet18')

    parser.add_argument('--manual_seed',
                        default=1,
                        type=int,
                        help='Manually set random seed')

    parser.add_argument('--value_scale',
                        default=1,
                        type=int,
                        help='If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')

    parser.add_argument('--no_train',
                        action='store_true',
                        help='If true, training is not performed.')
    parser.add_argument('--no_val',
                        # default=True,
                        action='store_true',
                        help='If true, validation is not performed.')

    ###  distributed parallel ###
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--tensorboard',
                        action='store_true',
                        help='If true, output tensorboard log file.')
    parser.add_argument('--distributed',
                        default=False,
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs.')
    parser.add_argument('--batchnorm_sync',
                        default=False,
                        help='If true, SyncBatchNorm is used instead of BatchNorm.')
    parser.add_argument('--dist_url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')

    

    args = parser.parse_args()

    return args
