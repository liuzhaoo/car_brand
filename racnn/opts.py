import argparse
from pathlib import Path


def parse_opts():

    parser = argparse.ArgumentParser()


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
                        default='/home/zhaoliu/car_brand/racnn/results',
                        type=Path,
                        help='Result directory path')
    
    ####  model para and train para

    parser.add_argument('--num_classes',
                        default=1189,
                        type=int,
                        help='class number of full brand')

    parser.add_argument('--size',
                        default=224,
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

    parser.add_argument('--weight_decay',
                        default=0.0005,
                        type=float,
                        help='Weight Decay')

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
                        default=128,
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



    args = parser.parse_args()

    return args
