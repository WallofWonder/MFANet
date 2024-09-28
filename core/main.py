# Code is heavily borrowed from https://github.com/zjhthu/OANet
# Author: Shihua Zhang
# Date: 2022/11/21
# E-mail: suhzhang001@gmail.com


from config import get_config, print_usage
# config, unparsed = get_config()
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
import torch
import torch.distributed as dist
import torch.utils.data
import sys
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(1024)


from data import collate_fn, CorrespondencesDataset
from mfanet import MFANet as Model
from train import train


print("-------------------------Deep Essential-------------------------")
print("Note: To combine datasets, use .")

def create_log_dir(config):
    if not os.path.isdir(config.log_base):
        os.makedirs(config.log_base)
    if config.log_suffix == "":
        suffix = "-".join(sys.argv)
    else:
        suffix = config.log_suffix
    if not os.path.isdir(config.log_base):
        os.makedirs(config.log_base)
    result_path = config.log_base + suffix
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if not os.path.isdir(result_path+'/train'):
        os.makedirs(result_path+'/train')
    if not os.path.isdir(result_path+'/valid'):
        os.makedirs(result_path+'/valid')
    if os.path.exists(result_path+'/config.th'):
        print('warning: will overwrite config file')
    torch.save(config, result_path+'/config.th')
    # path for saving traning logs
    config.log_path = result_path#+'/train'

def main(config):
    """The main function."""

    # Initialize network
    model = Model(config)
    # model = torch.nn.DataParallel(model)

    # Run propper mode
    create_log_dir(config)

    # initialize ddp
    torch.cuda.set_device(config.local_rank)
    device = torch.device(f'cuda:{config.local_rank}')
    model.to(device)
    dist.init_process_group(backend='nccl', init_method='env://')
    sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # convert bn to sync_bn
    model = torch.nn.parallel.DistributedDataParallel(sync_bn_module, device_ids=[config.local_rank])

    train_dataset = CorrespondencesDataset(config.data_tr, config)

    # train_loader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=config.train_batch_size, shuffle=True,
    #         num_workers=16, pin_memory=False, collate_fn=collate_fn)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.train_batch_size // torch.distributed.get_world_size(),
                                               num_workers=16 // dist.get_world_size(), pin_memory=False,
                                               sampler=train_sampler, collate_fn=collate_fn)

    valid_dataset = CorrespondencesDataset(config.data_va, config)
    # valid_loader = torch.utils.data.DataLoader(
    #         valid_dataset, batch_size=config.train_batch_size, shuffle=False,
    #         num_workers=8, pin_memory=False, collate_fn=collate_fn)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=config.train_batch_size // torch.distributed.get_world_size(),
                                               num_workers=8 // dist.get_world_size(), pin_memory=False,
                                               collate_fn=collate_fn, sampler=valid_sampler)
    #valid_loader = None
    print('start training .....')
    print('save to {}'.format(config.log_path))
    train(model, train_loader, valid_loader, config)


if __name__ == "__main__":
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print(unparsed)
        print_usage()
        exit(1)
    print(config.att_mode)
    main(config)
    print("finished")

#
# main.py ends here
