import argparse
import torch
from codes import mvtecad
from functools import reduce
from torch.utils.data import DataLoader
from codes.datasets import *
from codes.networks import *
from codes.inspection import eval_encoder_NN_multiK
from codes.utils import *
from tqdm.auto import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

current_time = datetime.datetime.now() + datetime.timedelta(hours= 9)
current_time = current_time.strftime('%Y-%m-%d-%H:%M')

saved_loc = os.path.join('/content/drive/MyDrive/Patch_SVDD_Result', current_time)
if os.path.exists(saved_loc):
    shutil.rmtree(saved_loc)
os.mkdir(saved_loc)

print("결과 저장 위치: ", saved_loc)

writer = SummaryWriter(saved_loc)


parser = argparse.ArgumentParser()

parser.add_argument('--obj', default='hazelnut', type=str)
parser.add_argument('--lambda_value', default=1, type=float)
parser.add_argument('--D', default=64, type=int)

parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=1e-4, type=float)

args = parser.parse_args()


def train():
    obj = args.obj
    D = args.D
    lr = args.lr
        
    with task('Networks'):
        enc = EncoderHier(64, D).cuda()
        cls_64 = PositionClassifier(64, D).cuda()
        cls_32 = PositionClassifier(32, D).cuda()

        modules = [enc, cls_64, cls_32]
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        opt = torch.optim.Adam(params=params, lr=lr)

    with task('Datasets'):
        train_x = mvtecad.get_x_standardized(obj, mode='train')
        train_x = NHWC2NCHW(train_x)

        rep = 100
        datasets = dict()
        datasets[f'pos_64'] = PositionDataset(train_x, K=64, repeat=rep)
        datasets[f'pos_32'] = PositionDataset(train_x, K=32, repeat=rep)
        
        datasets[f'svdd_64'] = SVDD_Dataset(train_x, K=64, repeat=rep)
        datasets[f'svdd_32'] = SVDD_Dataset(train_x, K=32, repeat=rep)

        dataset = DictionaryConcatDataset(datasets)
        loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

    print('Start training!')
    for i_epoch in tqdm(range(args.epochs)):
        if i_epoch != 0:
            for module in modules:
                module.train()

            num_instance = 0
            total_lo = 0

            for d in loader:
                # count the number of patches
                x, _, _ = d['pos_64']
                num_instance += x.shape[0]

                x1, _, _ = d['pos_32']
                print('64 size: ', x.shape)
                print('32 size: ', x1.shape)


                d = to_device(d, 'cuda', non_blocking=True)
                opt.zero_grad()

                loss_pos_64 = PositionClassifier.infer(cls_64, enc, d['pos_64'])
                loss_pos_32 = PositionClassifier.infer(cls_32, enc.enc, d['pos_32'])
                loss_svdd_64 = SVDD_Dataset.infer(enc, d['svdd_64'])
                loss_svdd_32 = SVDD_Dataset.infer(enc.enc, d['svdd_32'])

                loss = loss_pos_64 + loss_pos_32 + args.lambda_value * (loss_svdd_64 + loss_svdd_32)

                total_lo += loss.item()

                loss.backward()
                opt.step()

            writer.add_scalar('Train/Training_Loss', total_lo / num_instance, i_epoch)


        # evaluation process is very heavy, so pass this process.
        #aurocs = eval_encoder_NN_multiK(enc, obj)
        #log_result(obj, aurocs)
        enc.save(obj, args.epochs, args.lambda_value, D)

    writer.close()


def log_result(obj, aurocs):
    det_64 = aurocs['det_64'] * 100
    seg_64 = aurocs['seg_64'] * 100

    det_32 = aurocs['det_32'] * 100
    seg_32 = aurocs['seg_32'] * 100

    det_sum = aurocs['det_sum'] * 100
    seg_sum = aurocs['seg_sum'] * 100

    det_mult = aurocs['det_mult'] * 100
    seg_mult = aurocs['seg_mult'] * 100

    print(f'|K64| Det: {det_64:4.1f} Seg: {seg_64:4.1f} |K32| Det: {det_32:4.1f} Seg: {seg_32:4.1f} |mult| Det: {det_sum:4.1f} Seg: {seg_sum:4.1f} |mult| Det: {det_mult:4.1f} Seg: {seg_mult:4.1f} ({obj})')


if __name__ == '__main__':
    train()
