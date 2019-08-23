import argparse
import os
import torch

parser = argparse.ArgumentParser(description='Cut trained model')
parser.add_argument('--trained_checkpoint', type=str, default=None, metavar='PTH.TAR', help='model(pth) path')
args = parser.parse_args()

if not os.path.exists(args.trained_checkpoint):
    raise OSError('%s does not exist!' % args.trained_checkpoint)

checkpoint = torch.load(args.trained_checkpoint)
simple_save = {
        'epoch': checkpoint['epoch'],
        'args': checkpoint['args'],
        'g_state_dict': checkpoint['g_state_dict'],
        'f1_state_dict': checkpoint['f1_state_dict'],
        'f2_state_dict': checkpoint['f2_state_dict'],
    }
checkpoint_fn = args.trained_checkpoint.split('.')[0] + '_simple.pth.tar'
torch.save(simple_save, checkpoint_fn)
