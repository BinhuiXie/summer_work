import argparse
import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from datasets import get_dataset
from model_util import get_models
from transform import Scale
from util import mkdir_if_not_exist
from logger import Logger
import warnings
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.deterministic = True
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Image Segmentation Task Tester (Pytorch Implementation)')
parser.add_argument('--gpu', type=str, nargs='?', default='0', help='device id to run')
parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--test_list', type=str, default='data/test.txt', help='the test path list')
parser.add_argument('--test_img_shape', default=(600, 800), nargs=2, help='W H, FOR test(600, 800)')
parser.add_argument('--trained_checkpoint', type=str, default=None, metavar='PTH.TAR', help='model(pth) path')
parser.add_argument('--output_path', type=str, default='snapshot/test/', help='save file path')
parser.add_argument('--log_file', type=str, default='shoe_adapt_seg_source(train_val)_target(test)', help='log name')
parser.add_argument('--saves_prob', action='store_true', help='whether you save probability tensors')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config = {'output_path': args.output_path,
          'path': {'log': args.output_path + 'log/',
                   'prob': args.output_path + 'prob/',
                   'label': args.output_path + 'label/'}
          }

mkdir_if_not_exist(config['path']['log'])
mkdir_if_not_exist(config['path']['label'])

config['logger'] = Logger(logroot=config['path']['log'], filename=args.log_file, level='debug')
config['logger'].logger.debug(str(args))

if not os.path.exists(args.trained_checkpoint):
    raise OSError('%s does not exist!' % args.trained_checkpoint)
config['logger'].logger.debug('==> loading checkpoint: ' + args.trained_checkpoint)
checkpoint = torch.load(args.trained_checkpoint)
train_args = checkpoint['args']
G, F1, F2 = get_models(input_ch=train_args.input_ch, n_class=train_args.n_class, res=train_args.res,
                       is_data_parallel=train_args.is_data_parallel)
G.load_state_dict(checkpoint['g_state_dict'])
F1.load_state_dict(checkpoint['f1_state_dict'])
F2.load_state_dict(checkpoint['f2_state_dict'])
config['logger'].logger.debug('==> loaded checkpoint:' + args.trained_checkpoint +
                              ' epoch: ' + str(checkpoint['epoch']))


train_img_shape = tuple([int(x) for x in train_args.train_img_shape])
test_img_shape = tuple([int(x) for x in args.test_img_shape])

img_transform = Compose([
    Scale(train_img_shape, Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])

test_dataset = get_dataset(dataset_name='test', img_lists=args.test_list, label_lists=None,
                           img_transform=img_transform, label_transform=None, test=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

G.eval()
F1.eval()
F2.eval()
if torch.cuda.is_available():
    G.cuda()
    F1.cuda()
    F2.cuda()

for index, (imgs, _, paths) in tqdm(enumerate(test_loader)):
    path = paths[0]

    # import torchvision.utils as vutils
    # vutils.save_image(imgs, 'target_shoe.png', normalize=True)

    if torch.cuda.is_available():
        imgs = imgs.cuda()

    feature = G(imgs)
    outputs = F1(feature)
    outputs += F2(feature)

    if args.saves_prob:
        # Save probability tensors
        mkdir_if_not_exist(config['path']['prob'])
        prob_outfn = os.path.join(config['path']['prob'], path.split('/')[-1].replace('jpg', 'npy'))
        np.save(prob_outfn, outputs[0].data.cpu().numpy())

    # Save predicted pixel labels(jpg)
    pred = outputs[0, :train_args.n_class].data.max(0)[1].cpu()
    # shoe --> white  background --> black
    pred[pred != 0] = 255
    img = Image.fromarray(np.uint8(pred.numpy()))
    img = img.resize(test_img_shape, Image.NEAREST)
    label_outdir = config['path']['label']
    if index == 0:
        config['logger'].logger.debug('pred label dir: ' + label_outdir)
    label_fn = os.path.join(label_outdir, path.split('/')[-1])
    img.save(label_fn)
