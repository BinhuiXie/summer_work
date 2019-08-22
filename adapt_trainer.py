from __future__ import division

import os
import argparse
import torch
import tqdm
from PIL import Image
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
from datasets import ConcatDataset, get_dataset
from loss import CrossEntropyLoss2d, DiscrepancyLoss2d
from model_util import get_models, get_optimizer
from transform import ReLabel, ToLabel, Scale
from util import adjust_learning_rate, seed_everything, mkdir_if_not_exist
from logger import Logger
from tensorboardX import SummaryWriter
import warnings
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.deterministic = True
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Image Segmentation Task Trainer (Pytorch Implementation)')
parser.add_argument('--seed', type=int, default=1, help='manual seed')
parser.add_argument('--gpu', type=str, nargs='?', default='1', help='device id to run')
parser.add_argument('--res', type=str, default='50', metavar='ResnetLayerNum',
                    choices=['18', '34', '50', '101', '152'], help='which resnet 18,50,101,152')
parser.add_argument('--is_data_parallel', action='store_true', help='whether you use torch.nn.DataParallel')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
parser.add_argument('--opt', type=str, default='sgd', choices=['sgd', 'adam', 'adadelta'], help='network optimizer')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--adjust_lr', action='store_true', help='whether you change lr')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum sgd (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=2e-5, help='weight_decay (default: 2e-5)')
parser.add_argument('--source_list', type=str, default='data/source.txt', help='the source path list')
parser.add_argument('--source_label_list', type=str, default='data/source_label.txt', help='the source label path list')
parser.add_argument('--target_list', type=str, default='data/target.txt', help='the target path list')
parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--train_img_shape', default=(600, 800), nargs=2, metavar=("W", "H"), help="W H")
parser.add_argument('--input_ch', type=int, default=3, choices=[1, 3, 4])
parser.add_argument('--n_class', type=int, default=2, help='types of class ')
parser.add_argument("--resume", type=str, default=None, metavar="PTH.TAR", help="model(pth) path")
parser.add_argument('--num_k', type=int, default=4, help='how many steps to repeat the generator update')
parser.add_argument('--b_weight', type=float, default=1.0, help='weight for background class loss')
parser.add_argument('--s_weight', type=float, default=1.0, help='weight for shoe class loss')
parser.add_argument('--output_path', type=str, default='snapshot/', help='save file path')
parser.add_argument('--log_file', type=str, default='shoe_adapt_seg_source(train_val)_target(test)', help='log file')
parser.add_argument('--is_writer', action='store_true', help='whether you use SummaryWriter or not')

args = parser.parse_args()
seed_everything(args.seed)
os.environ['PYTHONASHSEED'] = str(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config = {'output_path': args.output_path,
          'path': {'log': args.output_path + '/log/',
                   'scalar': args.output_path + '/scalar/',
                   'model': args.output_path + '/model/'},
          'is_writer': args.is_writer}

# Create output Dir
mkdir_if_not_exist(config['path']['log'])
mkdir_if_not_exist(config['path']['scalar'])
mkdir_if_not_exist(config['path']['model'])
if config['is_writer']:
    config['writer'] = SummaryWriter(log_dir=config['path']['scalar'])
config['logger'] = Logger(logroot=config['path']['log'], filename=args.log_file, level='debug')
config['logger'].logger.debug(str(args))

# whether resume training
start_epoch = 0
if args.resume:
    config['logger'].logger.debug('==> loading checkpoint: ' + args.resume)
    if not os.path.exists(args.resume):
        raise OSError("%s does not exist!" % args.resume)

    # load model and args
    checkpoint = torch.load(args.resume)
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']

    G, F1, F2 = get_models(input_ch=args.input_ch, n_class=args.n_class, res=args.res,
                           is_data_parallel=args.is_data_parallel)
    optimizer_g = get_optimizer(G.parameters(), lr=args.lr, momentum=args.momentum, opt=args.opt,
                                weight_decay=args.weight_decay)
    optimizer_f = get_optimizer(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, momentum=args.momentum,
                                opt=args.opt, weight_decay=args.weight_decay)
    G.load_state_dict(checkpoint['g_state_dict'])
    F1.load_state_dict(checkpoint['f1_state_dict'])
    F2.load_state_dict(checkpoint['f2_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_f.load_state_dict(checkpoint['optimizer_f'])
    config['logger'].logger.debug('==> loaded checkpoint: ' + args.resume)
else:
    G, F1, F2 = get_models(input_ch=args.input_ch, n_class=args.n_class, res=args.res,
                           is_data_parallel=args.is_data_parallel)
    optimizer_g = get_optimizer(G.parameters(), lr=args.lr, momentum=args.momentum, opt=args.opt,
                                weight_decay=args.weight_decay)
    optimizer_f = get_optimizer(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, momentum=args.momentum,
                                opt=args.opt, weight_decay=args.weight_decay)

# load image
train_img_shape = tuple([int(x) for x in args.train_img_shape])
img_transform = Compose([
    Scale(train_img_shape, Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225])])
label_transform = Compose([
    Scale(train_img_shape, Image.NEAREST),
    ToLabel(),
    ReLabel(255, args.n_class - 1),  # convert label
])

source_dataset = get_dataset(dataset_name='source', img_lists=args.source_list, label_lists=args.source_label_list,
                             img_transform=img_transform, label_transform=label_transform, test=False)
target_dataset = get_dataset(dataset_name='target', img_lists=args.target_list, label_lists=None,
                             img_transform=img_transform, label_transform=None, test=False)

train_loader = torch.utils.data.DataLoader(
    ConcatDataset(
        source_dataset,
        target_dataset),
    batch_size=args.batch_size, shuffle=True,
    pin_memory=True)

# start training
# background weight: 1  shoe weight: 1
class_weighted = torch.Tensor([args.b_weight, args.s_weight])
if torch.cuda.is_available():
    G.cuda()
    F1.cuda()
    F2.cuda()
    class_weighted = class_weighted.cuda()
G.train()
F1.train()
F2.train()

criterion_c = CrossEntropyLoss2d(class_weighted)
criterion_d = DiscrepancyLoss2d()

for epoch in range(start_epoch, args.epochs):
    d_loss_per_epoch = 0
    c_loss_per_epoch = 0

    for ind, (source, target) in tqdm.tqdm(enumerate(train_loader)):
        source_img, source_labels = source[0], source[1]
        target_img = target[0]

        # import torchvision.utils as vutils
        # vutils.save_image(source_img, 'train_source_shoe.png', normalize=True)
        # vutils.save_image(target_img, 'train_target_shoe.png', normalize=True)
        # vutils.save_image(source_labels, 'train_source_label_shoe.png', normalize=True)

        if torch.cuda.is_available():
            source_img, source_labels, target_img = source_img.cuda(), source_labels.cuda(), target_img.cuda()

        # 1 step: minimize the source class loss
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        outputs = G(source_img)
        outputs1 = F1(outputs)
        outputs2 = F2(outputs)
        loss = criterion_c(outputs1, source_labels)
        loss += criterion_c(outputs2, source_labels)
        loss.backward()
        c_loss = loss.item()
        c_loss_per_epoch += c_loss
        optimizer_g.step()
        optimizer_f.step()

        # 2 step: maximum the two classifier's discrepancy loss and minimize the source class loss
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        outputs = G(source_img)
        outputs1 = F1(outputs)
        outputs2 = F2(outputs)
        loss = criterion_c(outputs1, source_labels)
        loss += criterion_c(outputs2, source_labels)
        outputs = G(target_img)
        outputs1 = F1(outputs)
        outputs2 = F2(outputs)
        loss -= criterion_d(outputs1, outputs2)
        loss.backward()
        optimizer_f.step()

        # minimize the two classifier's discrepancy loss
        for i in range(args.num_k):
            optimizer_g.zero_grad()
            outputs = G(target_img)
            outputs1 = F1(outputs)
            outputs2 = F2(outputs)
            loss = criterion_d(outputs1, outputs2)
            loss.backward()
            optimizer_g.step()

        d_loss = loss.item() / args.num_k
        d_loss_per_epoch += d_loss
        if ind % 20 == 0:
            config['logger'].logger.debug("Epoch [%d] iter [%d] DLoss: %.6f CLoss: %.6f" % (epoch, ind, d_loss, c_loss))

    config['logger'].logger.debug("Epoch [%d] DLoss: %.6f CLoss: %.6f" % (epoch, d_loss_per_epoch, c_loss_per_epoch))
    if config['is_writer']:
        config['writer'].add_scalars('train', {'DLoss': d_loss_per_epoch,
                                               'CLoss': c_loss_per_epoch},
                                     epoch)
    if args.adjust_lr:
        args.lr = adjust_learning_rate(optimizer_g, args.lr, args.weight_decay, epoch, args.epochs)
        args.lr = adjust_learning_rate(optimizer_f, args.lr, args.weight_decay, epoch, args.epochs)

    save_dic = {
        'epoch': epoch + 1,
        'args': args,
        'g_state_dict': G.state_dict(),
        'f1_state_dict': F1.state_dict(),
        'f2_state_dict': F2.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_f': optimizer_f.state_dict(),
    }
    checkpoint_fn = config['path']['model'] + 'fcn-res%s-%s.pth.tar' % (args.res, epoch + 1)
    torch.save(save_dic, checkpoint_fn)
