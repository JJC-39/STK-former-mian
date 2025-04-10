import argparse
import os
import torch
import model
from util import util

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    def initialize(self, parser):
        # base define
        parser.add_argument('--name', type=str,  help='name of the experiment.')
        parser.add_argument('--model', type=str,  help='name of the model type. ')
        parser.add_argument('--mask_type', type=int,   help='mask type, 0: center mask, 1:random regular mask, ''2: random irregular mask. 3: external irregular mask. [0],[1,2],[1,2,3]')
        parser.add_argument('--checkpoints_dir', type=str,  help='models are save here')
        parser.add_argument('--which_iter', type=str,  help='which iterations to load')
        parser.add_argument('--gpu_ids', type=str, help='gpu ids: e.g. 0, 1, 2 use -1 for CPU')
        parser.add_argument('--img_file', type=str,help='training and testing dataset')
        parser.add_argument('--mask_file', type=str,  help='load test mask')
        parser.add_argument('--loadSize', type=int,  help='scale images to this size')
        parser.add_argument('--fineSize', type=int,  help='then crop to this size')
        parser.add_argument('--resize_or_crop', type=str, help='scaling and cropping of images at load time [resize_and_crop|crop|]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the image for data augmentation')
        parser.add_argument('--no_rotation', action='store_true', help='if specified, do not rotation for data augmentation')
        parser.add_argument('--no_augment', action='store_true', help='if specified, do not augment the image for data augmentation')
        parser.add_argument('--batchSize', type=int,  help='input batch size')
        parser.add_argument('--nThreads', type=int,  help='# threads for loading data')
        parser.add_argument('--no_shuffle', action='store_true',help='if true, takes images serial')
        # display parameter define
        parser.add_argument('--display_winsize', type=int,help='display window size')
        parser.add_argument('--display_id', type=int, help='display id of the web')
        parser.add_argument('--display_port', type=int, help='visidom port of the web display')
        parser.add_argument('--display_single_pane_ncols', type=int, help='if positive, display all images in a single visidom web panel')

        return parser

    def gather_options(self):
        """Add additional model-specific options"""

        if not self.initialized:
            parser = self.initialize(self.parser)

        # get basic options
        opt, _ = parser.parse_known_args()

        # modify the options for different models
        model_option_set = model.get_option_setter(opt.model)
        parser = model_option_set(parser, self.isTrain)
        opt = parser.parse_args()

        return opt

    def parse(self):
        """Parse the options"""

        opt = self.gather_options()
        opt.isTrain = self.isTrain

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids):
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt

        return self.opt

    @staticmethod
    def print_options(opt):
        """print and save options"""

        print('--------------Options--------------')
        for k, v in sorted(vars(opt).items()):
            print('%s: %s' % (str(k), str(v)))
        print('----------------End----------------')

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        if opt.isTrain:
            file_name = os.path.join(expr_dir, 'train_opt.txt')
        else:
            file_name = os.path.join(expr_dir, 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('--------------Options--------------\n')
            for k, v in sorted(vars(opt).items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('----------------End----------------\n')

class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, help='# of the test examples')
        parser.add_argument('--results_dir', type=str,  help='saves results here')
        parser.add_argument('--how_many', type=int, help='how many test images to run')
        parser.add_argument('--phase', type=str, help='train, val, test')
        parser.add_argument('--nsampling', type=int, help='ramplimg # times for each images')
        parser.add_argument('--save_number', type=int, help='choice # reasonable results based on the discriminator score')
        self.isTrain = False
        return parser
class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--topk', type=int, help='initial value of topk in Sparse Attention')
        # training epoch
        parser.add_argument('--iter_count', type=int, help='the starting epoch count')
        parser.add_argument('--niter', type=int,  help='# of iter with initial learning rate')
        parser.add_argument('--niter_decay', type=int,  help='# of iter to decay learning rate to zero')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        # learning rate and loss weight
        parser.add_argument('--lr_policy', type=str,  help='learning rate policy[lambda|step|plateau]')
        parser.add_argument('--lr', type=float,  help='initial learning rate for adam')
        parser.add_argument('--beta1', type=float, help='initial learning rate for adam')
        parser.add_argument('--beta2', type=float, help='initial learning rate for adam')
        parser.add_argument('--weight_decay', type=float, help='weight decay')
        parser.add_argument('--gan_mode', type=str,  choices=['wgan-gp', 'hinge', 'lsgan'])        # display the results
        parser.add_argument('--display_freq', type=int,  help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int,  help='frequency of saving the latest results')
        parser.add_argument('--save_iters_freq', type=int, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results')
        self.isTrain = True
        return parser
