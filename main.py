
"""
"Fusion and Orthogonal Projection for Improved Face-Voice Association"
Muhammad Saad Saeed and Muhammad Haris Khan and Shah Nawaz and Muhammad Haroon Yousaf and Alessio Del Bue
ICASSP 2022
"""

from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import pandas as pd
# from scipy import random                                  # <<< CHANGE 1
import random                                               # <<< CHANGE 1
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')                                       # <<< CHANGE 10 (headless)
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import online_evaluation
from tqdm import tqdm

# In[0]

# STAGE7 = '/home/swapnilkhandoker/Bachelor_Thesis/project/utils/stage7_feature_extraction/mavceleb_baseline/features_v3'
STAGE7 = '/home/swapnilkhandoker/Bachelor_Thesis/project/utils/stage7_feature_extraction/mavceleb_baseline/features_v1x'

def read_data():
    # train_file = 'features/faceTrain.csv'                  # <<< CHANGE 2
    # train_file_voice = 'feaures/voiceTrain.csv'            # <<< CHANGE 2
    # Edit these two per run: swap English <-> Bangla
    train_file       = STAGE7 + '/faces/train/train_Bangla_faces.csv'    # <<< CHANGE 2
    train_file_voice = STAGE7 + '/voices/train/train_Bangla_voices.csv'  # <<< CHANGE 2
    # train_file       = STAGE7 + '/faces/train/train_Bangla_matched_faces.csv'
    # train_file_voice = STAGE7 + '/voices/train/train_Bangla_matched_voices.csv'

    print('Reading Train Faces')
    img_train = pd.read_csv(train_file, header=None)
    train_label = img_train[4096]
    img_train = np.asarray(img_train)
    img_train = img_train[:, 0:-1]
    train_label = np.asarray(train_label)
    print('Reading Voices')
    voice_train = pd.read_csv(train_file_voice, header=None)
    voice_train = np.asarray(voice_train)

    # <<< CHANGE 11: the face and voice CSVs each carry a class index in their
    # last column, written independently by the two extractors. If they ever
    # disagree, row i of one file is not the same training sample as row i of
    # the other and every training pair is mismatched. Cheap to check, and the
    # failure is otherwise invisible.
    voice_label = voice_train[:, -1].astype(np.int64)                      # <<< CHANGE 11
    n_bad = int(np.sum(voice_label != train_label.astype(np.int64)))       # <<< CHANGE 11
    if n_bad:                                                              # <<< CHANGE 11
        raise SystemExit('Label columns disagree on %d of %d rows -- the face '
                         'and voice CSVs are not row-aligned.'
                         % (n_bad, len(train_label)))
    print('Label columns agree on all %d rows' % len(train_label))         # <<< CHANGE 11

    voice_train = voice_train[:, 0:-1]

    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    train_label = le.transform(train_label)
    print("Train file length", len(img_train))

    print('Shuffling\n')
    combined = list(zip(img_train, voice_train, train_label))
    img_train = []
    voice_train = []
    train_label = []
    random.shuffle(combined)
    # img_train[:], voice_train, train_label[:] = zip(*combined)           # <<< CHANGE 12
    img_train[:], voice_train[:], train_label[:] = zip(*combined)          # <<< CHANGE 12
    combined = [] 
    # img_train = np.asarray(img_train).astype(np.float)
    # voice_train = np.asarray(voice_train).astype(np.float)
    img_train = np.asarray(img_train).astype(np.float32)        # <<< CHANGE 3
    voice_train = np.asarray(voice_train).astype(np.float32)
    train_label = np.asarray(train_label)
    
    
    return img_train, voice_train, train_label

face_train, voice_train, train_label = read_data()

face_test, voice_test = online_evaluation.read_data()


# In[1]

print('Training')
from retrieval_model import FOP

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
 
def get_batch(batch_index, batch_size, labels, f_lst):
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])

def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform(m.weight)              # <<< CHANGE 13 (deprecated spelling)
        torch.nn.init.xavier_uniform_(m.weight)               # <<< CHANGE 13
        m.bias.data.fill_(0.01)

def main(face_train, voice_train, train_label):
    
    # model = FOP(FLAGS, face_train.shape[1], voice_train.shape[1])
    n_class = int(np.max(train_label)) + 1                            # <<< CHANGE 6
    print('  + n_class: %d' % n_class)                                # <<< CHANGE 6
    model = FOP(FLAGS, face_train.shape[1], voice_train.shape[1], n_class)
    model.apply(init_weights)
    
    ce_loss = nn.CrossEntropyLoss().cuda()
    opl_loss = OrthogonalProjectionLoss().cuda()
    
    if FLAGS.cuda:
        model.cuda()
        ce_loss.cuda()    
        opl_loss.cuda()
        cudnn.benchmark = True
    
# =============================================================================
#     For Linear Fusion
# =============================================================================
    
    if FLAGS.fusion == 'linear':
    
        parameters = [
                      {'params' : model.face_branch.fc1.parameters()},
                      {'params' : model.voice_branch.fc1.parameters()},
                      {'params': model.logits_layer.parameters()},
                        {'params' : model.fusion_layer.weight1},
                        {'params' : model.fusion_layer.weight2}]
    
    
# =============================================================================
#     For Gated Fusion
# =============================================================================
    
    elif FLAGS.fusion == 'gated':
    
        parameters = [
                      {'params' : model.face_branch.fc1.parameters()},
                      {'params' : model.voice_branch.fc1.parameters()},
                      {'params': model.logits_layer.parameters()},
                      {'params' : model.fusion_layer.attention.parameters()}]

    optimizer = optim.Adam(parameters, lr=FLAGS.lr, weight_decay=0.01)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    os.makedirs('output', exist_ok=True)                              # <<< CHANGE 14
    results = []                                                      # <<< CHANGE 15

    for alpha in FLAGS.alpha_list:
        eer_list = []
        epoch=1
        num_of_batches = (len(train_label) // FLAGS.batch_size)
        loss_plot = []
        auc_list = []
        loss_per_epoch = 0
        # save_dir = '%s_%s_alpha_%0.2f'%(FLAGS.fusion, FLAGS.save_dir, alpha)
        save_dir = os.path.join(FLAGS.save_dir,                                   # <<< CHANGE 7
                                '%s_%s_alpha_%0.2f'%(FLAGS.fusion, FLAGS.tag, alpha))

        # txt = 'output/%s_ce_opl_%03d_%0.2f.txt'%(FLAGS.fusion, FLAGS.max_num_epoch, alpha)   # <<< CHANGE 14
        txt = 'output/%s_%s_ce_opl_%03d_%0.2f.txt'%(FLAGS.fusion, FLAGS.tag,      # <<< CHANGE 14
                                                    FLAGS.max_num_epoch, alpha)
        
        with open(txt,'w+') as f:
            # f.write('EPOCH\tLOSS\tEER\tAUC\n')                                  # <<< CHANGE 8
            f.write('EPOCH\tLOSS\tVAL_EER\tVAL_AUC\tBEST_EER\n')                  # <<< CHANGE 8
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # save_best = 'best_%s'%(save_dir)
        save_best = os.path.join(FLAGS.save_dir,                                  # <<< CHANGE 7
                                 'best_%s_%s_alpha_%0.2f'%(FLAGS.fusion, FLAGS.tag, alpha))
        
        # if not os.path.exists(save_best):                                       # <<< CHANGE 7
        #     os.mkdir(save_best)          # os.mkdir cannot create parent dirs
        os.makedirs(save_best, exist_ok=True)                                     # <<< CHANGE 7

        min_eer, max_auc = 1.0, 0.0                                               # <<< CHANGE 8
        best_eer = 1.0; best_epoch = 0; since_improved = 0                        # <<< CHANGE 8

        with open(txt,'a+') as f:
            # while (epoch < FLAGS.max_num_epoch):        # ran one epoch short   # <<< CHANGE 8
            while (epoch <= FLAGS.max_num_epoch):                                 # <<< CHANGE 8
                print('Epoch %03d'%(epoch))
                for idx in range(num_of_batches):
                    face_feats, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, face_train)
                    voice_feats, _ = get_batch(idx, FLAGS.batch_size, train_label, voice_train)
                    loss_tmp, loss_opl, loss_soft, _, _ = train(face_feats, voice_feats, 
                                                                 batch_labels, 
                                                                 model, optimizer, ce_loss, opl_loss, alpha)
                    loss_per_epoch+=loss_tmp
                loss_per_epoch = loss_per_epoch/num_of_batches
                loss_plot.append(loss_per_epoch)
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict()}, save_dir, 'checkpoint_%04d.pth.tar'%(epoch))
                print('==> Epoch: %d/%d Loss: %0.2f Alpha:%0.2f'%(epoch, FLAGS.max_num_epoch, loss_per_epoch, alpha))
                
                eer, auc = online_evaluation.test(FLAGS, model, face_test, voice_test)
                eer_list.append(eer)
                auc_list.append(auc)

                # <<< CHANGE 8: was  if eer <= min(eer_list):  -- eer_list already
                # contains the current value, so `<=` re-saved on every plateau.
                # A strict improvement of at least --min_delta now counts.
                # if eer <= min(eer_list):
                #     min_eer = eer
                #     max_auc = auc
                #     save_checkpoint({
                #     'epoch': epoch,
                #     'state_dict': model.state_dict()}, save_best, 'checkpoint_%04d.pth.tar'%(epoch))
                if eer < best_eer - FLAGS.min_delta:                              # <<< CHANGE 8
                    best_eer = eer; best_epoch = epoch; since_improved = 0
                    min_eer = eer
                    max_auc = auc
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'val_eer': eer, 'val_auc': auc,
                        'alpha': alpha, 'n_class': n_class,
                        'tag': FLAGS.tag},
                        save_best, 'checkpoint_best.pth.tar')
                else:                                                             # <<< CHANGE 8
                    since_improved += 1

                print('    val EER %.4f  val AUC %.4f  best %.4f @%d  patience %d/%d'
                      % (eer, auc, best_eer, best_epoch, since_improved, FLAGS.patience))

                epoch += 1
                # f.write('%04d\t%0.4f\t%0.2f\t%0.2f\n'%(epoch, loss_per_epoch, eer, auc))   # <<< CHANGE 8
                f.write('%04d\t%0.4f\t%0.4f\t%0.4f\t%0.4f\n'                      # <<< CHANGE 8
                        % (epoch - 1, loss_per_epoch, eer, auc, best_eer))
                f.flush()                                                         # <<< CHANGE 8
                loss_per_epoch = 0

                if since_improved >= FLAGS.patience:                              # <<< CHANGE 8
                    print('Early stop at epoch %d: no val improvement > %.4g for '
                          '%d epochs. Best val EER %.4f at epoch %d.'
                          % (epoch - 1, FLAGS.min_delta, FLAGS.patience,
                             best_eer, best_epoch))
                    break
        
        plt.figure(1)
        plt.title('Total Loss_%f'%(alpha))
        plt.plot(loss_plot)
        plt.savefig('output/%s_%s_%0.2f_total_loss.jpg'%(FLAGS.fusion, FLAGS.tag, alpha), dpi=800)
        plt.clf()                                                                 # <<< CHANGE 15
        
        plt.figure(2)
        plt.title('EER_%f'%(alpha))
        plt.plot(eer_list)
        plt.savefig('output/%s_%s_%0.2f_eer.jpg'%(FLAGS.fusion, FLAGS.tag, alpha), dpi=800)
        plt.clf()                                                                 # <<< CHANGE 15
        
        plt.figure(3)
        plt.title('AUC_%f'%(alpha))
        plt.plot(auc_list)
        plt.savefig('output/%s_%s_%0.2f_auc.jpg'%(FLAGS.fusion, FLAGS.tag, alpha), dpi=800)
        plt.clf()                                                                 # <<< CHANGE 15

        print('  best checkpoint: %s/checkpoint_best.pth.tar (epoch %d, val EER %.4f)'
              % (save_best, best_epoch, best_eer))
        results.append((alpha, best_eer, max_auc, best_epoch))                    # <<< CHANGE 15

        # <<< CHANGE 15: this `return` sat INSIDE the alpha loop, so only the
        # first alpha ever ran. Dedented to after the loop.
        # return loss_plot, min_eer, max_auc

    print('\n%s\nSUMMARY (%s)\n%s' % ('=' * 60, FLAGS.tag, '=' * 60))             # <<< CHANGE 15
    print('%-8s%-12s%-12s%s' % ('alpha', 'val_EER', 'val_AUC', 'best_epoch'))
    for a, e, u, ep in results:
        print('%-8.2f%-12.4f%-12.4f%d' % (a, e, u, ep))
    best = min(results, key=lambda r: r[1])
    print('\nLowest val EER: alpha %.2f, %.4f at epoch %d' % (best[0], best[1], best[3]))
    print('Select this alpha and checkpoint, THEN evaluate on test.')
    return loss_plot, best[1], best[2]                                            # <<< CHANGE 15
    
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self):
        super(OrthogonalProjectionLoss, self).__init__()
        self.device = (torch.device('cuda') if FLAGS.cuda else torch.device('cpu'))

    def forward(self, features, labels=None):
        
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]

        mask = torch.eq(labels, labels.t()).bool().to(self.device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(self.device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + (0.7 * neg_pairs_mean)

        return loss, pos_pairs_mean, neg_pairs_mean


def train(face_feats, voice_feats, labels, model, optimizer, ce_loss, opl_loss, alpha):
    
    average_loss = RunningAverage()
    soft_losses = RunningAverage()
    opl_losses = RunningAverage()

    model.train()
    face_feats = torch.from_numpy(face_feats).float()
    voice_feats = torch.from_numpy(voice_feats).float()
    labels = torch.from_numpy(labels)
    
    if FLAGS.cuda:
        face_feats, voice_feats, labels = face_feats.cuda(), voice_feats.cuda(), labels.cuda()

    face_feats, voice_feats, labels = Variable(face_feats), Variable(voice_feats), Variable(labels)
    comb, face_embeds, voice_embeds = model.train_forward(face_feats, voice_feats, labels)
    
    loss_opl, s_fac, d_fac = opl_loss(comb[0], labels)
    
    loss_soft = ce_loss(comb[1], labels)
    
    loss = loss_soft + alpha * loss_opl

    optimizer.zero_grad()
    
    loss.backward()
    average_loss.update(loss.item())
    opl_losses.update(loss_opl.item())
    soft_losses.update(loss_soft.item())
    
    optimizer.step()

    return average_loss.avg(), opl_losses.avg(), soft_losses.avg(), s_fac, d_fac

class RunningAverage(object):
    def __init__(self):
        self.value_sum = 0.
        self.num_items = 0. 

    def update(self, val):
        self.value_sum += val 
        self.num_items += 1

    def avg(self):
        average = 0.
        if self.num_items > 0:
            average = self.value_sum / self.num_items

        return average

def save_checkpoint(state, directory, filename):
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random Seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='CUDA Training')
    # parser.add_argument('--save_dir', type=str, default='model', ...)          # <<< CHANGE 7
    parser.add_argument('--save_dir', type=str, default='model_split_v1x',        # <<< CHANGE 7
                        help='Parent directory for all checkpoints.')
    parser.add_argument('--tag', type=str, default='english',                    # <<< CHANGE 7
                        help='Names the run; keeps english and bangla apart.')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-4)')
    # parser.add_argument('--batch_size', type=int, default=1, ...)              # <<< CHANGE 16
    parser.add_argument('--batch_size', type=int, default=128,                   # <<< CHANGE 16
                        help='Batch size. OPL needs several same-class rows per '
                             'batch to form its positive mask; 1 makes it a no-op.')
    parser.add_argument('--max_num_epoch', type=int, default=200, help='Max number of epochs to train, number')
    # parser.add_argument('--alpha_list', type=list, default=[1], ...)           # <<< CHANGE 9
    parser.add_argument('--alpha_list', type=str, default='2.0',                 # <<< CHANGE 9
                        help='Comma-separated, e.g. 0.0,1.0,2.0,5.0')
    parser.add_argument('--dim_embed', type=int, default=128,
                        help='Embedding Size')
    parser.add_argument('--fusion', type=str, default='gated', help='Fusion Type')
    parser.add_argument('--patience', type=int, default=20,                      # <<< CHANGE 8
                        help='Epochs of no val improvement before stopping.')
    parser.add_argument('--min_delta', type=float, default=0.0005,               # <<< CHANGE 8
                        help='EER gain below this counts as no improvement.')
    
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    # <<< CHANGE 9: type=list turned "2.0" into ['2', '.', '0']
    FLAGS.alpha_list = [float(a) for a in FLAGS.alpha_list.split(',') if a.strip()]
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()                        # <<< CHANGE 17
    torch.manual_seed(FLAGS.seed)
    random.seed(FLAGS.seed)                                                      # <<< CHANGE 17
    np.random.seed(FLAGS.seed)                                                   # <<< CHANGE 17
    if FLAGS.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
    print('[cfg] tag=%s save_dir=%s alphas=%s batch=%d patience=%d'
          % (FLAGS.tag, FLAGS.save_dir, FLAGS.alpha_list, FLAGS.batch_size, FLAGS.patience))
    loss_tmp, eer_tmp, auc_tmp = main(face_train, voice_train, train_label)










# """
# "Fusion and Orthogonal Projection for Improved Face-Voice Association"
# Muhammad Saad Saeed and Muhammad Haris Khan and Shah Nawaz and Muhammad Haroon Yousaf and Alessio Del Bue
# ICASSP 2022
# """

# from __future__ import division
# from __future__ import print_function

# import argparse
# import os

# import numpy as np
# import torch
# import torch.optim as optim
# import torch.utils.data
# from torch.autograd import Variable
# import torch.backends.cudnn as cudnn

# import pandas as pd
# # from scipy import random
# import random
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# import torch.nn as nn
# import online_evaluation
# from tqdm import tqdm

# # In[0]

# def read_data():
#     train_file = 'features/faceTrain.csv'
#     train_file_voice = 'feaures/voiceTrain.csv'
        
#     print('Reading Train Faces')
#     img_train = pd.read_csv(train_file, header=None)
#     train_label = img_train[4096]
#     img_train = np.asarray(img_train)
#     img_train = img_train[:, 0:-1]
#     train_label = np.asarray(train_label)
#     print('Reading Voices')
#     voice_train = pd.read_csv(train_file_voice, header=None)
#     voice_train = np.asarray(voice_train)
#     voice_train = voice_train[:, 0:-1]
    
#     le = preprocessing.LabelEncoder()
#     le.fit(train_label)
#     train_label = le.transform(train_label)
#     print("Train file length", len(img_train))
        
#     print('Shuffling\n')
#     combined = list(zip(img_train, voice_train, train_label))
#     img_train = []
#     voice_train = []
#     train_label = []
#     random.shuffle(combined)
#     img_train[:], voice_train, train_label[:] = zip(*combined)
#     combined = [] 
#     # img_train = np.asarray(img_train).astype(np.float)
#     # voice_train = np.asarray(voice_train).astype(np.float)
#     img_train = np.asarray(img_train).astype(np.float32)        # <<< CHANGE 3
#     voice_train = np.asarray(voice_train).astype(np.float32)
#     train_label = np.asarray(train_label)
    
    
#     return img_train, voice_train, train_label

# face_train, voice_train, train_label = read_data()

# face_test, voice_test = online_evaluation.read_data()


# # In[1]

# print('Training')
# from retrieval_model import FOP

# # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
 
# def get_batch(batch_index, batch_size, labels, f_lst):
#     start_ind = batch_index * batch_size
#     end_ind = (batch_index + 1) * batch_size
#     return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])

# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.xavier_uniform(m.weight)
#         m.bias.data.fill_(0.01)

# def main(face_train, voice_train, train_label):
    
#     # model = FOP(FLAGS, face_train.shape[1], voice_train.shape[1])
#     n_class = int(np.max(train_label)) + 1                            # <<< CHANGE 6
#     print('  + n_class: %d' % n_class)                                # <<< CHANGE 6
#     model = FOP(FLAGS, face_train.shape[1], voice_train.shape[1], n_class)
#     model.apply(init_weights)
    
#     ce_loss = nn.CrossEntropyLoss().cuda()
#     opl_loss = OrthogonalProjectionLoss().cuda()
    
#     if FLAGS.cuda:
#         model.cuda()
#         ce_loss.cuda()    
#         opl_loss.cuda()
#         cudnn.benchmark = True
    
# # =============================================================================
# #     For Linear Fusion
# # =============================================================================
    
#     if FLAGS.fusion == 'linear':
    
#         parameters = [
#                       {'params' : model.face_branch.fc1.parameters()},
#                       {'params' : model.voice_branch.fc1.parameters()},
#                       {'params': model.logits_layer.parameters()},
#                         {'params' : model.fusion_layer.weight1},
#                         {'params' : model.fusion_layer.weight2}]
    
    
# # =============================================================================
# #     For Gated Fusion
# # =============================================================================
    
#     elif FLAGS.fusion == 'gated':
    
#         parameters = [
#                       {'params' : model.face_branch.fc1.parameters()},
#                       {'params' : model.voice_branch.fc1.parameters()},
#                       {'params': model.logits_layer.parameters()},
#                       {'params' : model.fusion_layer.attention.parameters()}]

#     optimizer = optim.Adam(parameters, lr=FLAGS.lr, weight_decay=0.01)

#     n_parameters = sum([p.data.nelement() for p in model.parameters()])
#     print('  + Number of params: {}'.format(n_parameters))
    
#     for alpha in FLAGS.alpha_list:
#         eer_list = []
#         epoch=1
#         num_of_batches = (len(train_label) // FLAGS.batch_size)
#         loss_plot = []
#         auc_list = []
#         loss_per_epoch = 0
#         # save_dir = '%s_%s_alpha_%0.2f'%(FLAGS.fusion, FLAGS.save_dir, alpha)
#         save_dir = os.path.join(FLAGS.save_dir,                                   # <<< CHANGE 7
#                                 '%s_%s_alpha_%0.2f'%(FLAGS.fusion, FLAGS.tag, alpha))

#         txt = 'output/%s_ce_opl_%03d_%0.2f.txt'%(FLAGS.fusion, FLAGS.max_num_epoch, alpha)
        
#         with open(txt,'w+') as f:
#             f.write('EPOCH\tLOSS\tEER\tAUC\n')
        
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
        
#         # save_best = 'best_%s'%(save_dir)
#         save_best = os.path.join(FLAGS.save_dir,                                  # <<< CHANGE 7
#                                  'best_%s_%s_alpha_%0.2f'%(FLAGS.fusion, FLAGS.tag, alpha))
        
#         if not os.path.exists(save_best):
#             os.mkdir(save_best)
#         with open(txt,'a+') as f:
#             while (epoch < FLAGS.max_num_epoch):
#                 print('Epoch %03d'%(epoch))
#                 for idx in tqdm(range(num_of_batches)):
#                     face_feats, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, face_train)
#                     voice_feats, _ = get_batch(idx, FLAGS.batch_size, train_label, voice_train)
#                     loss_tmp, loss_opl, loss_soft, _, _ = train(face_feats, voice_feats, 
#                                                                  batch_labels, 
#                                                                  model, optimizer, ce_loss, opl_loss, alpha)
#                     loss_per_epoch+=loss_tmp
#                 loss_per_epoch = loss_per_epoch/num_of_batches
#                 loss_plot.append(loss_per_epoch)
#                 save_checkpoint({
#                     'epoch': epoch,
#                     'state_dict': model.state_dict()}, save_dir, 'checkpoint_%04d.pth.tar'%(epoch))
#                 print('==> Epoch: %d/%d Loss: %0.2f Alpha:%0.2f'%(epoch, FLAGS.max_num_epoch, loss_per_epoch, alpha))
                
#                 eer, auc = online_evaluation.test(FLAGS, model, face_test, voice_test)
#                 eer_list.append(eer)
#                 auc_list.append(auc)
#                 if eer <= min(eer_list):
#                     min_eer = eer
#                     max_auc = auc
#                     save_checkpoint({
#                     'epoch': epoch,
#                     'state_dict': model.state_dict()}, save_best, 'checkpoint_%04d.pth.tar'%(epoch))

#                 epoch += 1
#                 f.write('%04d\t%0.4f\t%0.2f\t%0.2f\n'%(epoch, loss_per_epoch, eer, auc))
#                 loss_per_epoch = 0
        
#         plt.figure(1)
#         plt.title('Total Loss_%f'%(alpha))
#         plt.plot(loss_plot)
#         plt.savefig('output/%s_%0.2f_total_loss.jpg'%(FLAGS.fusion, alpha), dpi=800)
        
#         plt.figure(2)
#         plt.title('EER_%f'%(alpha))
#         plt.plot(eer_list)
#         plt.savefig('output/%s_%0.2f_eer.jpg'%(FLAGS.fusion, alpha), dpi=800)
        
#         plt.figure(3)
#         plt.title('AUC_%f'%(alpha))
#         plt.plot(auc_list)
#         plt.savefig('output/%s_%0.2f_auc.jpg'%(FLAGS.fusion, alpha), dpi=800)
                
#         return loss_plot, min_eer, max_auc
    
# class OrthogonalProjectionLoss(nn.Module):
#     def __init__(self):
#         super(OrthogonalProjectionLoss, self).__init__()
#         self.device = (torch.device('cuda') if FLAGS.cuda else torch.device('cpu'))

#     def forward(self, features, labels=None):
        
#         features = F.normalize(features, p=2, dim=1)

#         labels = labels[:, None]

#         mask = torch.eq(labels, labels.t()).bool().to(self.device)
#         eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(self.device)

#         mask_pos = mask.masked_fill(eye, 0).float()
#         mask_neg = (~mask).float()
#         dot_prod = torch.matmul(features, features.t())

#         pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
#         neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

#         loss = (1.0 - pos_pairs_mean) + (0.7 * neg_pairs_mean)

#         return loss, pos_pairs_mean, neg_pairs_mean


# def train(face_feats, voice_feats, labels, model, optimizer, ce_loss, opl_loss, alpha):
    
#     average_loss = RunningAverage()
#     soft_losses = RunningAverage()
#     opl_losses = RunningAverage()

#     model.train()
#     face_feats = torch.from_numpy(face_feats).float()
#     voice_feats = torch.from_numpy(voice_feats).float()
#     labels = torch.from_numpy(labels)
    
#     if FLAGS.cuda:
#         face_feats, voice_feats, labels = face_feats.cuda(), voice_feats.cuda(), labels.cuda()

#     face_feats, voice_feats, labels = Variable(face_feats), Variable(voice_feats), Variable(labels)
#     comb, face_embeds, voice_embeds = model.train_forward(face_feats, voice_feats, labels)
    
#     loss_opl, s_fac, d_fac = opl_loss(comb[0], labels)
    
#     loss_soft = ce_loss(comb[1], labels)
    
#     loss = loss_soft + alpha * loss_opl

#     optimizer.zero_grad()
    
#     loss.backward()
#     average_loss.update(loss.item())
#     opl_losses.update(loss_opl.item())
#     soft_losses.update(loss_soft.item())
    
#     optimizer.step()

#     return average_loss.avg(), opl_losses.avg(), soft_losses.avg(), s_fac, d_fac

# class RunningAverage(object):
#     def __init__(self):
#         self.value_sum = 0.
#         self.num_items = 0. 

#     def update(self, val):
#         self.value_sum += val 
#         self.num_items += 1

#     def avg(self):
#         average = 0.
#         if self.num_items > 0:
#             average = self.value_sum / self.num_items

#         return average

# def save_checkpoint(state, directory, filename):
#     filename = os.path.join(directory, filename)
#     torch.save(state, filename)
    
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random Seed')
#     parser.add_argument('--cuda', action='store_true', default=True, help='CUDA Training')
#     parser.add_argument('--save_dir', type=str, default='model', help='Directory for saving checkpoints.')
#     parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
#                         help='learning rate (default: 1e-4)')
#     parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
#     parser.add_argument('--max_num_epoch', type=int, default=5, help='Max number of epochs to train, number')
#     parser.add_argument('--alpha_list', type=list, default=[1], help='Alpha Values List')
#     parser.add_argument('--dim_embed', type=int, default=128,
#                         help='Embedding Size')
#     parser.add_argument('--fusion', type=str, default='gated', help='Fusion Type')
    
#     global FLAGS
#     FLAGS, unparsed = parser.parse_known_args()
#     torch.manual_seed(FLAGS.seed)
#     if FLAGS.cuda and torch.cuda.is_available():
#         torch.cuda.manual_seed(FLAGS.seed)
#     loss_tmp, eer_tmp, auc_tmp = main(face_train, voice_train, train_label)
