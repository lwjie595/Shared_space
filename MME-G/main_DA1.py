from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase,Predictor_deep_gate,Predictor_deep_gate_softmax,Predictor_deep_gate_relu,\
    Predictor_deep_gate_tanh,Predictor_deep_gate_clip
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from utils.loss import entropy, adentropy,WGAN_loss,GAN_loss
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--Analysis', type=str, default='',
                    help='select analysis parameters for Gate')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='MME_Gate',
                    choices=['S+T', 'ENT', 'MME','MME_Gate','MME_Gate_relu','MME_Gate_clip','MME_Gate_softmax','MME_Gate_tanh'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model/save_model_DA2_FeaGate',
                    help='dir to save checkpoint')
parser.add_argument('--checkpath_step', type=str, default='./save_model/save_model_4090',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34',
                    help='which network to use')
parser.add_argument('--inc', type=int, default=512,
                    help='which network to use')
parser.add_argument('--source', type=str, default='painting',
                    help='source domain')
parser.add_argument('--target', type=str, default='real',
                    help='target domain')

parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--bs', type=int, default=24,
                    help='batch_size')

parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=20, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=False,
                    help='early stopping on validation or not')
parser.add_argument('--GAN_lambda',type=float,default=0.1)

parser.add_argument('--flag',type=int,default=4)
# parser.add_argument('--')

args = parser.parse_args()
print('Dataset %s Source %s  Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))
source_loader, source_loader2,target_loader, target_loader_unl,target_loader_val, source_loader_val, \
    target_loader_test, class_list = return_dataset(args)
use_gpu = torch.cuda.is_available()
record_dir = 'record_DA2_FeaGate/%s/%s/%s_num_%s_%d' % (args.Analysis,args.dataset, args.method,args.num,args.bs)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)


torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    G = resnet34()
    inc = args.inc
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr':args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

# if "resnet" in args.net:
if args.method=='MME_Gate':
    F1 = Predictor_deep_gate(num_class=len(class_list),
                        inc=inc)
elif  args.method=='MME_Gate_relu':
    F1 = Predictor_deep_gate_relu(num_class=len(class_list),
                        inc=inc)
elif  args.method=='MME_Gate_clip':
    F1 = Predictor_deep_gate_clip(num_class=len(class_list),
                        inc=inc)
elif  args.method=='MME_Gate_softmax':
    F1 = Predictor_deep_gate_softmax(num_class=len(class_list),
                        inc=inc)
else:
    F1 = Predictor_deep_gate_tanh(num_class=len(class_list),
                        inc=inc)

weights_init(F1)
lr = args.lr
G.cuda()
F1.cuda()


im_data_s = torch.FloatTensor(1)
im_data_s2 = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_s2 = torch.LongTensor(1)
im_data_t=torch.FloatTensor(1)
gt_labels_t=torch.LongTensor(1)
im_data_tu = torch.FloatTensor(1)
# sample_labels_t = torch.LongTensor(1)
# sample_labels_s = torch.LongTensor(1)

im_data_s = im_data_s.cuda()
im_data_s2 = im_data_s2.cuda()
gt_labels_s = gt_labels_s.cuda()
gt_labels_s2= gt_labels_s2.cuda()
im_data_t=im_data_t.cuda()
gt_labels_t=gt_labels_t.cuda()
im_data_tu = im_data_tu.cuda()
# sample_labels_t = sample_labels_t.cuda()
# sample_labels_s = sample_labels_s.cuda()

im_data_s = Variable(im_data_s)
im_data_s2= Variable(im_data_s2)
gt_labels_s = Variable(gt_labels_s)
gt_labels_s2 = Variable(gt_labels_s2)
im_data_t=Variable(im_data_t)
gt_labels_t=Variable(gt_labels_t)
im_data_tu = Variable(im_data_tu)
# sample_labels_t = Variable(sample_labels_t)
# sample_labels_s = Variable(sample_labels_s)

if os.path.exists(args.checkpath) == False:
    os.makedirs(args.checkpath)


def train(flag):
    record_file = os.path.join(record_dir,
                               '%s_test1_%s_%s_to_%s_num_%s_test.txt' %
                               (args.method, args.net, args.source,
                                args.target,args.num))

    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    criterion = nn.CrossEntropyLoss().cuda()
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t_unl=iter(target_loader_unl)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)
    # len_test_target = len(target_loader_test)
    counter = 0
    best_acc_test=0
    with open(record_file, 'a') as f:
            f.write(
                '################  {} Feature Gate  Source {} to {} bs {} for learning rate of {},lr: g :{:.5f}, f:{:.5f},with lambda {}, inc {}######################## \n'.format(
                    args.method, args.source, args.target, args.bs, args.lr,param_lr_g[0], param_lr_f[0],  args.lamda,args.inc))  # Discriminator,dc is leakyRelu



    for step in range (all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)

        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)

        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)

        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        data_unl=next(data_iter_t_unl)
        im_data_s.data=im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])
        gt_labels_s.data=gt_labels_s.data.resize_(data_s[1].size()).copy_(data_s[1])
        im_data_t.data=im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
        # im_data_t.data = im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
        gt_labels_t.data=gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
        im_data_tu.data=im_data_tu.data.resize_(data_unl[0].size()).copy_(data_unl[0])

        zero_grad_all()
        data = torch.cat((im_data_s, im_data_t), 0)
        label = torch.cat((gt_labels_s, gt_labels_t), 0)
        if flag==1:
            output_G =  output_s
            out_all = F1(output_G[:,0,:])
            loss_cls=criterion(out_all, label)
            output_s=output_s[:,1,:]
            output_s2 = output_s2[:, 1, :]
        else:
            output_G = G(data)
            out_all = F1(output_G)
            loss_cls = criterion(out_all, label)
        # out_target =F1(G(im_data_t))
        # loss_target = criterion(out_target, torch.max(out_target, dim=1)[1])

        loss=loss_cls

        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()



        if not args.method == 'S+T':
            output = G(im_data_tu)
            if args.method == 'ENT':
                loss_t = entropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            elif args.method == 'MME':
                loss_t = adentropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            else:
                loss_t = adentropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            # else:
            #     raise ValueError('Method cannot be recognized.')
            log_train = 'S {} T {} Train Ep: {} lr{:.6f} \t ' \
                        'Loss Classification: {:.6f} Loss T {:.6f} ' \
                        'Method {}\n'.format(args.source, args.target,
                                             step, lr, loss.data,
                                             -loss_t.data, args.method)
        else:
            log_train = 'S {} T {} Train Ep: {} lr{:.6f} \t ' \
                        'Loss Classification: {:.6f} Method {}\n'.\
                format(args.source, args.target,
                       step, lr, loss.data,
                       args.method)
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:

            # loss_test, acc_test = test(target_loader_val,flag)
            loss_test, acc_test = test(target_loader_test, flag)
            #
            loss_train,acc_train=test(source_loader_val,flag)

            # adv_loss = GAN_loss(Dc, output_s, output_s2,1.0,criterion_GAN,zero=False)
            if acc_test >= best_acc_test:
                # best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1


            print('best acc test %f now test %f  train source acc %f ' % (best_acc_test,acc_test,
                                                        acc_train))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d cls loss %.3f target all best %.3f now target loss %.5f acc %.3f train source loss %.5f acc  %.3f  \n' % (step,loss_cls,
                                                         best_acc_test,loss_test ,acc_test,
                                                         loss_train,acc_train))

            #



            G.train()
            F1.train()
            if args.early:
                if counter > args.patience:
                    break


        if step%10000==9999:
            if args.save_check:
                save_path=args.checkpath_step
                if os.path.exists(save_path) == False:
                    os.makedirs(save_path)
                torch.save(G.state_dict(),
                           os.path.join(save_path,
                                        "G_iter_model_{}_{}_to_{}"
                                        "_step_{}.pth.tar".
                                        format(args.method, args.source,args.target,
                                               step)))
                torch.save(F1.state_dict(),
                           os.path.join(save_path,
                                        "F1_iter_model_{}_{}_to_{}"
                                        "_step_{}.pth.tar".
                                        format(args.method, args.source,args.target
                                               ,step)))
    if args.save_check:
        loss_test, acc_test = test(target_loader_test,flag)
        #
        loss_train, acc_train = test(source_loader,flag)
        print('Final  now test %f  train source acc %f  ' % (acc_test,
                                                    acc_train))
        print('record %s' % record_file)
        with open(record_file, 'a') as f:
            f.write('step Final all data %d target loss %.5f acc %.3f train source loss %.5f acc  %.3f  \n' % (step,
                                                    loss_test ,acc_test,
                                                    loss_train,acc_train))
        if flag==4:
            path_special="Predictor_C"
            args.checkpath=args.checkpath
        else:
            path_special="Feature_C"
            args.checkpath = './save_model/save_model_DA2_FeaGate'
        print('saving model')
        torch.save(G.state_dict(),
                   os.path.join(args.checkpath,
                                "G_iter_model_{}_{}"
                                "spe_{}.pth.tar".
                                format(args.method, args.source,
                                       path_special)))
        torch.save(F1.state_dict(),
                   os.path.join(args.checkpath,
                                "F1_iter_model_{}_{}"
                                "spe_{}.pth.tar".
                                format(args.method, args.source,
                                       path_special)))






def test(loader,flag):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.data=im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.data=gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.0f}%)\n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size



train(flag=4)

