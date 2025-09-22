import os
import argparse
import torch
import torch.nn as nn
from model.resnet import resnet34
from model.basenet import Predictor_deep_gate
import torch.functional as F
import torch.optim as optim
parser=argparse.ArgumentParser()
parser.add_argument('--value',type=int,default=1)

args=parser.parse_args()
t_start=args.value
print('now it is start of %d :'%(t_start))
for i in range(t_start,t_start+10):
    print(' %d'%(i))

print('\n')

G=resnet34().cuda()
Fcls=Predictor_deep_gate(num_class=10,inc=512).cuda()

imgs=torch.randn((64,3,224,224)).cuda()
labels=torch.randint(0,10,(64,)).cuda()
creiterion=nn.CrossEntropyLoss().cuda()
optimizer_g=optim.SGD(list(G.parameters()),momentum=0.9,lr=0.1,weight_decay=0.005,nesterov=True)
optimizer_f=optim.SGD(list(Fcls.parameters()),momentum=0.9,lr=0.1,weight_decay=0.005,nesterov=True)
G.train()
Fcls.train()
for i in range(100):
    out=G(imgs)
    pred=Fcls(out)
    loss=creiterion(pred,labels)
    loss.backward()
    optimizer_f.step()
    optimizer_g.step()

print('finish')

