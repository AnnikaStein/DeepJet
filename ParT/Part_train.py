import torch 
import torch.nn as nn
from pytorch_Part import training_base
from ParT import ParticleTransformer
#from pytorch_deepjet_transformer_v4 import DeepJetTransformer, ParticleTransformer
#from ParT_old import DeepJetTransformer, ParticleTransformer
from pytorch_ranger import *
#from schedulers import *

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

num_epochs = 30

lr_epochs = max(1, int(num_epochs * 0.3))
lr_rate = 0.01 ** (1.0 / lr_epochs)
mil = list(range(num_epochs - lr_epochs, num_epochs))
print(lr_rate)
print(mil)

model = ParticleTransformer(num_classes = 6, num_enc = 3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#model = nn.DataParallel(model)
scaler = torch.cuda.amp.GradScaler()

criterion = cross_entropy_one_hot
optimizer = Ranger(model.parameters(), lr = 1e-3)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [1], gamma = 0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = mil, gamma = lr_rate)
#scheduler = CosineAnealingWarmRestartsWeightDecay(optimizer, T_0=14714, T_mul=1, eta_min=5e-6, last_epoch=-1, gamma = 0.9, max_lr = 5e-4)

train=training_base(model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, scaler = scaler, testrun=False)

train.train_data.maxFilesOpen=1

model,history = train.trainModel(nepochs=num_epochs, 
                                 batchsize=512)
