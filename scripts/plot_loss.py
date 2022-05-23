import numpy as np
import torch
print("finish import")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
#Run3 setup:
model_names = ['nominal_with_etarel_phirel', 'adversarial_with_etarel_phirel']
'''
model_names = ['nominal', 'adversarial_eps0p01']
tagger = 'DF_Run2' # 'DF'
dirz = [f'/eos/user/a/anstein/DeepJet/Train_{tagger}/{model_name}/' \
        for model_name in model_names]

nominal_epochs = 39
adversarial_epochs = 73

paths = {
    'nominal' : [dirz[0] + f'checkpoint_epoch_{i}.pth' for i in range(1,nominal_epochs+1)],
    'adversarial' : [dirz[1] + f'checkpoint_epoch_{i}.pth' for i in range(1,adversarial_epochs+1)]
    }

for p,key in enumerate(paths):
    
    tr_losses = []
    val_losses = []
        
    for i in range(len(paths[key])):
        checkpoint = torch.load(paths[key][i], map_location=torch.device(device))
        tr_losses.append(checkpoint['train_loss'])
        val_losses.append(checkpoint['val_loss'])
    
    np.save(dirz[p] + f'loss_trainval.npy',np.array([tr_losses,val_losses]))