from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


## to call it from cammand lines
import sys
import os
from argparse import ArgumentParser
import shutil
from DeepJetCore.DataCollection import DataCollection
#from DeepJetCore.dataPipeline import TrainDataGenerator
#from DeepJetCore.compiledc_trainDataGenerator import trainDataGenerator
from DeepJetCore.DJCLosses import *
from DeepJetCore.DJCLayers import *
from pdb import set_trace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from tqdm import tqdm
import copy

import imp

from attacks_ParT import *
from definitions_ParT import epsilons_per_feature, vars_per_candidate

glob_vars = vars_per_candidate['glob']

def train_loop(dataloader, nbatches, model, loss_fn, optimizer, device, epoch, epoch_pbar, acc_loss, scaler, scheduler, attack, att_magnitude, restrict_impact, epsilon_factors):
    print('Type of train_loop [Nominal(None)|Adversarial()]:', attack)
    for b in range(nbatches):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #should not happen unless files are broken (will give additional errors)
        #if dataloader.isEmpty():
        #    raise Exception("ran out of data") 
            
        features_list, truth_list = next(dataloader)

        glob = torch.Tensor(features_list[0]).to(device)
        cpf = torch.Tensor(features_list[1]).to(device)
        npf = torch.Tensor(features_list[2]).to(device)
        vtx = torch.Tensor(features_list[3]).to(device)
        cpf_4v = torch.Tensor(features_list[4]).to(device)
        cpf_4v = cpf_4v[:,:,:4]
        npf_4v = torch.Tensor(features_list[5]).to(device)
        vtx_4v = torch.Tensor(features_list[6]).to(device)
        #pair = torch.Tensor(features_list[7]).to(device)
        #print(pair[0,:28,:])
        y = torch.Tensor(truth_list[0]).to(device)
        
        glob[:,:] = torch.where(glob[:,:] == -999., torch.zeros(len(glob),glob_vars).to(device), glob[:,:])
        glob[:,:] = torch.where(glob[:,:] ==   -1., torch.zeros(len(glob),glob_vars).to(device), glob[:,:])
        
        # apply attack
        #print('Attack type:',attack)
        if attack == 'Noise':
            #print('Do Noise')
            #glob = apply_noise(glob, 
            #                   magn=att_magnitude,
            #                   offset=[0],
            #                   dev=device,
            #                   restrict_impact=restrict_impact,
            #                   var_group='glob')
            cpf = apply_noise(cpf, 
                               magn=att_magnitude,
                               offset=[0],
                               dev=device,
                               restrict_impact=restrict_impact,
                               var_group='cpf')
            npf = apply_noise(npf, 
                               magn=att_magnitude,
                               offset=[0],
                               dev=device,
                               restrict_impact=restrict_impact,
                               var_group='npf')
            vtx = apply_noise(vtx, 
                               magn=att_magnitude,
                               offset=[0],
                               dev=device,
                               restrict_impact=restrict_impact,
                               var_group='vtx')
            cpf_4v = apply_noise(cpf_4v, 
                               magn=att_magnitude,
                               offset=[0],
                               dev=device,
                               restrict_impact=restrict_impact,
                               var_group='cpf_pts')
            npf_4v = apply_noise(npf_4v, 
                               magn=att_magnitude,
                               offset=[0],
                               dev=device,
                               restrict_impact=restrict_impact,
                               var_group='npf_pts')
            vtx_4v = apply_noise(vtx_4v, 
                               magn=att_magnitude,
                               offset=[0],
                               dev=device,
                               restrict_impact=restrict_impact,
                               var_group='vtx_pts')

        elif attack == 'FGSM' or attack == 'NGM':
            #print('Do FGSM')
            with torch.cuda.amp.autocast():
                glob, cpf, npf, vtx, cpf_4v, npf_4v, vtx_4v = first_order_attack(sample=(glob,cpf,npf,vtx,cpf_4v,npf_4v,vtx_4v), 
                                                                          epsilon=att_magnitude,
                                                                          dev=device,
                                                                          targets=y,
                                                                          thismodel=model,
                                                                          thiscriterion=loss_fn,
                                                                          restrict_impact=restrict_impact,
                                                                          epsilon_factors=epsilon_factors,
                                                                          do_sign_or_normed_grad = attack)
            
        # Compute prediction and loss
        inpt = (cpf, npf, vtx, cpf_4v, npf_4v, vtx_4v)
        #ncpf = int(torch.max(glob[:,2]))
        #nnpf = int(torch.max(glob[:,3]))
        #nvtx = int(torch.max(glob[:,4]))
        # Backpropagation
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = model(inpt)
            loss = loss_fn(pred, y.type_as(pred)) # batch loss
        scaler.scale(loss).backward()
        #scaler.unscale_(optimizer)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        #loss.backward()
        #optimizer.step()
        #scheduler.step()
        acc_loss += loss.detach().item() # batch loss, is a mean over individual losses in batch
        # Update progress bar description
        avg_loss = acc_loss / (b + 1) # per batch, the avg_loss is updated
        # batch 0 -> batch0_loss / 1 = batch0_loss
        # batch 1 -> (batch0_loss + batch1_loss)/2
        # batch 2 -> (batch0_loss + batch1_loss + batch2_loss) / 3
        #if((b % 100) == 0):
        #    print(f"Training Error: \n batch : {(b):>0.1f} / {(nbatches):>0.1f}, Avg loss: {avg_loss:>6f} \n")
        desc = f'Epoch {epoch+1} - loss {avg_loss:.6f}'
        epoch_pbar.set_description(desc)
        epoch_pbar.update(1)
        
    return avg_loss

def val_loop(dataloader, nbatches, model, loss_fn, device, epoch):
    num_batches = nbatches
    test_loss, correct, total = 0, 0, 0
 
    with torch.no_grad():
        for b in range(nbatches):
        #should not happen unless files are broken (will give additional errors)
            #if dataloader.isEmpty():
            #    raise Exception("ran out of data") 

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            features_list, truth_list = next(dataloader)
            glob = torch.Tensor(features_list[0]).to(device)
            cpf = torch.Tensor(features_list[1]).to(device)
            npf = torch.Tensor(features_list[2]).to(device)
            vtx = torch.Tensor(features_list[3]).to(device)
            cpf_4v = torch.Tensor(features_list[4]).to(device)
            cpf_4v = cpf_4v[:,:,:4]
            npf_4v = torch.Tensor(features_list[5]).to(device)
            vtx_4v = torch.Tensor(features_list[6]).to(device)
            #pair = torch.Tensor(features_list[7]).to(device)
            y = torch.Tensor(truth_list[0]).to(device)    
            # Compute prediction and loss
            inpt = (cpf, npf, vtx, cpf_4v, npf_4v, vtx_4v)
            pred = model(inpt)
            # Compute prediction and loss
            _, labels = y.max(dim=1)
            #print(labels)
            #print(y)
            #total += cpf[0].shape[0] # this does not have the length of one batch, but rather the number of cpfs !
            total += cpf.shape[0]
            test_loss += loss_fn(pred, y.long()).item() # batch loss, added for every batch
            avg_loss = test_loss / (b + 1)
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
            #print(correct)
            #print(total)
 
    #test_loss /= num_batches # will be total loss over all batches, divided by number of batches
    #correct /= (num_batches * cpf[0].shape[0])
    correct /= total # this is dividing by the total length of validation set, also works for the edge case of the last batch
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.6f}%, Avg loss: {test_loss:>6f} \n")
    print(f"Test Error: \n Accuracy: {(100*correct):>0.6f}%, Avg loss: {avg_loss:>6f} \n")
    return avg_loss

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

class training_base(object):
    
    def __init__(self, model = None, criterion = cross_entropy_one_hot, optimizer = None,
                scheduler = None, scaler = None, splittrainandtest=0.85, useweights=False, 
                 testrun=False, testrun_fraction=0.1, resumeSilently=False, renewtokens=True,
		 collection_class=DataCollection, parser=None, recreate_silently=False):
        
        import sys
        scriptname=sys.argv[0]
        
        parser = ArgumentParser('Run the training')
        parser.add_argument('inputDataCollection')
        parser.add_argument('outputDir')
        parser.add_argument("--submitbatch",  help="submits the job to condor" , default=False, action="store_true")
        parser.add_argument("--walltime",  help="sets the wall time for the batch job, format: 1d5h or 2d or 3h etc" , default='1d')
        parser.add_argument("--isbatchrun",   help="is batch run", default=False, action="store_true")


        args = parser.parse_args()
    
        self.inputData = os.path.abspath(args.inputDataCollection)
        self.outputDir = args.outputDir
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.trainedepoches = 0
        self.best_loss = np.inf
        self.checkpoint = 0
    
        isNewTraining=True
        if os.path.isdir(self.outputDir):
            if not (resumeSilently or recreate_silently):
                var = input('output dir exists. To recover a training, please type "yes"\n')
                var = 'yes'
                if not var == 'yes':
                    raise Exception('output directory must not exists yet')
                isNewTraining=False
                if recreate_silently:
                    isNewTraining=True     
        else:
            os.mkdir(self.outputDir)
        self.outputDir = os.path.abspath(self.outputDir)
        self.outputDir+='/'
        
        if recreate_silently:
            os.system('rm -rf '+ self.outputDir +'*')
        
        #copy configuration to output dir
        if not args.isbatchrun:
            try:
                shutil.copyfile(scriptname,self.outputDir+os.path.basename(scriptname))
            except shutil.SameFileError:
                pass
            except BaseException as e:
                raise e
                
            self.copied_script = self.outputDir+os.path.basename(scriptname)
        else:
            self.copied_script = scriptname
        
        self.train_data = DataCollection()
        self.train_data.readFromFile(self.inputData)
        self.train_data.useweights=useweights

        self.val_data = self.train_data.split(splittrainandtest)
        
        #shapes = self.train_data.getNumpyFeatureShapes()
        #inputdtypes = self.train_data.getNumpyFeatureDTypes()
        #inputnames= self.train_data.getNumpyFeatureArrayNames()
        #for i in range(len(inputnames)): #in case they are not named
         #   if inputnames[i]=="" or inputnames[i]=="_rowsplits":
          #      inputnames[i]="input_"+str(i)+inputnames[i]
            
        #print("shapes", shapes)
        #print("inputdtypes", inputdtypes)
        #print("inputnames", inputnames)
        
        #self.torch_inputsshapes=[]
        #counter=0
        #for s,dt,n in zip(shapes,inputdtypes,inputnames):
        #    self.torch_inputsshapes.append(s)
            
        if not isNewTraining:
            if os.path.isfile(self.outputDir+'/checkpoint.pth'):
                kfile = self.outputDir+'/checkpoint.pth' 
            if os.path.isfile(kfile):
                print(kfile)

                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                self.checkpoint = torch.load(kfile)
                self.trainedepoches = self.checkpoint['epoch']
                self.best_loss = self.checkpoint['best_loss']
                
                self.model.load_state_dict(self.checkpoint['state_dict'])
                self.model.to(self.device)
                self.optimizer.load_state_dict(self.checkpoint['optimizer'])
                #self.optimizer.to(self.device)
                self.scheduler.load_state_dict(self.checkpoint['scheduler'])
            
            else:
                print('no model found in existing output dir, starting training from scratch')

    #def __del__(self):
     #   if hasattr(self, 'train_data'):
      #      del self.train_data
       #     del self.val_data
            
    def saveModel(self,model, optimizer, epoch, scheduler, best_loss, train_loss, val_loss, is_best = False):
        checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict(),'epoch': epoch,'scheduler': scheduler.state_dict(),'best_loss': best_loss,'train_loss': train_loss,'val_loss': val_loss}
        if is_best:
            torch.save(checkpoint, self.outputDir+'checkpoint_best_loss.pth')
        else:
            torch.save(checkpoint, self.outputDir+'checkpoint.pth')
            torch.save(checkpoint, self.outputDir+'checkpoint_epoch_'+str(epoch)+'.pth')
        
    def _initTraining(self, batchsize, use_sum_of_squares=False):
        
        #if self.submitbatch:
         #   from DeepJetCore.training.batchTools import submit_batch
          #  submit_batch(self, self.args.walltime)
           # exit() #don't delete this!
        
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        self.train_data.batch_uses_sum_of_squares=use_sum_of_squares
        self.val_data.batch_uses_sum_of_squares=use_sum_of_squares
        
        self.train_data.writeToFile(self.outputDir+'trainsamples.djcdc')
        self.val_data.writeToFile(self.outputDir+'valsamples.djcdc')
                
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        
    def trainModel(self, nepochs, batchsize, batchsize_use_sum_of_squares = False, extend_truth_list_by=0,
                   load_in_mem = False, max_files = -1, plot_batch_loss = False, attack = None, att_magnitude = 0., restrict_impact = -1, start_attack_after = 0, do_micro_tests_only = False, **trainargs):
        
        self._initTraining(batchsize, batchsize_use_sum_of_squares)
        print('starting training')
        if load_in_mem:
            print('make features')
            X_train = self.train_data.getAllFeatures(nfiles=max_files)
            X_test = self.val_data.getAllFeatures(nfiles=max_files)
            print('make truth')
            Y_train = self.train_data.getAllLabels(nfiles=max_files)
            Y_test = self.val_data.getAllLabels(nfiles=max_files)
            self.keras_model.fit(X_train, Y_train, batch_size=batchsize, epochs=nepochs,
                                 callbacks=self.callbacks.callbacks,
                                 validation_data=(X_test, Y_test),
                                 max_queue_size=1,
                                 use_multiprocessing=False,
                                 workers=0,    
                                 **trainargs)
            
        else:
            #prepare generator 
            print("setting up generator... can take a while")
            traingen = self.train_data.invokeGenerator()
            valgen = self.val_data.invokeGenerator()
            #this is fixed
            traingen.setBatchSize(batchsize)
            valgen.setBatchSize(batchsize)
            #traingen.extend_truth_list_by = extend_truth_list_by
            #valgen.extend_truth_list_by = extend_truth_list_by
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            #self.optimizer.to(self.device)
            
            epsilon_factors = {
                'glob' : torch.Tensor(np.load(epsilons_per_feature['glob']).transpose()).to(self.device),
                'cpf' : torch.Tensor(np.load(epsilons_per_feature['cpf']).transpose()).to(self.device),
                'npf' : torch.Tensor(np.load(epsilons_per_feature['npf']).transpose()).to(self.device),
                'vtx' : torch.Tensor(np.load(epsilons_per_feature['vtx']).transpose()).to(self.device),
                #'cpf_pts' : torch.cat((torch.Tensor(np.load(epsilons_per_feature['cpf_pts']).transpose()).to(self.device)
                #                     torch.zeros(6, device=self.device))), # more features not currently covered with attacks
                'cpf_pts' : torch.Tensor(np.load(epsilons_per_feature['cpf_pts']).transpose()).to(self.device),
                'npf_pts' : torch.Tensor(np.load(epsilons_per_feature['npf_pts']).transpose()).to(self.device),
                'vtx_pts' : torch.Tensor(np.load(epsilons_per_feature['vtx_pts']).transpose()).to(self.device),
            }
            
            while(self.trainedepoches < nepochs):
                if self.trainedepoches < start_attack_after:
                    dynamic_attack = None
                else:
                    dynamic_attack = attack
                #this can change from epoch to epoch
                #calculate steps for this epoch
                #feed info below
                traingen.prepareNextEpoch()
                valgen.prepareNextEpoch()

                if do_micro_tests_only:
                    nbatches_train = 4
                    nbatches_val = 4
                else:
                    nbatches_train = traingen.getNBatches() #might have changed due to shuffeling
                    nbatches_val = valgen.getNBatches()
                
                train_generator=traingen.feedNumpyData()
                val_generator=valgen.feedNumpyData()
            
                print('>>>> epoch', self.trainedepoches,"/",nepochs)
                print('training batches: ',nbatches_train)
                print('validation batches: ',nbatches_val)
                
                with tqdm(total = nbatches_train) as epoch_pbar:
                    epoch_pbar.set_description(f'Epoch {self.trainedepoches + 1}')

                    self.model.train()
                    for param_group in self.optimizer.param_groups:
                        print('/n Learning rate = '+str(param_group['lr'])+' /n')
                    train_loss = train_loop(train_generator, nbatches_train, self.model, self.criterion, self.optimizer, self.device, self.trainedepoches, epoch_pbar, acc_loss=0, scaler = self.scaler, scheduler = self.scheduler, attack = dynamic_attack, att_magnitude = att_magnitude, restrict_impact = restrict_impact, epsilon_factors = epsilon_factors)
                    self.scheduler.step()
                
                    self.model.eval()
                    val_loss = val_loop(val_generator, nbatches_val, self.model, self.criterion, self.device, self.trainedepoches)

                    self.trainedepoches += 1
                    
                    if(val_loss < self.best_loss):
                        self.best_loss = val_loss
                        self.saveModel(self.model, self.optimizer, self.trainedepoches, self.scheduler, self.best_loss, train_loss, val_loss, is_best = True)

                    
                    self.saveModel(self.model, self.optimizer, self.trainedepoches, self.scheduler, self.best_loss, train_loss, val_loss, is_best = False)
                    print(self.trainedepoches, nepochs)
                    if self.trainedepoches < nepochs:
                        print('Shuffeling...')
                        traingen.shuffleFileList() # Note: DJC3 shuffleFileList, older: shuffleFilelist