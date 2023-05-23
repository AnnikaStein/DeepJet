#!/usr/bin/env python3
from argparse import ArgumentParser

parser = ArgumentParser('Apply a model to a (test) source sample.')
parser.add_argument('model')
parser.add_argument('inputModel')
parser.add_argument('trainingDataCollection', help="the training data collection. Used to infer data format and batch size.")
parser.add_argument('inputSourceFileList', help="can be text file or a DataCollection file in the same directory as the sample files, or just a single traindata file.")
parser.add_argument('outputDir', help="will be created if it doesn't exist.")
parser.add_argument("-b", help="batch size, overrides the batch size from the training data collection.",default="-1")
parser.add_argument("--gpu",  help="select specific GPU", metavar="OPT", default="")
parser.add_argument("--unbuffered", help="do not read input in memory buffered mode (for lower memory consumption on fast disks)", default=False, action="store_true")
parser.add_argument("--pad_rowsplits", help="pad the row splits if the input is ragged", default=False, action="store_true")
parser.add_argument("-attack", help="use adversarial attack (Noise|FGSM|NGM) or leave blank to use undisturbed features only", default="")
parser.add_argument("-att_magnitude", help="distort input features with adversarial attack, using specified magnitude of attack", default="-1")
parser.add_argument("-restrict_impact", help="limit attack impact to this fraction of the input value (percent-cap on distortion)", default="-1")
parser.add_argument("-save_inputs", help="besides predictions, save also all inputs (useful for attacked samples per model, once per dataset for raw as they would be all the same) [yes|no]", default="no")

args = parser.parse_args()
batchsize = int(args.b)
attack = args.attack
att_magnitude = float(args.att_magnitude)
restrict_impact = float(args.restrict_impact)

import imp
import numpy as np
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator
import tempfile
#import atexit
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_deepjet import DeepJet
#from pytorch_deepjet_transformer_v2 import DeepJetTransformer, ParticleTransformer
#from pytorch_deepjet_transformer_v2_corr import DeepJetTransformer, ParticleTransformer
#from pytorch_deepjet_transformer_v4_corr import DeepJetTransformer, ParticleTransformer
#from ParT import DeepJetTransformer, ParticleTransformer
from ParT import ParticleTransformer
from torch.optim import Adam, SGD
from tqdm import tqdm
import gc

inputdatafiles=[]
inputdir=None

from definitions_ParT import epsilons_per_feature, vars_per_candidate, defaults_per_variable
from attacks_ParT import first_order_attack #, apply_noise

glob_vars = vars_per_candidate['glob']

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

def test_loop(dataloader, model, nbatches, pbar, attack = "", att_magnitude = -1., restrict_impact = -1., loss_fn = cross_entropy_one_hot, epsilon_factors=None, defaults_device = None):
    #predictions = 0
    
    #with torch.no_grad():
    for b in range(nbatches):

        features_list, truth_list = next(dataloader)

        #glob = torch.Tensor(features_list[0]).to(device)
        cpf = torch.Tensor(features_list[1]).to(device)
        npf = torch.Tensor(features_list[2]).to(device)
        vtx = torch.Tensor(features_list[3]).to(device)
        cpf_4v = torch.Tensor(features_list[4]).to(device)
        npf_4v = torch.Tensor(features_list[5]).to(device)
        vtx_4v = torch.Tensor(features_list[6]).to(device)
        #dist = torch.Tensor(features_list[7]).to(device)
        # not necessary in general case: y = torch.Tensor(truth_list[0]).to(device)
        # do the above only for adversarial attack

        #glob[:,:] = torch.where(glob[:,:] == -999., torch.zeros(len(glob),glob_vars).to(device), glob[:,:])
        #glob[:,:] = torch.where(glob[:,:] ==   -1., torch.zeros(len(glob),glob_vars).to(device), glob[:,:])

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

        elif (attack == 'FGSM' or attack == 'NGM'):
            #print('Appyling attack', attack, 'to batch', b)
            with torch.cuda.amp.autocast():
                y = torch.Tensor(truth_list[0]).to(device)
                #glob, cpf, npf, vtx, cpf_4v, npf_4v, vtx_4v = first_order_attack(sample=(glob,cpf,npf,vtx,cpf_4v,npf_4v,vtx_4v), 
                cpf, npf, vtx, cpf_4v, npf_4v, vtx_4v = first_order_attack(sample=(cpf,npf,vtx,cpf_4v,npf_4v,vtx_4v), 
                                                                          epsilon=att_magnitude,
                                                                          dev=device,
                                                                          targets=y,
                                                                          thismodel=model,
                                                                          thiscriterion=loss_fn,
                                                                          restrict_impact=restrict_impact,
                                                                          epsilon_factors=epsilon_factors,
                                                                          defaults_per_variable = defaults_device,
                                                                          do_sign_or_normed_grad = attack)
        inpt = (cpf, npf, vtx, cpf_4v, npf_4v, vtx_4v)#, dist)
        #_, pred = model(inpt)#.cpu().numpy()
        #pred = pred.cpu().numpy()
        pred = nn.Softmax(dim=1)(model(inpt)).detach().cpu().numpy()
        del inpt
        gc.collect()
        if b == 0:
            predictions = pred
            NPYcpf, NPYnpf, NPYvtx, NPYcpf_4v, NPYnpf_4v, NPYvtx_4v = cpf.detach().cpu().numpy(),\
                                                                      npf.detach().cpu().numpy(),\
                                                                      vtx.detach().cpu().numpy(),\
                                                                      cpf_4v.detach().cpu().numpy(),\
                                                                      npf_4v.detach().cpu().numpy(),\
                                                                      vtx_4v.detach().cpu().numpy(),
                                                                       
                             
        else:
            predictions = np.concatenate((predictions, pred), axis=0)
            NPYcpf = np.concatenate((NPYcpf, cpf.detach().cpu().numpy()), axis=0)
            NPYnpf = np.concatenate((NPYnpf, npf.detach().cpu().numpy()), axis=0)
            NPYvtx = np.concatenate((NPYvtx, vtx.detach().cpu().numpy()), axis=0)
            NPYcpf_4v = np.concatenate((NPYcpf_4v, cpf_4v.detach().cpu().numpy()), axis=0)
            NPYnpf_4v = np.concatenate((NPYnpf_4v, npf_4v.detach().cpu().numpy()), axis=0)
            NPYvtx_4v = np.concatenate((NPYvtx_4v, vtx_4v.detach().cpu().numpy()), axis=0)
            
        desc = 'Predicting probs : '
        pbar.set_description(desc)
        pbar.update(1)

    return predictions, NPYcpf, NPYnpf, NPYvtx, NPYcpf_4v, NPYnpf_4v, NPYvtx_4v

## prepare input lists for different file formats
if args.inputSourceFileList[-6:] == ".djcdc":
    print('reading from data collection',args.inputSourceFileList)
    predsamples = DataCollection(args.inputSourceFileList)
    inputdir = predsamples.dataDir
    for s in predsamples.samples:
        inputdatafiles.append(s)
        
elif args.inputSourceFileList[-6:] == ".djctd":
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    infile = os.path.basename(args.inputSourceFileList)
    inputdatafiles.append(infile)
else:
    print('reading from text file',args.inputSourceFileList)
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    with open(args.inputSourceFileList, "r") as f:
        for s in f:
            inputdatafiles.append(s.replace('\n', '').replace(" ",""))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model == 'ParticleTransformer':
    #model = ParticleTransformer(num_classes = 6, num_enc = 3, for_inference = False)
    model = ParticleTransformer(num_classes = 6,
                                num_enc = 3,
                                num_head = 8,
                                embed_dim = 128,
                                cpf_dim = 16,#17,
                                npf_dim = 8,
                                vtx_dim = 14,
                                for_inference = False)

if args.model == 'ParticleTransformerOLD':
    #model = ParticleTransformer(num_classes = 6, num_enc = 3, for_inference = False)
    model = ParticleTransformer(num_classes = 6,
                                num_enc = 3,
                                num_head = 8,
                                embed_dim = 128,
                                cpf_dim = 17,
                                npf_dim = 8,
                                vtx_dim = 14,
                                for_inference = False)   
check = torch.load(args.inputModel, map_location=torch.device('cpu'))
model.to(device)
model.load_state_dict(check['state_dict'])

model.eval()
dc = None
if args.inputSourceFileList[-6:] == ".djcdc" and not args.trainingDataCollection[-6:] == ".djcdc":
    dc = DataCollection(args.inputSourceFileList)
    if batchsize < 1:
        batchsize = 1
    print('No training data collection given. Using batch size of',batchsize)
else:
    dc = DataCollection(args.trainingDataCollection)

outputs = []
os.system('mkdir -p '+args.outputDir)

for inputfile in inputdatafiles:
    
    print('predicting ',inputdir+"/"+inputfile)
    
    use_inputdir = inputdir
    if inputfile[0] == "/":
        use_inputdir=""
    outfilename = "pred_"+os.path.basename( inputfile )
    
    td = dc.dataclass()

    if inputfile[-5:] == 'djctd':
        if args.unbuffered:
            td.readFromFile(use_inputdir+"/"+inputfile)
        else:
            if inputdir[:4] == '/eos':
                td.readFromFileBuffered(inputfile)
            elif inputdir[:4] == '/afs':
                td.readFromFileBuffered(inputfile)
            else:
                td.readFromFileBuffered(use_inputdir+"/"+inputfile)
    else:
        print('converting '+inputfile)
        if inputfile[:4] == '/eos':
            td.readFromSourceFile(inputfile, dc.weighterobjects, istraining=False)
        if inputfile[:4] == '/afs':
            td.readFromSourceFile(inputfile, dc.weighterobjects, istraining=False)
        else:
            td.readFromSourceFile(use_inputdir+"/"+inputfile, dc.weighterobjects, istraining=False)

    gen = TrainDataGenerator()
    if batchsize < 1:
        batchsize = dc.getBatchSize()
    print('batch size',batchsize)
    gen.setBatchSize(batchsize)
    gen.setSquaredElementsLimit(dc.batch_uses_sum_of_squares)
    gen.setSkipTooLargeBatches(False)
    gen.setBuffer(td)

    with tqdm(total = gen.getNBatches()) as pbar:
        pbar.set_description('Predicting : ')
    
    nbatches = gen.getNBatches()
    
    if attack != "":
        print('With attack', attack)
        epsilon_factors = {
                        'glob' : torch.Tensor(np.load(epsilons_per_feature['glob']).transpose()).to(device),
                        'cpf' : torch.Tensor(np.load(epsilons_per_feature['cpf']).transpose()).to(device),
                        'npf' : torch.Tensor(np.load(epsilons_per_feature['npf']).transpose()).to(device),
                        'vtx' : torch.Tensor(np.load(epsilons_per_feature['vtx']).transpose()).to(device),
                        #'cpf_pts' : torch.cat((torch.Tensor(np.load(epsilons_per_feature['cpf_pts']).transpose()).to(device)
                        #                     torch.zeros(6, device=device))), # more features not currently covered with attacks
                        'cpf_pts' : torch.Tensor(np.load(epsilons_per_feature['cpf_pts']).transpose()).to(device),
                        'npf_pts' : torch.Tensor(np.load(epsilons_per_feature['npf_pts']).transpose()).to(device),
                        'vtx_pts' : torch.Tensor(np.load(epsilons_per_feature['vtx_pts']).transpose()).to(device),
                    }

        # defaults_device = {
        #     'glob' : torch.Tensor(defaults_per_variable['glob']).to(device),
        #     'cpf' : torch.Tensor(defaults_per_variable['cpf']).to(device),
        #     'npf' : torch.Tensor(defaults_per_variable['npf']).to(device),
        #     'vtx' : torch.Tensor(defaults_per_variable['vtx']).to(device),
        #     'cpf_pts' : torch.Tensor(defaults_per_variable['cpf_pts']).to(device),
        #     'npf_pts' : torch.Tensor(defaults_per_variable['npf_pts']).to(device),
        #     'vtx_pts' : torch.Tensor(defaults_per_variable['vtx_pts']).to(device),
        # }
        defaults_device = defaults_per_variable
        predicted, NPYcpf, NPYnpf, NPYvtx, NPYcpf_4v, NPYnpf_4v, NPYvtx_4v = test_loop(gen.feedNumpyData(), model, nbatches = nbatches, pbar = pbar, attack = attack, att_magnitude = att_magnitude, restrict_impact = restrict_impact, epsilon_factors = epsilon_factors, defaults_device = defaults_device)
        att_str = '_' + attack
        
    else:
        print('Without attack')
        predicted, NPYcpf, NPYnpf, NPYvtx, NPYcpf_4v, NPYnpf_4v, NPYvtx_4v = test_loop(gen.feedNumpyData(), model, nbatches = nbatches, pbar = pbar, attack = attack, att_magnitude = att_magnitude, restrict_impact = restrict_impact, epsilon_factors = None, defaults_device = None)
        att_str = ''
        
    if args.save_inputs == 'yes':
        np.save((args.outputDir + "/" + outfilename).strip('.root')+f'_NPYcpf{att_str}.npy', NPYcpf)
        np.save((args.outputDir + "/" + outfilename).strip('.root')+f'_NPYnpf{att_str}.npy', NPYnpf)
        np.save((args.outputDir + "/" + outfilename).strip('.root')+f'_NPYvtx{att_str}.npy', NPYvtx)
        np.save((args.outputDir + "/" + outfilename).strip('.root')+f'_NPYcpf_4v{att_str}.npy', NPYcpf_4v)
        np.save((args.outputDir + "/" + outfilename).strip('.root')+f'_NPYnpf_4v{att_str}.npy', NPYnpf_4v)
        np.save((args.outputDir + "/" + outfilename).strip('.root')+f'_NPYvtx_4v{att_str}.npy', NPYvtx_4v)
    del NPYcpf
    del NPYnpf
    del NPYvtx
    del NPYcpf_4v
    del NPYnpf_4v
    del NPYvtx_4v
    gc.collect()
    
    x = td.transferFeatureListToNumpy(args.pad_rowsplits)
    w = td.transferWeightListToNumpy(args.pad_rowsplits)
    y = td.transferTruthListToNumpy(args.pad_rowsplits)

    td.clear()
    gen.clear()
    
    if not type(predicted) == list: #circumvent that keras return only an array if there is just one list item
        predicted = [predicted]   
    overwrite_outname = td.writeOutPrediction(predicted, x, y, w, args.outputDir + "/" + outfilename, use_inputdir+"/"+inputfile)
    del x
    del y
    del w
    gc.collect()
    if overwrite_outname is not None:
        outfilename = overwrite_outname
    outputs.append(outfilename)
    
with open(args.outputDir + "/outfiles.txt","w") as f:
    for l in outputs:
        f.write(l+'\n')
