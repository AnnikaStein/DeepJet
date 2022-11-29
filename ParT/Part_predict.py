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

args = parser.parse_args()
batchsize = int(args.b)

import imp
import numpy as np
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator
import tempfile
import atexit
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_deepjet import DeepJet
#from pytorch_deepjet_transformer_v2 import DeepJetTransformer, ParticleTransformer
#from pytorch_deepjet_transformer_v2_corr import DeepJetTransformer, ParticleTransformer
from pytorch_deepjet_transformer_v4_corr import DeepJetTransformer, ParticleTransformer
from torch.optim import Adam, SGD
from tqdm import tqdm

inputdatafiles=[]
inputdir=None

def test_loop(dataloader, model, nbatches, pbar):
    predictions = 0
    
    with torch.no_grad():
        for b in range(nbatches):

            features_list, truth_list = next(dataloader)

            glob = torch.Tensor(features_list[0]).to(device)
            cpf = torch.Tensor(features_list[1]).to(device)
            npf = torch.Tensor(features_list[2]).to(device)
            vtx = torch.Tensor(features_list[3]).to(device)
            cpf_4v = torch.Tensor(features_list[4]).to(device)
            npf_4v = torch.Tensor(features_list[5]).to(device)
            vtx_4v = torch.Tensor(features_list[6]).to(device)
            #dist = torch.Tensor(features_list[7]).to(device)
            inpt = (cpf, npf, vtx, cpf_4v, npf_4v, vtx_4v)#, dist)
            #_, pred = model(inpt)#.cpu().numpy()
            #pred = pred.cpu().numpy()
            pred = nn.Softmax(dim=1)(model(inpt)).cpu().numpy()
            
            if b == 0:
                predictions = pred
            else:
                predictions = np.concatenate((predictions, pred), axis=0)
            desc = 'Predicting probs : '
            pbar.set_description(desc)
            pbar.update(1)
        
    return predictions

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
    model = ParticleTransformer(num_classes = 6, num_enc = 3, for_inference = False)
    
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
            else:
                td.readFromFileBuffered(use_inputdir+"/"+inputfile)
    else:
        print('converting '+inputfile)
        if inputfile[:4] == '/eos':
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
    
    predicted = test_loop(gen.feedNumpyData(), model, nbatches=gen.getNBatches(), pbar = pbar)
    
    x = td.transferFeatureListToNumpy(args.pad_rowsplits)
    w = td.transferWeightListToNumpy(args.pad_rowsplits)
    y = td.transferTruthListToNumpy(args.pad_rowsplits)

    td.clear()
    gen.clear()
    
    if not type(predicted) == list: #circumvent that keras return only an array if there is just one list item
        predicted = [predicted]   
    overwrite_outname = td.writeOutPrediction(predicted, x, y, w, args.outputDir + "/" + outfilename, use_inputdir+"/"+inputfile)
    if overwrite_outname is not None:
        outfilename = overwrite_outname
    outputs.append(outfilename)
    
with open(args.outputDir + "/outfiles.txt","w") as f:
    for l in outputs:
        f.write(l+'\n')
