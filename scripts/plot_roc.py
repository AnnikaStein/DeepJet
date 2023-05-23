print("start import")
import os
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy.interpolate import InterpolatedUnivariateSpline
from pdb import set_trace
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import root_numpy
import ROOT
#from ROOT import TCanvas, TGraph, TGraphAsymmErrors, TH2F, TH1F
#from root_numpy import fill_hist
import uproot as u
print("finish import")


#model_name = 'adversarial_with_etarel_phirel'
#model_name = 'nominal'
#model_name = 'adversarial_eps0p005'
#model_name = 'adversarial_eps0p01'
#model_name = 'reduced_adversarial_eps0p005'
#model_name = 'nominal_bsize10k'
#model_name = 'nominal_bsize4k'
#model_name = 'adversarial_eps0p01_bsize10k'
model_name = 'testNGMPUPPI'
sampl = 'qcd'
ptmin = 30 if sampl == 'ttbar' else 300 # NOTE: if ttbar, use pt > 30, if qcd, use pt > 300 below
ptmax = 1000
etamax = 2.5
print(ptmin)
prediction_setup = f'_{sampl}_17'
#prediction_files = 'one_prediction'
prediction_files = 'outfiles'


def spit_out_roc(disc,truth_array,selection_array):

    #newx = np.logspace(-3.5, 0, 100)
    tprs = pd.DataFrame()
    truth = truth_array[selection_array]*1
    disc = disc[selection_array]
    tmp_fpr, tmp_tpr, _ = roc_curve(truth, disc)
    coords = pd.DataFrame()
    coords['fpr'] = tmp_fpr
    coords['tpr'] = tmp_tpr
    clean = coords.drop_duplicates(subset=['fpr'])
    #spline = InterpolatedUnivariateSpline(clean.fpr, clean.tpr,k=1)
    #tprs = spline(newx)
    auc_ = auc(clean.fpr,clean.tpr)
    print('AUC: ', str(auc_))
    #return tprs, newx
    return clean.tpr, clean.fpr, auc_



pred = []
isDeepJet = True
if isDeepJet:
    listbranch = ['prob_isB', 'prob_isBB','prob_isLeptB', 'prob_isC','prob_isUDS','prob_isG','isB', 'isBB', 'isLeptB', 'isC','isUDS','isG','jet_pt', 'jet_eta']
else:
    listbranch = ['prob_isB', 'prob_isBB', 'prob_isC','prob_isUDSG','isB', 'isBB', 'isC','isUDSG','jet_pt', 'jet_eta']

    

#dirz = f'/eos/user/a/anstein/DeepJet/Train_DF/{model_name}/predict{prediction_setup}/'
#dirz = f'/eos/user/a/anstein/public/DeepJet/Train_DF_Run2/{model_name}/predict{prediction_setup}/'
dirz = f'/eos/user/a/anstein/ParT/{model_name}/predict{prediction_setup}/'
truthfile = open( dirz+prediction_files+'.txt','r')

config_name = model_name + prediction_setup + '_' + prediction_files

print("opened text file")
#rfile1 = ROOT.TChain("tree")
count = 0
import numpy.lib.recfunctions as rf
for i,line in enumerate(truthfile):
    count += 1
    if len(line) < 1: continue
    file1name=str(dirz+line.split('\n')[0])
    # ToDo
    #events = u.open(file1name)["tree"]
    #nparray = events.arrays(listbranch, library = 'np') if i==0 else np.concatenate((nparray,events.arrays(listbranch, library = 'np')))
    events = rf.structured_to_unstructured(np.array(np.load(file1name.strip('.root')+'.npy')))
    
    #events = np.core.records.fromfile(file1name.strip('.root')+'.npy', dtype=float)
    nparray = events if i == 0 else np.concatenate((nparray, events))
    #rfile1.Add(file1name)

print("added files")
#df = root_numpy.tree2array(rfile1, branches = listbranch)
#keys = list(nparray.keys())
#print(nparray[0][0])
print(type(nparray))
print(len(nparray))
print(type(nparray[0]))
#for k in range(len(listbranch)):
#    nparray[listbranch[k]] = nparray[:,k]
df = np.core.records.fromarrays([nparray[:,k] for k in range(len(listbranch))],names=listbranch)
print("converted to df")

if isDeepJet:
    b_jets = df['isB']+df['isBB']+df['isLeptB']
    c_jets = df['isC']
    b_out = df['prob_isB']+df['prob_isBB']+df['prob_isLeptB']
    c_out = df['prob_isC']
    light_out = df['prob_isUDS']+df['prob_isG']
    bvsl = np.where((b_out + light_out)>=0,
                    (b_out)/(b_out + light_out),
                    -1)
    bvsc = np.where((b_out + c_out)>=0,
                    (b_out)/(b_out + c_out),
                    -1)
    cvsb = np.where((b_out + c_out)>=0,
                    (c_out)/(b_out + c_out),
                    -1)
    cvsl = np.where((light_out + c_out)>=0,
                    (c_out)/(light_out + c_out),
                    -1)
    summed_truth = df['isB']+df['isBB']+df['isLeptB']+df['isC']+df['isUDS']+df['isG']
    
    veto_b = (df['isB'] != 1) & (df['isBB'] != 1) & (df['isLeptB'] != 1) & (summed_truth != 0)
    veto_c = (df['isC'] != 1) & (summed_truth != 0)
    veto_udsg = (df['isUDS'] != 1) & (df['isG'] != 1) & (summed_truth != 0)
    
else:
    b_jets = df['isB']+df['isBB']
    b_disc = df['prob_isB']+df['prob_isBB']
    summed_truth = df['isB']+df['isBB']+df['isC']+df['isUDSG']
    veto_c = (df['isC'] != 1) & (summed_truth != 0)
    veto_udsg = (df['isUDSG'] != 1) & (summed_truth != 0)


#f = ROOT.TFile("ROCS_DeepJet_adversarial_FGSM_onefile.root", "recreate")

x1, y1, auc1 = spit_out_roc(bvsl[( df['jet_pt'] > ptmin) & ( df['jet_pt'] < ptmax) & ( abs(df['jet_eta']) < etamax)],
                            b_jets[( df['jet_pt'] > ptmin) & ( df['jet_pt'] < ptmax) & ( abs(df['jet_eta']) < etamax)],
                            veto_c[( df['jet_pt'] > ptmin) & ( df['jet_pt'] < ptmax) & ( abs(df['jet_eta']) < etamax)])
x2, y2, auc2 = spit_out_roc(cvsb[( df['jet_pt'] > ptmin) & ( df['jet_pt'] < ptmax) & ( abs(df['jet_eta']) < etamax)],
                            c_jets[( df['jet_pt'] > ptmin) & ( df['jet_pt'] < ptmax) & ( abs(df['jet_eta']) < etamax)],
                            veto_udsg[( df['jet_pt'] > ptmin) & ( df['jet_pt'] < ptmax) & ( abs(df['jet_eta']) < etamax)])
x3, y3, auc3 = spit_out_roc(cvsl[( df['jet_pt'] > ptmin) & ( df['jet_pt'] < ptmax) & ( abs(df['jet_eta']) < etamax)],
                            c_jets[( df['jet_pt'] > ptmin) & ( df['jet_pt'] < ptmax) & ( abs(df['jet_eta']) < etamax)],
                            veto_b[( df['jet_pt'] > ptmin) & ( df['jet_pt'] < ptmax) & ( abs(df['jet_eta']) < etamax)])
x4, y4, auc4 = spit_out_roc(bvsc[( df['jet_pt'] > ptmin) & ( df['jet_pt'] < ptmax) & ( abs(df['jet_eta']) < etamax)],
                            b_jets[( df['jet_pt'] > ptmin) & ( df['jet_pt'] < ptmax) & ( abs(df['jet_eta']) < etamax)],
                            veto_udsg[( df['jet_pt'] > ptmin) & ( df['jet_pt'] < ptmax) & ( abs(df['jet_eta']) < etamax)])
np.save(dirz + f'BvL_{prediction_files}_NEW.npy',np.array([x1,y1,auc1],dtype=object))
np.save(dirz + f'CvB_{prediction_files}_NEW.npy',np.array([x2,y2,auc2],dtype=object))
np.save(dirz + f'CvL_{prediction_files}_NEW.npy',np.array([x3,y3,auc3],dtype=object))
np.save(dirz + f'BvC_{prediction_files}_NEW.npy',np.array([x4,y4,auc4],dtype=object))


#gr1 = TGraph( 100, x1, y1 )
#gr1.SetName("roccurve_0")
#gr2 = TGraph( 100, x2, y2 )
#gr2.SetName("roccurve_1")
#gr1.Write()
#gr2.Write()
#f.Write()
