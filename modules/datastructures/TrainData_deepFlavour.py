import gc
from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy as np
import uproot3 as u3
import uproot as u
import awkward as ak
import pandas as pd

GLOBAL_PREFIX = ""

def uproot_root2array(tree, stop=None, branches=None):
    dtypes = np.dtype( [(b, np.dtype("O")) for b in branches] )
#    if isinstance(fname, list):
 #       fname = fname[0]
    #tree = u.open(fname)[treename]

    print ("0",branches[0], fname)

    new_arr = np.empty( len(tree[branches[0]].array()), dtype=dtypes)

    for branch in branches:
        print (branch)
        new_arr[branch] = np.array( ak.to_list( tree[branch].array() ), dtype="O")

    return new_arr

def uproot_tree_to_numpy(tree, inbranches_listlist, nMaxlist, nevents, stop=None, branches=None, flat=True):
    
    #open the root file, tree and get the branches list wanted for building the array
    #tree  = u.open(fname)[treename]
    branches = tree.arrays(inbranches_listlist, library = 'numpy')
    #branches = [tree[branch_name].array() for branch_name in inbranches_listlist]
        
    #Initialize the output_array with the correct dimension and 0s everywhere. We will fill the correct 
    if nMaxlist == 1:
        output_array = np.zeros(shape=(nevents, len(inbranches_listlist)))
        
        #Loop and fill our output_array
        for i in range(nevents):
            for j, branch in enumerate(inbranches_listlist):
                output_array[i,j] = branches[branch][i]
                
    if nMaxlist > 1:
        output_array = np.zeros(shape=(nevents, len(inbranches_listlist), nMaxlist))
        
        #Loop and fill w.r.t. the zero padding method our output_array
        for i in range(nevents):
            lenght = len(branches[inbranches_listlist[0]][i])
            for j, branch in enumerate(inbranches_listlist):
                if lenght >= nMaxlist:
                    output_array[i,j,:] = branches[branch][i][:nMaxlist]
                if lenght < nMaxlist:
                    output_array[i,j,:lenght] = branches[branch][i]
                    
        output_array = np.transpose(output_array, (0, 2, 1))
    
    ### Debugging lines ###
    print(output_array.shape)
    #print(output_array[:3,0])
    
    return  output_array

def uproot_MeanNormZeroPad(Filename_in,MeanNormTuple,inbranches_listlist, nMaxslist,nevents):
    # savely copy lists (pass by value)
    import copy
    inbranches_listlist=copy.deepcopy(inbranches_listlist)
    nMaxslist=copy.deepcopy(nMaxslist)

    # Read in total number of events
    totallengthperjet = 0
    for i in range(len(nMaxslist)):
        if nMaxslist[i]>=0:
            totallengthperjet+=len(inbranches_listlist[i])*nMaxslist[i]
        else:
            totallengthperjet+=len(inbranches_listlist[i]) #flat branch

    print("Total event-length per jet: {}".format(totallengthperjet))

    #shape could be more generic here... but must be passed to c module then
    array = numpy.zeros((nevents,totallengthperjet) , dtype='float32')

    # filling mean and normlist
    normslist=[]
    meanslist=[]
    for inbranches in inbranches_listlist:
        means=[]
        norms=[]
        for b in inbranches:
            if MeanNormTuple is None:
                means.append(0)
                norms.append(1)
            else:
                means.append(MeanNormTuple[b][0])
                norms.append(MeanNormTuple[b][1])
        meanslist.append(means)
        normslist.append(norms)

    # now start filling the array


def map_prefix(elements):
    if isinstance(elements, list):
        return list(map( lambda x: GLOBAL_PREFIX + x, elements))
    elif isinstance(elements, tuple):
        return tuple(map( lambda x: GLOBAL_PREFIX + x, elements))
    elif isinstance(elements, (str)):
        return GLOBAL_PREFIX + elements
    elif isinstance(elements, bytes):
        return GLOBAL_PREFIX + elements.decode("utf-8")
    else:
        print("Error, you gave >>{}<< which is unknown".format(elements))
        raise NotImplementedError

class TrainData_ParT(TrainData):
    def __init__(self):

        TrainData.__init__(self)        
        
        self.description = "ParT inputs"
        
        self.truth_branches = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isGCC','isCC','isUD','isS','isG']
        self.undefTruth=['isUndefined', 'isPU']
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        self.remove = True
        self.referenceclass= 'isB'  #'flatten'  #Choose 'flatten' for flat or one of the truth branch for ref
        self.red_classes = ['cat_B','cat_C','cat_UDS','cat_G'] #reduced classes (flat only)
        self.truth_red_fusion = [('isB','isBB','isGBB','isLeptonicB','isLeptonicB_C'),('isC','isGCC','isCC'),('isUD','isS'),('isG')] #Indicates how you are making the fusion of your truth branches to the reduced classes for the flat reweighting
        self.class_weights = np.array([1.00,1.00,2.5,5.0], dtype=float)  #Ratio between our reduced classes (flat only)
        self.weight_binX = np.array([15, 20, 26, 35, 46, 61, 80, 106, 141, 186, 247, 326, 432, 571, 756, 1000],dtype=float) #Flat reweighting
        self.weight_binY = np.array(
            [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float) #Flat reweighting
 
        self.global_branches = ['jet_pt', 'jet_eta',
                                'nCpfcand','nNpfcand',
                                'nsv','npv',
                                'TagVarCSV_trackSumJetEtRatio',
                                'TagVarCSV_trackSumJetDeltaR',
                                'TagVarCSV_vertexCategory',
                                'TagVarCSV_trackSip2dValAboveCharm',
                                'TagVarCSV_trackSip2dSigAboveCharm',
                                'TagVarCSV_trackSip3dValAboveCharm',
                                'TagVarCSV_trackSip3dSigAboveCharm',
                                'TagVarCSV_jetNSelectedTracks',
                                'TagVarCSV_jetNTracksEtaRel']
                
        self.cpf_branches = ['Cpfcan_BtagPf_trackEtaRel',
                             'Cpfcan_BtagPf_trackPtRel',
                             'Cpfcan_BtagPf_trackPPar',
                             'Cpfcan_BtagPf_trackDeltaR',
                             'Cpfcan_BtagPf_trackPParRatio',
                             'Cpfcan_BtagPf_trackSip2dVal',
                             'Cpfcan_BtagPf_trackSip2dSig',
                             'Cpfcan_BtagPf_trackSip3dVal',
                             'Cpfcan_BtagPf_trackSip3dSig',
                             'Cpfcan_BtagPf_trackJetDistVal',
                             'Cpfcan_ptrel',
                             'Cpfcan_drminsv',
                            # 'Cpfcan_distminsv', ## don't use this for Run 3 2023 model !!!
                             'Cpfcan_VTX_ass',
                             'Cpfcan_puppiw',
                             'Cpfcan_chi2',
                             'Cpfcan_quality']

        self.n_cpf = 25

        self.npf_branches = ['Npfcan_ptrel', 'Npfcan_etarel', 'Npfcan_phirel', 'Npfcan_deltaR','Npfcan_isGamma','Npfcan_HadFrac','Npfcan_drminsv','Npfcan_puppiw']
        self.n_npf = 25

        self.vtx_branches = ['sv_pt','sv_deltaR',
                             'sv_mass',
                             'sv_etarel',
                             'sv_phirel',
                             'sv_ntracks',
                             'sv_chi2',
                             'sv_normchi2',
                             'sv_dxy',
                             'sv_dxysig',
                             'sv_d3d',
                             'sv_d3dsig',
                             'sv_costhetasvpv',
                             'sv_enratio',
        ]

        self.n_vtx = 5
        
        self.cpf_pts_branches = ['Cpfcan_pt','Cpfcan_eta',
                                 'Cpfcan_phi', 'Cpfcan_e',
                                 #'Cpfcan_nhitpixelBarrelLayer1', 'Cpfcan_nhitpixelBarrelLayer2',
                                 #'Cpfcan_nhitpixelEndcapLayer1', 'Cpfcan_nhitpixelEndcapLayer2',
                                 #'Cpfcan_numberOfValidHits', 'Cpfcan_numberOfValidPixelHits'
                                ]
        
        self.npf_pts_branches = ['Npfcan_pt','Npfcan_eta',
                                 'Npfcan_phi', 'Npfcan_e']
        
        self.vtx_pts_branches = ['sv_pt','sv_eta',
                                 'sv_phi','sv_e']

        self.reduced_truth = ['isB','isBB','isLeptonicB','isC','isUDS','isG']
        
    def createWeighterObjects(self, allsourcefiles):
        # 
        # Calculates the weights needed for flattening the pt/eta spectrum
        
        from DeepJetCore.Weighter import Weighter
        weighter = Weighter()
        weighter.undefTruth = self.undefTruth
        weighter.class_weights = self.class_weights

        branches = [self.weightbranchX,self.weightbranchY]
        branches.extend(self.truth_branches)

        if self.remove:
            weighter.setBinningAndClasses(
                [self.weight_binX,self.weight_binY],
                self.weightbranchX,self.weightbranchY,
                self.truth_branches, self.red_classes,
                self.truth_red_fusion, method = self.referenceclass
            )

        counter=0
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                events = u.open(fname)["deepntuplizer/tree"]
                nparray = events.arrays(branches, library = 'np')
                keys = list(nparray.keys())
                for k in range(len(branches)):
                    nparray[branches[k]] = nparray.pop(keys[k])
                nparray = pd.Series(nparray)    
                norm_hist = True
                if self.referenceclass == 'flatten':
                    norm_hist = False
                weighter.addDistributions(nparray, norm_h = norm_hist)
                #del nparray
                counter=counter+1
            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)
            return {'weigther':weighter}
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        # Function to produce the numpy training arrays from root files

        from DeepJetCore.Weighter import Weighter
        from DeepJetCore.stopwatch import stopwatch
        sw=stopwatch()
        swall=stopwatch()
        if not istraining:
            self.remove = False

        def reduceTruth(uproot_arrays):
            
            b = uproot_arrays['isB']
            
            bb = uproot_arrays['isBB']
            gbb = uproot_arrays['isGBB']
            
            bl = uproot_arrays['isLeptonicB']
            blc = uproot_arrays['isLeptonicB_C']
            lepb = bl+blc
            
            c = uproot_arrays['isC']
            cc = uproot_arrays['isCC']
            gcc = uproot_arrays['isGCC']
            
            ud = uproot_arrays['isUD']
            s = uproot_arrays['isS']
            uds = ud+s
            
            g = uproot_arrays['isG']
            
            return np.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()
        
        print('reading '+filename)
        
        import ROOT
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()
        
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles

        gc.collect()
        urfile = u.open(filename)["deepntuplizer/tree"]

        x_global = uproot_tree_to_numpy(urfile,
                                        self.global_branches,1,self.nsamples,
                                        flat = True)

        x_cpf = uproot_tree_to_numpy(urfile,
                                     self.cpf_branches,self.n_cpf,self.nsamples,
                                     flat = False)

        x_npf = uproot_tree_to_numpy(urfile,
                                     self.npf_branches,self.n_npf,self.nsamples,
                                     flat = False)

        x_vtx = uproot_tree_to_numpy(urfile,
                                     self.vtx_branches,self.n_vtx,self.nsamples,
                                     flat = False)

        cpf_pts = uproot_tree_to_numpy(urfile,
                                       self.cpf_pts_branches,self.n_cpf,self.nsamples,
                                       flat = True)

        npf_pts = uproot_tree_to_numpy(urfile,
                                       self.npf_pts_branches,self.n_npf,self.nsamples,
                                       flat = False)

        vtx_pts = uproot_tree_to_numpy(urfile,
                                       self.vtx_pts_branches,self.n_vtx,self.nsamples,
                                       flat = False)

        truth_arrays = urfile.arrays(self.truth_branches, library='numpy')
        truth = reduceTruth(truth_arrays)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!

        x_global = x_global.astype(dtype='float32', order='C')
        x_cpf = x_cpf.astype(dtype='float32', order='C')
        x_npf = x_npf.astype(dtype='float32', order='C')
        x_vtx = x_vtx.astype(dtype='float32', order='C')
        cpf_pts = cpf_pts.astype(dtype='float32', order='C')
        npf_pts = npf_pts.astype(dtype='float32', order='C')
        vtx_pts = vtx_pts.astype(dtype='float32', order='C')
        
        if self.remove:
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            events = u.open(filename)["deepntuplizer/tree"]
            for_remove = events.arrays(b, library = 'pd')

            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove, use_uproot = True)
            undef=for_remove['isUndefined']
            notremoves-=undef
            pu=for_remove['isPU']
            notremoves-=pu
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.remove:
            print('remove')
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_vtx=x_vtx[notremoves > 0]
            cpf_pts=cpf_pts[notremoves > 0]
            npf_pts=npf_pts[notremoves > 0]
            vtx_pts=vtx_pts[notremoves > 0]
            truth=truth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        
        print('remove nans')
        x_global = np.where(np.isfinite(x_global) , x_global, 0)
        x_cpf = np.where(np.isfinite(x_cpf), x_cpf, 0)
        x_npf = np.where(np.isfinite(x_npf), x_npf, 0)
        x_vtx = np.where(np.isfinite(x_vtx), x_vtx, 0)
        cpf_pts = np.where(np.isfinite(cpf_pts), cpf_pts, 0)
        npf_pts = np.where(np.isfinite(npf_pts), npf_pts, 0)
        vtx_pts = np.where(np.isfinite(vtx_pts), vtx_pts, 0)
        
        return [x_global, x_cpf, x_npf, x_vtx, cpf_pts, npf_pts, vtx_pts], [truth], []

    # defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):

        print(predicted[0].shape)
        print(truth[0].shape)
        print(features[0][:,0:2].shape)
        arr = np.array(np.hstack((predicted[0],truth[0], features[0][:,0:2])))
        print(arr.shape)
        out = pd.DataFrame(arr, columns=['prob_isB', 'prob_isBB', 'prob_isLeptB', 'prob_isC', 'prob_isUDS', 'prob_isG', 'isB', 'isBB', 'isLeptB', 'isC', 'isUDS', 'isG', 'jet_pt', 'jet_eta'])
        
        files = u.recreate(outfilename)
        files["tree"] = out
        files["tree"]
        files["tree"].show()
        
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), features[0][:,0:2].transpose() ) ),
                                         names='prob_isB, prob_isBB,prob_isLeptB, prob_isC,prob_isUDS,prob_isG,isB, isBB, isLeptB, isC,isUDS,isG,jet_pt, jet_eta')
        #out = np.core.records.fromarrays(arr,
        #                                 names='prob_isB, prob_isBB,prob_isLeptB, prob_isC,prob_isUDS,prob_isG,isB, isBB, isLeptB, isC,isUDS,isG,jet_pt, jet_eta')
        #array2root(out, outfilename, 'tree')
        print('now also saving numpy arrays to', outfilename)
        np.save(outfilename.strip('.root')+'.npy', out)