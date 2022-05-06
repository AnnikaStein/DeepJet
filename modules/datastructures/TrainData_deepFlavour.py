from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy as np
import uproot3 as u3
import awkward as ak

GLOBAL_PREFIX = ""

def uproot_root2array(fname, treename, stop=None, branches=None):
    dtypes = np.dtype( [(b, np.dtype("O")) for b in branches] )
    if isinstance(fname, list):
        fname = fname[0]
    tree = u3.open(fname)[treename]

    print ("0",branches[0], fname)

    new_arr = np.empty( len(tree[branches[0]].array()), dtype=dtypes)

    for branch in branches:
        print (branch)
        new_arr[branch] = np.array( ak.to_list( tree[branch].array() ), dtype="O")

    return new_arr

def uproot_tree_to_numpy(fname, inbranches_listlist, nMaxlist, nevents, treename="deepntuplizer/tree", stop=None, branches=None, flat=True):
    
    #open the root file, tree and get the branches list wanted for building the array
    tree  = u3.open(fname)[treename]
    branches = [tree[branch_name].array() for branch_name in inbranches_listlist]
        
    #Initialize the output_array with the correct dimension and 0s everywhere. We will fill the correct 
    if nMaxlist == 1:
        output_array = np.zeros(shape=(nevents, len(inbranches_listlist)))
        
        #Loop and fill our output_array
        for i in range(nevents):
            for j, branch in enumerate(inbranches_listlist):
                output_array[i,j] = branches[j][i]
                
    if nMaxlist > 1:
        output_array = np.zeros(shape=(nevents, len(inbranches_listlist), nMaxlist))
        
        #Loop and fill w.r.t. the zero padding method our output_array
        for i in range(nevents):
            lenght = len(branches[0][i])
            for j, branch in enumerate(inbranches_listlist):
                if lenght >= nMaxlist:
                    output_array[i,j,:] = branches[j][i,:nMaxlist]
                if lenght < nMaxlist:
                    output_array[i,j,:lenght] = branches[j][i,:]
                    
        output_array = np.transpose(output_array, (0, 2, 1))
    
    ### Debugging lines ###
    print(output_array.shape)
    
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

class TrainData_DeepCluster(TrainData):
    def __init__(self):

        TrainData.__init__(self)        
        
        self.description = "DeepCluster inputs"
        
        self.truth_branches = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isGCC','isCC','isUD','isS','isG']
        self.undefTruth=['isUndefined', 'isPU']
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        self.remove = True
        self.referenceclass='isB'  #Choose 'flatten' for flat or one of the truth branch for ref
        self.red_classes = ['cat_B','cat_C','cat_UDS','cat_G'] #reduced classes (flat only)
        self.truth_red_fusion = [('isB','isBB','isGBB','isLeptonicB','isLeptonicB_C'),('isC','isGCC','isCC'),('isUD','isS'),('isG')] #Indicates how you are making the fusion of your truth branches to the reduced classes for the flat reweighting
        self.class_weights = np.array([1.00,1.00,2.50,5.00], dtype=float)  #Ratio between our reduced classes (flat only)
        #self.weight_binX = np.array([15, 20, 26, 35, 46, 61, 80, 106, 141, 186, 247, 326, 432, 571, 756, 1000],dtype=float) #Flat reweighting
        self.weight_binX = np.arange(100,1001,100)
        #self.weight_binX = (np.exp(np.linspace(np.log(15), np.log(1000), 16))).tolist()
        #self.weight_binX = np.array([10,25,30,35,40,45,50,60,75,100,125,150,175,200,250,300,400,500,600,2000],dtype=float) #Ref reweighting
        self.weight_binY = np.array(
            [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float) #Flat reweighting
        #self.weight_binY = np.array([-2.5,-1.136,-0.516,-0.235,-0.107,-0.0484,-0.0220, 0.0220, 0.0484, 0.107, 0.235,0.516, 1.136, 2.5], dtype=float) #Flat-hybrid reweighting
         #self.weight_binY = np.array(
         #   [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float) #Ref reweighting

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
                             'Cpfcan_VTX_ass',
                             'Cpfcan_puppiw',
                             'Cpfcan_chi2',
                             'Cpfcan_quality',
                             'Cpfcan_nhitpixelBarrelLayer1', 'Cpfcan_nhitpixelBarrelLayer2', 'Cpfcan_nhitpixelBarrelLayer3', 'Cpfcan_nhitpixelBarrelLayer4',
                             'Cpfcan_nhitpixelEndcapLayer1', 'Cpfcan_nhitpixelEndcapLayer2', 'Cpfcan_numberOfValidHits', 'Cpfcan_numberOfValidPixelHits',
                             'Cpfcan_numberOfValidStripHits', 'Cpfcan_numberOfValidStripTIBHits', 'Cpfcan_numberOfValidStripTIDHits', 'Cpfcan_numberOfValidStripTOBHits',
                             'Cpfcan_numberOfValidStripTECHits']
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

#        self.pixel_branches = ['r004', 'r006', 'r008', 'r010', 'r016', 'rvar', 'rvwt']
 #       self.n_pixel = 4


        self.reduced_truth = ['isB', 'isC', 'isUDS', 'isG']
        #self.reduced_truth = ['isB','isBB', 'isLeptonicB', 'isC', 'isUDS', 'isG']

        
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
        import ROOT
        from root_numpy import tree2array, root2array
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                nparray = root2array(
                    fname,
                    treename = "deepntuplizer/tree",
                    stop = None,
                    branches = branches
                )
                norm_hist = True
                if self.referenceclass == 'flatten':
                    norm_hist = False
                weighter.addDistributions(nparray, norm_h = norm_hist)
                #del nparray
                counter=counter+1
            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)
            weighter.printHistos('/afs/cern.ch/user/a/ademoor/Flatten/') #If you need to print the 2D histo, choose your output dir
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
            
            b = uproot_arrays[b'isB']
            
            bb = uproot_arrays[b'isBB']
            gbb = uproot_arrays[b'isGBB']
            
            bl = uproot_arrays[b'isLeptonicB']
            blc = uproot_arrays[b'isLeptonicB_C']
            lepb = bl+blc
            
            c = uproot_arrays[b'isC']
            cc = uproot_arrays[b'isCC']
            gcc = uproot_arrays[b'isGCC']
            
            ud = uproot_arrays[b'isUD']
            s = uproot_arrays[b'isS']
            uds = ud+s
            
            g = uproot_arrays[b'isG']
            
            return np.vstack((b+bb+gbb+lepb,c+cc+gcc,uds,g)).transpose()
            #return np.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()
        
        print('reading '+filename)
        
        import ROOT
        from root_numpy import tree2array, root2array
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()
        
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles
        
#        x_global = MeanNormZeroPad(filename,None,
 #                                  [self.global_branches],
  #                                 [1],self.nsamples)

#        x_cpf = MeanNormZeroPadParticles(filename,None,
 #                                  self.cpf_branches,
  #                                 self.n_cpf,self.nsamples)

#        x_npf = MeanNormZeroPadParticles(filename,None,
 #                                        self.npf_branches,
  #                                       self.n_npf,self.nsamples)

#        x_vtx = MeanNormZeroPadParticles(filename,None,
 #                                        self.vtx_branches,
  #                                       self.n_vtx,self.nsamples)

#        x_tracks = MeanNormZeroPadParticles(filename,None,
 #                                        self.tracks_branches,
  #                                       self.n_tracks,self.nsamples)

        x_global = uproot_tree_to_numpy(filename,
                                        self.global_branches,1,self.nsamples,
                                        treename='deepntuplizer/tree', flat = True)

        x_cpf = uproot_tree_to_numpy(filename,
                                     self.cpf_branches,self.n_cpf,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        x_npf = uproot_tree_to_numpy(filename,
                                     self.npf_branches,self.n_npf,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        x_vtx = uproot_tree_to_numpy(filename,
                                     self.vtx_branches,self.n_vtx,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)
        
#        x_pixel = uproot_tree_to_numpy(filename,
 #                                   self.pixel_branches,self.n_pixel,self.nsamples,
  #                                  treename='deepntuplizer/tree', flat = False)

        
        import uproot3 as uproot
        #import uproot
        urfile = uproot.open(filename)["deepntuplizer/tree"]
        truth_arrays = urfile.arrays(self.truth_branches)
        truth = reduceTruth(truth_arrays)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!

        x_global = x_global.astype(dtype='float32', order='C')
        x_cpf = x_cpf.astype(dtype='float32', order='C')
        x_npf = x_npf.astype(dtype='float32', order='C')
        x_vtx = x_vtx.astype(dtype='float32', order='C')
#        x_pixel = x_pixel.astype(dtype='float32', order='C')

        print("Check des shapes : ")
        print(truth.shape)
        print(x_global.shape)
        print(x_cpf.shape)
        print(x_npf.shape)
        print(x_vtx.shape)
#        print(x_pixel.shape)
        
        if self.remove:
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            for_remove = root2array(
                filename,
                treename = "deepntuplizer/tree",
                stop = None,
                branches = b
            )
            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
            undef=for_remove['isUndefined']
            notremoves-=undef
            pu=for_remove['isPU']
            notremoves-=pu
            print('took ', sw.getAndReset(), ' to create remove indices')

#        random_pick = np.random.rand(x_global.shape[0])
 #       selection1 = (x_global[:,0] <= 80)
  #      selection2 = (x_global[:,0] > 80) & (x_global[:,0] <= 140)
   #     selection3 = (x_global[:,0] > 140) & (x_global[:,0] <= 186)
    #    selection4 = (x_global[:,0] > 186) & (x_global[:,0] <= 247)

#        notremoves[selection1] -= 4*random_pick[selection1]
 #       notremoves[selection2] -= 2.667*random_pick[selection2]
  #      notremoves[selection3] -= 2*random_pick[selection3]
   #     notremoves[selection4] -= 1*random_pick[selection4]

        if self.remove:
            print('remove')
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_vtx=x_vtx[notremoves > 0]
#            x_pixel=x_pixel[notremoves > 0]
            truth=truth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        
        print('remove nans')
        x_global = np.where(np.isfinite(x_global) , x_global, 0)
        x_cpf = np.where(np.isfinite(x_cpf), x_cpf, 0)
        x_npf = np.where(np.isfinite(x_npf), x_npf, 0)
        x_vtx = np.where(np.isfinite(x_vtx), x_vtx, 0)
#        x_pixel = np.where(np.isfinite(x_pixel), x_pixel, 0)
        
        return [x_global,x_cpf,x_npf,x_vtx], [truth], []

    # defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), features[0][:,0:2].transpose())),
                                        #names='prob_isB, prob_isBB, prob_isLeptB, prob_isC, prob_isUDS, prob_isG, isB, isBB, isLeptB, isC, isUDS, isG, jet_pt, jet_eta')
                                         names='prob_isB, prob_isC, prob_isUDS, prob_isG, isB, isC, isUDS, isG, jet_pt, jet_eta')
                                        #names='prob_isB, prob_isBB,prob_isLeptB,prob_isC,prob_isUDS,prob_isG,isB,isBB,isLeptB,isC,isUDS,isG,jet_pt, jet_eta')
        array2root(out, outfilename, 'tree')



class TrainData_DeepVertex(TrainData):
    def __init__(self):

        TrainData.__init__(self)        
        
        self.description = "DeepVertex inputs"
        
        self.truth_branches = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isGCC','isCC','isUD','isS','isG']
        self.undefTruth=['isUndefined', 'isPU']
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        self.remove = True
        self.referenceclass='isB'  #Choose 'flatten' for flat or one of the truth branch for ref
        self.red_classes = ['cat_B','cat_C','cat_UDS','cat_G'] #reduced classes (flat only)
        self.truth_red_fusion = [('isB','isBB','isGBB','isLeptonicB','isLeptonicB_C'),('isC','isGCC','isCC'),('isUD','isS'),('isG')] #Indicates how you are making the fusion of your truth branches to the reduced classes for the flat reweighting
        self.class_weights = np.array([1.00,1.00,2.50,5.00], dtype=float)  #Ratio between our reduced classes (flat only)
        #self.weight_binX = np.array([15, 20, 26, 35, 46, 61, 80, 106, 141, 186, 247, 326, 432, 571, 756, 1000],dtype=float) #Flat reweighting
        self.weight_binX = np.array([100,125,150,175,200,250,300,400,500,600,2000],dtype=float)
        #self.weight_binX = (np.exp(np.linspace(np.log(15), np.log(1000), 16))).tolist()
        #self.weight_binX = np.array([10,25,30,35,40,45,50,60,75,100,125,150,175,200,250,300,400,500,600,2000],dtype=float) #Ref reweighting
        self.weight_binY = np.array(
            [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float) #Flat reweighting
        #self.weight_binY = np.array([-2.5,-1.136,-0.516,-0.235,-0.107,-0.0484,-0.0220, 0.0220, 0.0484, 0.107, 0.235,0.516, 1.136, 2.5], dtype=float) #Flat-hybrid reweighting
         #self.weight_binY = np.array(
         #   [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float) #Ref reweighting

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

        self.pixel_branches = ['r004', 'r006', 'r008', 'r010', 'r016', 'rvar', 'rvwt']
        self.n_pixel = 4


        self.reduced_truth = ['isB', 'isC', 'isUDS', 'isG']
        #self.reduced_truth = ['isB','isBB', 'isLeptonicB', 'isC', 'isUDS', 'isG']

        
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
        import ROOT
        from root_numpy import tree2array, root2array
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                nparray = root2array(
                    fname,
                    treename = "deepntuplizer/tree",
                    stop = None,
                    branches = branches
                )
                norm_hist = True
                if self.referenceclass == 'flatten':
                    norm_hist = False
                weighter.addDistributions(nparray, norm_h = norm_hist)
                #del nparray
                counter=counter+1
            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)
            weighter.printHistos('/afs/cern.ch/user/a/ademoor/Flatten/') #If you need to print the 2D histo, choose your output dir
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
            
            b = uproot_arrays[b'isB']
            
            bb = uproot_arrays[b'isBB']
            gbb = uproot_arrays[b'isGBB']
            
            bl = uproot_arrays[b'isLeptonicB']
            blc = uproot_arrays[b'isLeptonicB_C']
            lepb = bl+blc
            
            c = uproot_arrays[b'isC']
            cc = uproot_arrays[b'isCC']
            gcc = uproot_arrays[b'isGCC']
            
            ud = uproot_arrays[b'isUD']
            s = uproot_arrays[b'isS']
            uds = ud+s
            
            g = uproot_arrays[b'isG']
            
            return np.vstack((b+bb+gbb+lepb,c+cc+gcc,uds,g)).transpose()
            #return np.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()
        
        print('reading '+filename)
        
        import ROOT
        from root_numpy import tree2array, root2array
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()
        
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles
        
#        x_global = MeanNormZeroPad(filename,None,
 #                                  [self.global_branches],
  #                                 [1],self.nsamples)

#        x_cpf = MeanNormZeroPadParticles(filename,None,
 #                                  self.cpf_branches,
  #                                 self.n_cpf,self.nsamples)

#        x_npf = MeanNormZeroPadParticles(filename,None,
 #                                        self.npf_branches,
  #                                       self.n_npf,self.nsamples)

#        x_vtx = MeanNormZeroPadParticles(filename,None,
 #                                        self.vtx_branches,
  #                                       self.n_vtx,self.nsamples)

#        x_tracks = MeanNormZeroPadParticles(filename,None,
 #                                        self.tracks_branches,
  #                                       self.n_tracks,self.nsamples)

        x_global = uproot_tree_to_numpy(filename,
                                        self.global_branches,1,self.nsamples,
                                        treename='deepntuplizer/tree', flat = True)

        x_cpf = uproot_tree_to_numpy(filename,
                                     self.cpf_branches,self.n_cpf,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        x_npf = uproot_tree_to_numpy(filename,
                                     self.npf_branches,self.n_npf,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        x_vtx = uproot_tree_to_numpy(filename,
                                     self.vtx_branches,self.n_vtx,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)
        
        x_pixel = uproot_tree_to_numpy(filename,
                                    self.pixel_branches,self.n_pixel,self.nsamples,
                                    treename='deepntuplizer/tree', flat = False)

        
        import uproot3 as uproot
        #import uproot
        urfile = uproot.open(filename)["deepntuplizer/tree"]
        truth_arrays = urfile.arrays(self.truth_branches)
        truth = reduceTruth(truth_arrays)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!

        x_global = x_global.astype(dtype='float32', order='C')
        x_cpf = x_cpf.astype(dtype='float32', order='C')
        x_npf = x_npf.astype(dtype='float32', order='C')
        x_vtx = x_vtx.astype(dtype='float32', order='C')
        x_pixel = x_pixel.astype(dtype='float32', order='C')

        print("Check des shapes : ")
        print(truth.shape)
        print(x_global.shape)
        print(x_cpf.shape)
        print(x_npf.shape)
        print(x_vtx.shape)
        print(x_pixel.shape)
        
        if self.remove:
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            for_remove = root2array(
                filename,
                treename = "deepntuplizer/tree",
                stop = None,
                branches = b
            )
            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
            undef=for_remove['isUndefined']
            notremoves-=undef
            pu=for_remove['isPU']
            notremoves-=pu
            print('took ', sw.getAndReset(), ' to create remove indices')

#        random_pick = np.random.rand(x_global.shape[0])
 #       selection1 = (x_global[:,0] <= 80)
  #      selection2 = (x_global[:,0] > 80) & (x_global[:,0] <= 140)
   #     selection3 = (x_global[:,0] > 140) & (x_global[:,0] <= 186)
    #    selection4 = (x_global[:,0] > 186) & (x_global[:,0] <= 247)

#        notremoves[selection1] -= 4*random_pick[selection1]
 #       notremoves[selection2] -= 2.667*random_pick[selection2]
  #      notremoves[selection3] -= 2*random_pick[selection3]
   #     notremoves[selection4] -= 1*random_pick[selection4]

        if self.remove:
            print('remove')
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_vtx=x_vtx[notremoves > 0]
            x_pixel=x_pixel[notremoves > 0]
            truth=truth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        
        print('remove nans')
        x_global = np.where(np.isfinite(x_global) , x_global, 0)
        x_cpf = np.where(np.isfinite(x_cpf), x_cpf, 0)
        x_npf = np.where(np.isfinite(x_npf), x_npf, 0)
        x_vtx = np.where(np.isfinite(x_vtx), x_vtx, 0)
        x_pixel = np.where(np.isfinite(x_pixel), x_pixel, 0)
        
        return [x_global,x_cpf,x_npf,x_vtx, x_pixel], [truth], []

    # defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), features[0][:,0:2].transpose())),
                                        #names='prob_isB, prob_isBB, prob_isLeptB, prob_isC, prob_isUDS, prob_isG, isB, isBB, isLeptB, isC, isUDS, isG, jet_pt, jet_eta')
                                         names='prob_isB, prob_isC, prob_isUDS, prob_isG, isB, isC, isUDS, isG, jet_pt, jet_eta')
                                        #names='prob_isB, prob_isBB,prob_isLeptB,prob_isC,prob_isUDS,prob_isG,isB,isBB,isLeptB,isC,isUDS,isG,jet_pt, jet_eta')
        array2root(out, outfilename, 'tree')

    ### This back up code is available if you don't define global vars of your jet as inputs ### 

    # defines how to write out the prediction
    #def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
     #   # predicted will be a list
      #  spectator_branches = ['jet_pt','jet_eta']
       # from root_numpy import array2root
        #if inputfile[-5:] == 'djctd':
         #               print("storing normed pt and eta... run on root files if you want something else for now")
          #              spectators = features[0][:,0:2].transpose()
        #else:
         #   import uproot3 as uproot
          #  print(inputfile)
           # urfile = uproot.open(inputfile)["deepntuplizer/tree"]
            #spectator_arrays = urfile.arrays(spectator_branches)
            #print(spectator_arrays)
            #spectators = [spectator_arrays[a.encode()] for a in spectator_branches]
        
        #out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), spectators) ),
         #                               names='prob_isB, prob_isBB,prob_isLeptB, prob_isC,prob_isUDS,prob_isG,isB, isBB, isLeptB, isC,isUDS,isG,jet_pt, jet_eta')
        #array2root(out, outfilename, 'tree')


class TrainData_ParticleNet(TrainData):
    def __init__(self):

        TrainData.__init__(self)        
        
        self.description = "ParticleNet inputs"
        
        self.truth_branches = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isGCC','isCC','isUD','isS','isG']
        self.undefTruth=['isUndefined', 'isPU']
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        self.remove = True
        self.referenceclass= 'isB'  #'flatten'  #Choose 'flatten' for flat or one of the truth branch for ref
        self.red_classes = ['cat_B','cat_C','cat_UDS','cat_G'] #reduced classes (flat only)
        self.truth_red_fusion = [('isB','isBB','isGBB','isLeptonicB','isLeptonicB_C'),('isC','isGCC','isCC'),('isUD','isS'),('isG')] #Indicates how you are making the fusion of your truth branches to the reduced classes for the flat reweighting
        self.class_weights = np.array([1.00,1.00,2.5,5.0], dtype=float)  #Ratio between our reduced classes (flat only)
        #self.weight_binX = np.array([15, 20, 26, 35, 46, 61, 80, 106, 141, 186, 247, 326, 432, 571, 756, 1000],dtype=float) #Flat reweighting
        self.weight_binX = np.array([100,125,150,175,200,250,300,400,500,600,2000],dtype=float)
        #self.weight_binX = (np.exp(np.linspace(np.log(15), np.log(1000), 16))).tolist()
        #self.weight_binX = np.array([10,25,30,35,40,45,50,60,75,100,125,150,175,200,250,300,400,500,600,2000],dtype=float) #Ref reweighting
        self.weight_binY = np.array(
            [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float) #Flat reweighting
        #self.weight_binY = np.array([-2.5,-1.136,-0.516,-0.235,-0.107,-0.0484,-0.0220, 0.0220, 0.0484, 0.107, 0.235,0.516, 1.136, 2.5], dtype=float) #Flat-hybrid reweighting
         #self.weight_binY = np.array(
         #   [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float) #Ref reweighting

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
                
        
        self.cpf_branches = [#'Cpfcan_ptlog', 'Cpfcan_elog', 'Cpfcan_eta',
                             'Cpfcan_BtagPf_trackEtaRel',
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
                             'Cpfcan_VTX_ass',
                             'Cpfcan_puppiw',
                             'Cpfcan_chi2',
                             'Cpfcan_quality']

        self.n_cpf = 25

        self.npf_branches = ['Npfcan_ptrel', 'Npfcan_etarel', 'Npfcan_phirel', 'Npfcan_deltaR','Npfcan_isGamma','Npfcan_HadFrac','Npfcan_drminsv','Npfcan_puppiw']
#        self.npf_branches = ['Npfcan_ptlog', 'Npfcan_elog', 'Npfcan_eta', 'Npfcan_etarel', 'Npfcan_phirel', 'Npfcan_deltaR','Npfcan_isGamma','Npfcan_HadFrac','Npfcan_drminsv','Npfcan_puppiw']
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
        
        self.cpf_pts_branches = ['Cpfcan_etarel',
                                 'Cpfcan_phirel']
        
        self.npf_pts_branches = ['Npfcan_etarel', 
                                 'Npfcan_phirel']
        
        self.vtx_pts_branches = ['sv_etarel',
                                 'sv_phirel']

#        self.pred_branches = ['pfDeepFlavourJetTags_probb','pfDeepFlavourJetTags_problepb',
 #                             'pfDeepFlavourJetTags_probbb','pfDeepFlavourJetTags_probc',
  #                            'pfDeepFlavourJetTags_probuds','pfDeepFlavourJetTags_probg',
   #                           'pfParticleNetAK4JetTags_probb','pfParticleNetAK4JetTags_probbb',
    #                          'pfParticleNetAK4JetTags_probc','pfParticleNetAK4JetTags_probcc',
     #                         'pfParticleNetAK4JetTags_probuds','pfParticleNetAK4JetTags_probg']

        #self.reduced_truth = ['isB','isBB','isLeptonicB','isC','isUDS','isG']
        self.reduced_truth = ['isB','isC','isUDS','isG']
        
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
        import ROOT
        from root_numpy import tree2array, root2array
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                nparray = root2array(
                    fname,
                    treename = "deepntuplizer/tree",
                    stop = None,
                    branches = branches
                )
                norm_hist = True
                if self.referenceclass == 'flatten':
                    norm_hist = False
                weighter.addDistributions(nparray, norm_h = norm_hist)
                #del nparray
                counter=counter+1
            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)
            weighter.printHistos('/afs/cern.ch/user/a/ademoor/Flatten/') #If you need to print the 2D histo, choose your output dir
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
            
            b = uproot_arrays[b'isB']
            
            bb = uproot_arrays[b'isBB']
            gbb = uproot_arrays[b'isGBB']
            
            bl = uproot_arrays[b'isLeptonicB']
            blc = uproot_arrays[b'isLeptonicB_C']
            lepb = bl+blc
            
            c = uproot_arrays[b'isC']
            cc = uproot_arrays[b'isCC']
            gcc = uproot_arrays[b'isGCC']
            
            ud = uproot_arrays[b'isUD']
            s = uproot_arrays[b'isS']
            uds = ud+s
            
            g = uproot_arrays[b'isG']
            
            #return np.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()
            return np.vstack((b+bb+gbb+lepb,c+cc+gcc,uds,g)).transpose()
        
        print('reading '+filename)
        
        import ROOT
        from root_numpy import tree2array, root2array
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()
        
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles
        
#        x_global = MeanNormZeroPad(filename,None,
 #                                  [self.global_branches],
  #                                 [1],self.nsamples)

#        x_cpf = MeanNormZeroPadParticles(filename,None,
 #                                  self.cpf_branches,
  #                                 self.n_cpf,self.nsamples)

#        x_npf = MeanNormZeroPadParticles(filename,None,
 #                                        self.npf_branches,
  #                                       self.n_npf,self.nsamples)

#        x_vtx = MeanNormZeroPadParticles(filename,None,
 #                                        self.vtx_branches,
  #                                       self.n_vtx,self.nsamples)
        
#        cpf_pts = MeanNormZeroPadParticles(filename,None,
 #                                        self.cpf_pts_branches,
  #                                       self.n_cpf,self.nsamples)
        
#        npf_pts = MeanNormZeroPadParticles(filename,None,
 #                                        self.npf_pts_branches,
  #                                       self.n_npf,self.nsamples)
        
#        vtx_pts = MeanNormZeroPadParticles(filename,None,
 #                                        self.vtx_pts_branches,
  #                                       self.n_vtx,self.nsamples)

  #      preds = MeanNormZeroPad(filename,None,
 #                                  [self.pred_branches],
#                                   [1],self.nsamples)

        x_global = uproot_tree_to_numpy(filename,
                                        self.global_branches,1,self.nsamples,
                                        treename='deepntuplizer/tree', flat = True)

        x_cpf = uproot_tree_to_numpy(filename,
                                     self.cpf_branches,self.n_cpf,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        x_npf = uproot_tree_to_numpy(filename,
                                         self.npf_branches,self.n_npf,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        x_vtx = uproot_tree_to_numpy(filename,
                                         self.vtx_branches,self.n_vtx,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        cpf_pts = uproot_tree_to_numpy(filename,
                                        self.cpf_pts_branches,self.n_cpf,self.nsamples,
                                        treename='deepntuplizer/tree', flat = True)

        npf_pts = uproot_tree_to_numpy(filename,
                                     self.npf_pts_branches,self.n_npf,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        vtx_pts = uproot_tree_to_numpy(filename,
                                         self.vtx_pts_branches,self.n_vtx,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

#        preds = uproot_tree_to_numpy(filename,
 #                                        self.pred_branches,1,self.nsamples,
  #                                   treename='deepntuplizer/tree', flat = False)



        import uproot3 as uproot
        urfile = uproot.open(filename)["deepntuplizer/tree"]
        truth_arrays = urfile.arrays(self.truth_branches)
        truth = reduceTruth(truth_arrays)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!


        x_global = x_global.astype(dtype='float32', order='C')
        x_cpf = x_cpf.astype(dtype='float32', order='C')
        x_npf = x_npf.astype(dtype='float32', order='C')
        x_vtx = x_vtx.astype(dtype='float32', order='C')
        cpf_pts = cpf_pts.astype(dtype='float32', order='C')
        npf_pts = npf_pts.astype(dtype='float32', order='C')
        vtx_pts = vtx_pts.astype(dtype='float32', order='C')
#        preds = preds.astype(dtype='float32', order='C')
        
        if self.remove:
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            for_remove = root2array(
                filename,
                treename = "deepntuplizer/tree",
                stop = None,
                branches = b
            )
            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
            undef=for_remove['isUndefined']
            notremoves-=undef
            pu=for_remove['isPU']
            notremoves-=pu
            print('took ', sw.getAndReset(), ' to create remove indices')

#        random_pick = np.random.rand(x_global.shape[0])
 #       selection1 = (x_global[:,0] <= 80)
  #      selection2 = (x_global[:,0] > 80) & (x_global[:,0] <= 140)
   #     selection3 = (x_global[:,0] > 140) & (x_global[:,0] <= 186)
    #    selection4 = (x_global[:,0] > 186) & (x_global[:,0] <= 247)

#        notremoves[selection1] -= 4*random_pick[selection1]
 #       notremoves[selection2] -= 2.667*random_pick[selection2]
  #      notremoves[selection3] -= 2*random_pick[selection3]
   #     notremoves[selection4] -= 1*random_pick[selection4]

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
#            preds = preds[notremoves > 0]

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
#        preds = np.where(np.isfinite(preds), preds, 0)
        
        return [x_global,x_cpf,x_npf,x_vtx, cpf_pts, npf_pts, vtx_pts], [truth], []

    # defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), features[0][:,0:2].transpose())),
                                        names='prob_isB, prob_isC, prob_isUDS, prob_isG, isB, isC, isUDS, isG, jet_pt, jet_eta')
                                        #names='prob_isB, prob_isBB, prob_isLeptB, prob_isC, prob_isUDS, prob_isG, isB, isBB, isLeptB, isC, isUDS, isG, jet_pt, jet_eta')
                                        
        array2root(out, outfilename, 'tree')

    ### This back up code is available if you don't define global vars of your jet as inputs ### 

    # defines how to write out the prediction
    #def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
     #   # predicted will be a list
      #  spectator_branches = ['jet_pt','jet_eta']
       # from root_numpy import array2root
        #if inputfile[-5:] == 'djctd':
         #               print("storing normed pt and eta... run on root files if you want something else for now")
          #              spectators = features[0][:,0:2].transpose()
        #else:
         #   import uproot3 as uproot
          #  print(inputfile)
           # urfile = uproot.open(inputfile)["deepntuplizer/tree"]
            #spectator_arrays = urfile.arrays(spectator_branches)
            #print(spectator_arrays)
            #spectators = [spectator_arrays[a.encode()] for a in spectator_branches]
        
        #out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), spectators) ),
         #                               names='prob_isB, prob_isBB,prob_isLeptB, prob_isC,prob_isUDS,prob_isG,isB, isBB, isLeptB, isC,isUDS,isG,jet_pt, jet_eta')
        #array2root(out, outfilename, 'tree')


class TrainData_DFE(TrainData):
    def __init__(self):

        TrainData.__init__(self)

        
        self.description = "DeepJet + Edge-Conv coord"
        
        self.truth_branches = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isGCC','isCC','isUD','isS','isG']
        self.undefTruth=['isUndefined']
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        self.remove = True
        self.referenceclass='isB'
        self.weight_binX = np.array([
            10,25,30,35,40,45,50,60,75,100,
            125,150,175,200,250,300,400,500,
            600,2000],dtype=float)
        
        self.weight_binY = np.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
        )

        #self.global_branches = ['jet_pt', 'jet_eta',
         #                       'nCpfcand','nNpfcand',
          #                      'nsv','npv',
           #                     'TagVarCSV_trackSumJetEtRatio',
            #                    'TagVarCSV_trackSumJetDeltaR',
             #                   'TagVarCSV_vertexCategory',
              #                  'TagVarCSV_trackSip2dValAboveCharm',
               #                 'TagVarCSV_trackSip2dSigAboveCharm',
                #                'TagVarCSV_trackSip3dValAboveCharm',
                 #               'TagVarCSV_trackSip3dSigAboveCharm',
                  #              'TagVarCSV_jetNSelectedTracks',
                   #             'TagVarCSV_jetNTracksEtaRel']
                
        
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
                             'Cpfcan_VTX_ass',
                             'Cpfcan_puppiw',
                             'Cpfcan_chi2',
                             'Cpfcan_quality']
        self.n_cpf = 25

        self.npf_branches = ['Npfcan_ptrel', 'Npfcan_etarel', 'Npfcan_phirel', 'Npfcan_deltaR','Npfcan_isGamma','Npfcan_HadFrac','Npfcan_drminsv','Npfcan_puppiw']
        self.n_npf = 25
        
        self.vtx_branches = ['sv_pt','sv_deltaR',
                             'sv_mass',
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

        self.n_vtx = 4
        
        self.cpf_pts_branches = ['Cpfcan_etarel',
                                 'Cpfcan_phirel']
        
        self.npf_pts_branches = ['Npfcan_etarel', 
                                 'Npfcan_phirel']
        
        self.n_pts = 25
        
        self.reduced_truth = ['isB','isBB','isLeptonicB','isC','isUDS','isG']

        
    def createWeighterObjects(self, allsourcefiles):
        # 
        # Calculates the weights needed for flattening the pt/eta spectrum
        
        from DeepJetCore.Weighter import Weighter
        weighter = Weighter()
        weighter.undefTruth = self.undefTruth
        branches = [self.weightbranchX,self.weightbranchY]
        branches.extend(self.truth_branches)

        if self.remove:
            weighter.setBinningAndClasses(
                [self.weight_binX,self.weight_binY],
                self.weightbranchX,self.weightbranchY,
                self.truth_branches
            )

        
        counter=0
        import ROOT
        from root_numpy import tree2array, root2array
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                nparray = root2array(
                    fname,
                    treename = "deepntuplizer/tree",
                    stop = None,
                    branches = branches
                )
                weighter.addDistributions(nparray)
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
            
            b = uproot_arrays[b'isB']
            
            bb = uproot_arrays[b'isBB']
            gbb = uproot_arrays[b'isGBB']
            
            bl = uproot_arrays[b'isLeptonicB']
            blc = uproot_arrays[b'isLeptonicB_C']
            lepb = bl+blc
            
            c = uproot_arrays[b'isC']
            cc = uproot_arrays[b'isCC']
            gcc = uproot_arrays[b'isGCC']
            
            ud = uproot_arrays[b'isUD']
            s = uproot_arrays[b'isS']
            uds = ud+s
            
            g = uproot_arrays[b'isG']
            
            return np.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()
        
        print('reading '+filename)
        
        import ROOT
        from root_numpy import tree2array, root2array
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()

        
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles
        
        #x_global = MeanNormZeroPad(filename,None,
         #                          [self.global_branches],
          #                         [1],self.nsamples)

        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.cpf_branches,
                                   self.n_cpf,self.nsamples)

        x_npf = MeanNormZeroPadParticles(filename,None,
                                         self.npf_branches,
                                         self.n_npf,self.nsamples)

        x_vtx = MeanNormZeroPadParticles(filename,None,
                                         self.vtx_branches,
                                         self.n_vtx,self.nsamples)
        
        cpf_pts = MeanNormZeroPadParticles(filename,None,
                                         self.cpf_pts_branches,
                                         self.n_pts,self.nsamples)
        
        npf_pts = MeanNormZeroPadParticles(filename,None,
                                         self.npf_pts_branches,
                                         self.n_pts,self.nsamples)

        import uproot
        urfile = uproot.open(filename)["deepntuplizer/tree"]
        truth_arrays = urfile.arrays(self.truth_branches)
        truth = reduceTruth(truth_arrays)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!

        #x_global = x_global.astype(dtype='float32', order='C')
        x_cpf = x_cpf.astype(dtype='float32', order='C')
        x_npf = x_npf.astype(dtype='float32', order='C')
        x_vtx = x_vtx.astype(dtype='float32', order='C')
        cpf_pts = cpf_pts.astype(dtype='float32', order='C')
        npf_pts = npf_pts.astype(dtype='float32', order='C')
        
        if self.remove:
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            for_remove = root2array(
                filename,
                treename = "deepntuplizer/tree",
                stop = None,
                branches = b
            )
            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
            undef=for_remove['isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')


        if self.remove:
            print('remove')
            #x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_vtx=x_vtx[notremoves > 0]
            cpf_pts=cpf_pts[notremoves > 0]
            npf_pts=npf_pts[notremoves > 0]
            truth=truth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')

        
        print('remove nans')
        #x_global = np.where(np.isfinite(x_global) , x_global, 0)
        x_cpf = np.where(np.isfinite(x_cpf), x_cpf, 0)
        x_npf = np.where(np.isfinite(x_npf), x_npf, 0)
        x_vtx = np.where(np.isfinite(x_vtx), x_vtx, 0)
        cpf_pts = np.where(np.isfinite(cpf_pts), cpf_pts, 0)
        npf_pts = np.where(np.isfinite(npf_pts), npf_pts, 0)

        return [x_cpf,x_npf,x_vtx, cpf_pts, npf_pts], [truth], []
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), features[0][:,0:2].transpose() ) ),
                                         names='prob_isB, prob_isBB,prob_isLeptB, prob_isC,prob_isUDS,prob_isG,isB, isBB, isLeptB, isC,isUDS,isG,jet_pt, jet_eta')
        array2root(out, outfilename, 'tree')


class TrainData_DF(TrainData):
    def __init__(self):
        TrainData.__init__(self)

        
        self.description = "DeepJet training datastructure"
        
        self.truth_branches = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isGCC','isCC','isUD','isS','isG']
        self.undefTruth=['isUndefined']
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        self.remove = True
        self.referenceclass='isB'
        self.weight_binX = np.array([
            10,25,30,35,40,45,50,60,75,100,
            125,150,175,200,250,300,400,500,
            600,2000],dtype=float)
        self.weight_binY = np.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
        )

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
                             'Cpfcan_VTX_ass',
                             'Cpfcan_puppiw',
                             'Cpfcan_chi2',
                             'Cpfcan_quality']
        self.n_cpf = 25

        self.npf_branches = ['Npfcan_ptrel','Npfcan_deltaR','Npfcan_isGamma','Npfcan_HadFrac','Npfcan_drminsv','Npfcan_puppiw']
        self.n_npf = 25
        
        self.vtx_branches = ['sv_pt','sv_deltaR',
                             'sv_mass',
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
        
        self.reduced_truth = ['isB','isBB','isLeptonicB','isC','isUDS','isG']

        
    def createWeighterObjects(self, allsourcefiles):
        # 
        # Calculates the weights needed for flattening the pt/eta spectrum
        
        from DeepJetCore.Weighter import Weighter
        weighter = Weighter()
        weighter.undefTruth = self.undefTruth
        branches = [self.weightbranchX,self.weightbranchY]
        branches.extend(self.truth_branches)

        if self.remove:
            weighter.setBinningAndClasses(
                [self.weight_binX,self.weight_binY],
                self.weightbranchX,self.weightbranchY,
                self.truth_branches
            )

        
        counter=0
        import ROOT
        from root_numpy import tree2array, root2array
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                nparray = root2array(
                    fname,
                    treename = "deepntuplizer/tree",
                    stop = None,
                    branches = branches
                )
                weighter.addDistributions(nparray)
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
            
            b = uproot_arrays[b'isB']
            
            bb = uproot_arrays[b'isBB']
            gbb = uproot_arrays[b'isGBB']
            
            bl = uproot_arrays[b'isLeptonicB']
            blc = uproot_arrays[b'isLeptonicB_C']
            lepb = bl+blc
            
            c = uproot_arrays[b'isC']
            cc = uproot_arrays[b'isCC']
            gcc = uproot_arrays[b'isGCC']
            
            ud = uproot_arrays[b'isUD']
            s = uproot_arrays[b'isS']
            uds = ud+s
            
            g = uproot_arrays[b'isG']
            
            return np.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()
        
        print('reading '+filename)
        
        import ROOT
        from root_numpy import tree2array, root2array
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()

        
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles
        
        '''
        x_global = MeanNormZeroPad(filename,None,
                                   [self.global_branches],
                                   [1],self.nsamples)

        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.cpf_branches,
                                   self.n_cpf,self.nsamples)

        x_npf = MeanNormZeroPadParticles(filename,None,
                                         self.npf_branches,
                                         self.n_npf,self.nsamples)

        x_vtx = MeanNormZeroPadParticles(filename,None,
                                         self.vtx_branches,
                                         self.n_vtx,self.nsamples)
        '''

        x_global = uproot_tree_to_numpy(filename,
                                        self.global_branches,1,self.nsamples,
                                        treename='deepntuplizer/tree', flat = True)

        x_cpf = uproot_tree_to_numpy(filename,
                                     self.cpf_branches,self.n_cpf,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        x_npf = uproot_tree_to_numpy(filename,
                                         self.npf_branches,self.n_npf,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        x_vtx = uproot_tree_to_numpy(filename,
                                         self.vtx_branches,self.n_vtx,self.nsamples,
                                     treename='deepntuplizer/tree', flat = False)

        
        import uproot3 as uproot
        urfile = uproot.open(filename)["deepntuplizer/tree"]
        truth_arrays = urfile.arrays(self.truth_branches)
        truth = reduceTruth(truth_arrays)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!

        x_global = x_global.astype(dtype='float32', order='C')
        x_cpf = x_cpf.astype(dtype='float32', order='C')
        x_npf = x_npf.astype(dtype='float32', order='C')
        x_vtx = x_vtx.astype(dtype='float32', order='C')


        
        if self.remove:
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            for_remove = root2array(
                filename,
                treename = "deepntuplizer/tree",
                stop = None,
                branches = b
            )
            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
            undef=for_remove['isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')


        if self.remove:
            print('remove')
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_vtx=x_vtx[notremoves > 0]
            truth=truth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')

        
        print('remove nans')
        x_global = np.where(np.isfinite(x_global) , x_global, 0)
        x_cpf = np.where(np.isfinite(x_cpf), x_cpf, 0)
        x_npf = np.where(np.isfinite(x_npf), x_npf, 0)
        x_vtx = np.where(np.isfinite(x_vtx), x_vtx, 0)

        return [x_global,x_cpf,x_npf,x_vtx], [truth], []
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), features[0][:,0:2].transpose() ) ),
                                         names='prob_isB, prob_isBB,prob_isLeptB, prob_isC,prob_isUDS,prob_isG,isB, isBB, isLeptB, isC,isUDS,isG,jet_pt, jet_eta')
        array2root(out, outfilename, 'tree')



class TrainData_DeepCSV(TrainData):
    def __init__(self):

        TrainData.__init__(self)

        self.description = "DeepCSV training datastructure"
       
        self.truth_branches = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isGCC','isCC','isUD','isS','isG']
        self.undefTruth=['isUndefined']
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        self.remove = True
        self.referenceclass='isB'
        self.weight_binX = np.array([
            10,25,30,35,40,45,50,60,75,100,
            125,150,175,200,250,300,400,500,
            600,2000],dtype=float)
        
        self.weight_binY = np.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
        )

        self.global_branches = ['jet_pt', 'jet_eta',
                                'TagVarCSV_jetNSecondaryVertices', 
                                'TagVarCSV_trackSumJetEtRatio',
                                'TagVarCSV_trackSumJetDeltaR',
                                'TagVarCSV_vertexCategory',
                                'TagVarCSV_trackSip2dValAboveCharm',
                                'TagVarCSV_trackSip2dSigAboveCharm',
                                'TagVarCSV_trackSip3dValAboveCharm',
                                'TagVarCSV_trackSip3dSigAboveCharm',
                                'TagVarCSV_jetNSelectedTracks',
                                'TagVarCSV_jetNTracksEtaRel']

        self.track_branches = ['TagVarCSVTrk_trackJetDistVal',
                              'TagVarCSVTrk_trackPtRel', 
                              'TagVarCSVTrk_trackDeltaR', 
                              'TagVarCSVTrk_trackPtRatio', 
                              'TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig', 
                              'TagVarCSVTrk_trackDecayLenVal']
        self.n_track = 6
        
        self.eta_rel_branches = ['TagVarCSV_trackEtaRel']
        self.n_eta_rel = 4

        self.vtx_branches = ['TagVarCSV_vertexMass', 
                          'TagVarCSV_vertexNTracks', 
                          'TagVarCSV_vertexEnergyRatio',
                          'TagVarCSV_vertexJetDeltaR',
                          'TagVarCSV_flightDistance2dVal', 
                          'TagVarCSV_flightDistance2dSig', 
                          'TagVarCSV_flightDistance3dVal', 
                          'TagVarCSV_flightDistance3dSig']
        self.n_vtx = 1
                
        self.reduced_truth = ['isB','isBB','isC','isUDSG']

    def readTreeFromRootToTuple(self, filenames, limit=None, branches=None):
        '''
        To be used to get the initial tupel for further processing in inherting classes
        Makes sure the number of entries is properly set
        
        can also read a list of files (e.g. to produce weights/removes from larger statistics
        (not fully tested, yet)
        '''
        
        if branches is None or len(branches) == 0:
            return np.array([],dtype='float32')
            
        #print(branches)
        #remove duplicates
        usebranches=list(set(branches))
        tmpbb=[]
        for b in usebranches:
            if len(b):
                tmpbb.append(b)
        usebranches=tmpbb
            
        import ROOT
        from root_numpy import tree2array, root2array
        if isinstance(filenames, list):
            for f in filenames:
                fileTimeOut(f,120)
            print('add files')
            nparray = root2array(
                filenames, 
                treename = "deepntuplizer/tree", 
                stop = limit,
                branches = usebranches
                )
            print('done add files')
            return nparray
            print('add files')
        else:    
            fileTimeOut(filenames,120) #give eos a minute to recover
            rfile = ROOT.TFile(filenames)
            tree = rfile.Get(self.treename)
            if not self.nsamples:
                self.nsamples=tree.GetEntries()
            nparray = tree2array(tree, stop=limit, branches=usebranches)
            return nparray
        
    def createWeighterObjects(self, allsourcefiles):
        # 
        # Calculates the weights needed for flattening the pt/eta spectrum
        
        from DeepJetCore.Weighter import Weighter
        weighter = Weighter()
        weighter.undefTruth = self.undefTruth
        branches = [self.weightbranchX,self.weightbranchY]
        branches.extend(self.truth_branches)

        if self.remove:
            weighter.setBinningAndClasses(
                [self.weight_binX,self.weight_binY],
                self.weightbranchX,self.weightbranchY,
                self.truth_branches
            )

        
        counter=0
        import ROOT
        from root_numpy import tree2array, root2array
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                nparray = root2array(
                    fname,
                    treename = "deepntuplizer/tree",
                    stop = None,
                    branches = branches
                )
                weighter.addDistributions(nparray)
                #del nparray
                counter=counter+1
            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)

        print("calculate means")
        from DeepJetCore.preprocessing import meanNormProd
        nparray = self.readTreeFromRootToTuple(allsourcefiles,branches=self.vtx_branches+self.eta_rel_branches+self.track_branches+self.global_branches, limit=500000)
        for a in (self.vtx_branches+self.eta_rel_branches+self.track_branches+self.global_branches):
            for b in range(len(nparray[a])):
                nparray[a][b] = np.where(nparray[a][b] < 100000.0, nparray[a][b], 0)
        means = np.array([],dtype='float32')
        if len(nparray):
            means = meanNormProd(nparray)
        return {'weigther':weighter,'means':means}
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):

        # Function to produce the numpy training arrays from root files

        from DeepJetCore.Weighter import Weighter
        from DeepJetCore.stopwatch import stopwatch
        sw=stopwatch()
        swall=stopwatch()
        if not istraining:
            self.remove = False
                
        def reduceTruth(uproot_arrays):
            
            b = uproot_arrays[b'isB']
            
            bb = uproot_arrays[b'isBB']
            gbb = uproot_arrays[b'isGBB']
            
            bl = uproot_arrays[b'isLeptonicB']
            blc = uproot_arrays[b'isLeptonicB_C']
            lepb = bl+blc
            
            c = uproot_arrays[b'isC']
            cc = uproot_arrays[b'isCC']
            gcc = uproot_arrays[b'isGCC']
            
            ud = uproot_arrays[b'isUD']
            s = uproot_arrays[b'isS']
            uds = ud+s
            
            g = uproot_arrays[b'isG']
            
            return np.vstack((b+lepb,bb+gbb,c+cc+gcc,uds+g)).transpose()
        
        print('reading '+filename)
        
        import ROOT
        from root_numpy import tree2array, root2array
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()  
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles
        x_global = MeanNormZeroPad(filename,weighterobjects['means'],
                                   [self.global_branches,self.track_branches,self.eta_rel_branches,self.vtx_branches],
                                   [1,self.n_track,self.n_eta_rel,self.n_vtx],self.nsamples)
                
        import uproot
        urfile = uproot.open(filename)["deepntuplizer/tree"]
        truth_arrays = urfile.arrays(self.truth_branches)
        truth = reduceTruth(truth_arrays)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!

        x_global = x_global.astype(dtype='float32', order='C')
        
        
        if self.remove:
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            for_remove = root2array(
                filename,
                treename = "deepntuplizer/tree",
                stop = None,
                branches = b
            )
            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
            undef=for_remove['isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')


        if self.remove:
            print('remove')
            x_global=x_global[notremoves > 0]
            truth=truth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')

        
        print('remove nans')
        x_global = np.where(np.isfinite(x_global)+(x_global < 100000.0), x_global, 0)
        return [x_global], [truth], []
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        spectator_branches = ['jet_pt','jet_eta']
        from root_numpy import array2root
        if inputfile[-5:] == 'djctd':
                        print("storing normed pt and eta... run on root files if you want something else for now")
                        spectators = features[0][:,0:2].transpose()
        else:
            import uproot
            print(inputfile)
            urfile = uproot.open(inputfile)["deepntuplizer/tree"]
            spectator_arrays = urfile.arrays(spectator_branches)
            print(spectator_arrays)
            spectators = [spectator_arrays[a.encode()] for a in spectator_branches]
        
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), spectators) ),
                                         names='prob_isB, prob_isBB, prob_isC,prob_isUDSG,isB, isBB, isC,isUDSG,jet_pt, jet_eta')
        array2root(out, outfilename, 'tree')
