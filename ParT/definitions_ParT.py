cands_per_variable = {
    'glob' : 1,
    'cpf' : 25,
    'npf' : 25,
    'vtx' : 5,
    'cpf_pts' : 25,
    'npf_pts' : 25,
    'vtx_pts' : 5,
    #'pxl' : ,
}
vars_per_candidate = {
    'glob' : 15,
    'cpf' : 17,
    'npf' : 8,
    'vtx' : 14,
    'cpf_pts' : 4, #10,
    'npf_pts' : 4,
    'vtx_pts' : 4,
    #'pxl' : ,
}
defaults_per_variable_before_prepro = {
    'glob' : [None,None,None,None,None,None,-999,-999,-999,-999,-999,-999,-999,None,None],
    'cpf' : [0 for i in range(vars_per_candidate['cpf'])],
    'npf' : [0 for i in range(vars_per_candidate['npf'])],
    'vtx' : [0 for i in range(vars_per_candidate['vtx'])],
    'cpf_pts' : [0 for i in range(vars_per_candidate['cpf_pts'])],
    'npf_pts' : [0 for i in range(vars_per_candidate['npf_pts'])],
    'vtx_pts' : [0 for i in range(vars_per_candidate['vtx_pts'])],
    #'pxl' : ,
}
epsilons_per_feature = {
    'glob' : '/eos/user/a/anstein/public/DeepJet/Train_ParT/auxiliary/global_epsilons.npy',
    'cpf' : '/eos/user/a/anstein/public/DeepJet/Train_ParT/auxiliary/cpf_epsilons.npy',
    'npf' : '/eos/user/a/anstein/public/DeepJet/Train_ParT/auxiliary/npf_epsilons.npy',
    'vtx' : '/eos/user/a/anstein/public/DeepJet/Train_ParT/auxiliary/vtx_epsilons.npy',
    'cpf_pts' : '/eos/user/a/anstein/public/DeepJet/Train_ParT/auxiliary/cpf_pts_epsilons.npy',
    'npf_pts' : '/eos/user/a/anstein/public/DeepJet/Train_ParT/auxiliary/npf_pts_epsilons.npy',
    'vtx_pts' : '/eos/user/a/anstein/public/DeepJet/Train_ParT/auxiliary/vtx_pts_epsilons.npy',
    #'pxl' : ,
}
defaults_per_variable = {
    'glob' : [0 for i in range(vars_per_candidate['glob'])],
    'cpf' : [0 for i in range(vars_per_candidate['cpf'])],
    'npf' : [0 for i in range(vars_per_candidate['npf'])],
    'vtx' : [0 for i in range(vars_per_candidate['vtx'])],
    'cpf_pts' : [0 for i in range(vars_per_candidate['cpf_pts'])],
    'npf_pts' : [0 for i in range(vars_per_candidate['npf_pts'])],
    'vtx_pts' : [0 for i in range(vars_per_candidate['vtx_pts'])],
    #'pxl' : ,
}
integer_variables_by_candidate = {
    'glob' : [2,3,4,5,8,13,14],
    'cpf' : [13,14,15,16],
    'npf' : [4],
    'vtx' : [5],
    'cpf_pts' : [],
    'npf_pts' : [],
    'vtx_pts' : [],
    #'pxl' : ,
}
