cands_per_variable = {
    'glob' : 1,
    'cpf' : 25,
    'npf' : 25,
    'vtx' : 5,
    #'pxl' : ,
}
vars_per_candidate = {
    'glob' : 15,
    'cpf' : 16,
    'npf' : 8,
    'vtx' : 14,
    #'pxl' : ,
}
defaults_per_variable = {
    'glob' : [0 for i in range(cands_per_variable['glob']*vars_per_candidate['glob'])],
    'cpf' : [0 for i in range(cands_per_variable['cpf']*vars_per_candidate['cpf'])],
    'npf' : [0 for i in range(cands_per_variable['npf']*vars_per_candidate['npf'])],
    'vtx' : [0 for i in range(cands_per_variable['vtx']*vars_per_candidate['vtx'])],
    #'pxl' : ,
}
integer_per_variable = {
    'glob' : [2,3,4,5,8,13,14],
    'cpf' : [-1], # ToDo
    'npf' : [-1], # ToDo
    'vtx' : [-1], # ToDo
    #'pxl' : ,
}