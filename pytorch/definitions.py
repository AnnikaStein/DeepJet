cands_per_variable = {
    'glob' : 1,
    'cpf' : 25,
    'npf' : 25,
#    'vtx' : 5,
    'vtx' : 4,
    #'pxl' : ,
}
vars_per_candidate = {
    'glob' : 15,
    'cpf' : 16,
#    'npf' : 8,
    'npf' : 6,
#    'vtx' : 14,
    'vtx' : 12,
    #'pxl' : ,
}
defaults_per_variable = {
    'glob' : [0 for i in range(vars_per_candidate['glob'])],
    'cpf' : [0 for i in range(vars_per_candidate['cpf'])],
    'npf' : [0 for i in range(vars_per_candidate['npf'])],
    'vtx' : [0 for i in range(vars_per_candidate['vtx'])],
    #'pxl' : ,
}
integer_variables_by_candidate = {
    'glob' : [2,3,4,5,8,13,14],
    'cpf' : [12,13,15],
#    'npf' : [4],
    'npf' : [2],
#    'vtx' : [5],
    'vtx' : [3],
    #'pxl' : ,
}
