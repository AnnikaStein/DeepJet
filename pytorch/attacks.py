from definitions import *

def apply_noise(sample, magn=1e-2,offset=[0], dev=torch.device("cpu"), restrict_impact=-1, var_group='glob'):
    if magn == 0:
        return sample
    
    seed = 0
    np.random.seed(seed)
        
    n_Vars = cands_per_variable[var_group] * vars_per_candidate[var_group]

    with torch.no_grad():
        noise = torch.Tensor(np.random.normal(offset,magn,(len(sample),n_Vars))).to(dev)
        xadv = sample + noise
        
        # use full indices and check if in int.vars. or defaults
        for i in range(n_Vars):
            if i in integer_per_variable[var_group]:
                xadv[:,i] = sample[:,i]
            else: # non integer, but might have defaults that should be excluded from shift
                defaults = sample[:,i].cpu() == defaults_per_variable[var_group][i]
                if torch.sum(defaults) != 0:
                    xadv[:,i][defaults] = sample[:,i][defaults]
                    
                if restrict_impact > 0:
                    difference = xadv[:,i] - sample[:,i]
                    allowed_perturbation = restrict_impact * torch.abs(sample[:,i])
                    high_impact = torch.abs(difference) > allowed_perturbation
                    
                    if np.sum(high_impact)!=0:
                        xadv[high_impact,i] = sample[high_impact,i] + allowed_perturbation[high_impact] * torch.sign(noise[high_impact,i])

        return xadv

def fgsm_attack(epsilon=1e-2,sample=None,targets=None,thismodel=None,thiscriterion=None,reduced=True, dev=torch.device("cpu"), restrict_impact=-1):
    if epsilon == 0:
        return sample
    
    glob, cpf, npf, vtx = sample
    xadv_glob = glob.clone().detach()
    xadv_cpf = cpf.clone().detach()
    xadv_npf = npf.clone().detach()
    xadv_vtx = vtx.clone().detach()
    
    n_Vars_glob = cands_per_variable['glob'] * vars_per_candidate['glob']
    n_Vars_cpf = cands_per_variable['cpf'] * vars_per_candidate['cpf']
    n_Vars_npf = cands_per_variable['npf'] * vars_per_candidate['npf']
    n_Vars_vtx = cands_per_variable['vtx'] * vars_per_candidate['vtx']
    
    xadv_glob.requires_grad = True
    xadv_cpf.requires_grad = True
    xadv_npf.requires_grad = True
    xadv_vtx.requires_grad = True
    
    preds = thismodel(xadv_glob,xadv_cpf,xadv_npf,xadv_vtx)
    
    loss = thiscriterion(preds, targets).mean()
    
    thismodel.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        dx_glob = torch.sign(xadv_glob.grad.detach())
        dx_cpf = torch.sign(xadv_cpf.grad.detach())
        dx_npf = torch.sign(xadv_npf.grad.detach())
        dx_vtx = torch.sign(xadv_vtx.grad.detach())
        
        xadv_glob += epsilon * dx_glob
        xadv_cpf += epsilon * dx_cpf
        xadv_npf += epsilon * dx_npf
        xadv_vtx += epsilon * dx_vtx
        
        if reduced:
            for i in range(n_Vars_glob):
                if i in integer_per_variable['glob']:
                    xadv_glob[:,i] = glob[:,i]
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults_glob = glob[:,i].cpu() == defaults_per_variable[var_group][i]
                    if torch.sum(defaults_glob) != 0:
                        xadv_glob[:,i][defaults_glob] = glob[:,i][defaults_glob]

                    if restrict_impact > 0:
                        difference = xadv_glob[:,i] - glob[:,i]
                        allowed_perturbation = restrict_impact * torch.abs(glob[:,i])
                        high_impact = torch.abs(difference) > allowed_perturbation
                        
                        if np.sum(high_impact)!=0:
                            xadv_glob[high_impact,i] = glob[high_impact,i] + allowed_perturbation[high_impact] * dx_glob[high_impact,i]
            
            for i in range(n_Vars_cpf):
                if i in integer_per_variable['cpf']:
                    xadv_cpf[:,i] = cpf[:,i]
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults_cpf = cpf[:,i].cpu() == defaults_per_variable[var_group][i]
                    if torch.sum(defaults_cpf) != 0:
                        xadv_cpf[:,i][defaults_cpf] = cpf[:,i][defaults_cpf]

                    if restrict_impact > 0:
                        difference = xadv_cpf[:,i] - cpf[:,i]
                        allowed_perturbation = restrict_impact * torch.abs(cpf[:,i])
                        high_impact = torch.abs(difference) > allowed_perturbation
                        
                        if np.sum(high_impact)!=0:
                            xadv_cpf[high_impact,i] = cpf[high_impact,i] + allowed_perturbation[high_impact] * dx_cpf[high_impact,i]        
                            
            for i in range(n_Vars_npf):
                if i in integer_per_variable['npf']:
                    xadv_npf[:,i] = npf[:,i]
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults_npf = npf[:,i].cpu() == defaults_per_variable[var_group][i]
                    if torch.sum(defaults_npf) != 0:
                        xadv_npf[:,i][defaults_npf] = npf[:,i][defaults_npf]

                    if restrict_impact > 0:
                        difference = xadv_npf[:,i] - npf[:,i]
                        allowed_perturbation = restrict_impact * torch.abs(npf[:,i])
                        high_impact = torch.abs(difference) > allowed_perturbation
                        
                        if np.sum(high_impact)!=0:
                            xadv_npf[high_impact,i] = npf[high_impact,i] + allowed_perturbation[high_impact] * dx_npf[high_impact,i]   
                            
            for i in range(n_Vars_vtx):
                if i in integer_per_variable['vtx']:
                    xadv_vtx[:,i] = vtx[:,i]
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults_vtx = vtx[:,i].cpu() == defaults_per_variable[var_group][i]
                    if torch.sum(defaults_vtx) != 0:
                        xadv_vtx[:,i][defaults_vtx] = vtx[:,i][defaults_vtx]

                    if restrict_impact > 0:
                        difference = xadv_vtx[:,i] - vtx[:,i]
                        allowed_perturbation = restrict_impact * torch.abs(vtx[:,i])
                        high_impact = torch.abs(difference) > allowed_perturbation
                        
                        if np.sum(high_impact)!=0:
                            xadv_vtx[high_impact,i] = vtx[high_impact,i] + allowed_perturbation[high_impact] * dx_vtx[high_impact,i]   
        
        return xadv_glob.detach(),xadv_cpf.detach(),xadv_npf.detach(),xadv_vtx.detach()
