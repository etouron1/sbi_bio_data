import torch
from sbi import utils as utils
from torch.distributions import MultivariateNormal
from itertools import combinations
from simulator import plot_C_genome
from sbi.inference import NPE, NLE, DirectPosterior
import random
import numpy as np
import pywt


import seaborn as sns
import matplotlib.pyplot as plt
torch.manual_seed(0)
def get_num_beads_and_start(chr_seq, sep):

    """ return a dict {chr : nb of beads}, a dict {chr : number of the start bead}, the total number of beads for all chr"""

    chr_bead = {}    # number of beads for each chromosome
    nbead = 0		# total number of beads for all chromosomes
    bead_start = {}  # bead label starts of a chr

    for i in chr_seq.keys(): # attention sorted()
        n = int(chr_seq[i] // sep) + int(chr_seq[i] % sep!=0)
        chr_bead[i] = n # number of beads for chromosome i
        
        nbead = nbead + n # total number of beads for all chromosmes
        bead_start[i] = nbead - n # the start bead for chr i
    return chr_bead, bead_start, nbead

def simulator(theta, resolution, sigma_spot, noisy):
        
    if sigma_spot=="variable":
        sig_2_simu = random.uniform(0.1, 10)
    else:
        sig_2_simu = sig_2_ref
               
    intensity_simu = 100

    C_simu = torch.zeros((nb_tot_bead,nb_tot_bead))

    for (chr_row, chr_col) in combinations(chr_seq.keys(),r=2):
        
        n_row = chr_seq[chr_row]//resolution
        n_col = chr_seq[chr_col]//resolution
        index_row = int(chr_row[chr_row.find("chr")+3:])-1
        index_col = int(chr_col[chr_col.find("chr")+3:])-1
        c_i_simu = theta[index_row]//resolution
        c_j_simu = theta[index_col]//resolution

        def simulator_1_bloc(n_row, n_col, c_i, c_j, sig_2, intensity, noisy=noisy):
            
            # Simulate a noisy matrix C_{n_row x n_col} with a gaussian spot at (c_i, c_j) of size sig_2 
            
            C = torch.zeros((n_row, n_col))
            
            distr = MultivariateNormal(torch.tensor([c_i, c_j]), sig_2*torch.eye(2))
                
            indices = torch.tensor([[(i, j) for j in range(len(C[0]))] for i in range(len(C))])
            C = intensity*torch.exp(distr.log_prob(torch.tensor(indices)))
            
            if noisy:
                mean = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2 
                sigma = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2
       
                noise = mean + sigma*torch.randn((n_row, n_col))

            else:
                noise = torch.zeros_like(C)
            
            return C+noise
        
        C_simu[start_bead[chr_row]:start_bead[chr_row]+nb_bead[chr_row]-1, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]-1] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)
    
    return C_simu

chr_seq = {"chr01": 230209, "chr02": 813179, "chr03": 316619}
chr_cen = {'chr01': 151584, 'chr02': 238325, 'chr03': 114499}

resolution= 3200
origin = "synthetic"

nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)

dim = 3
sig_2_ref = 1.0
#nb_train = 5000
nb_train = 10
nb_posterior_samples = 10000
prior_range = torch.tensor([230209,813179, 316619])
prior = utils.BoxUniform(torch.ones(dim), prior_range-1)

sigma_spot = "variable"
noisy = 1

if origin == "true":
    C_ref =  torch.from_numpy(np.load(f"ref/3_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy"))
    C_ref, (LH, HL, HH) = pywt.dwt2(C_ref, 'bior1.3')
    C_ref = torch.from_numpy(C_ref) 
else:
    theta_0 = prior.sample()
    C_ref = simulator(list(chr_cen.values()), resolution, sigma_spot, noisy)
    C_ref, (LH, HL, HH) = pywt.dwt2(C_ref, 'bior1.3')
    C_ref = torch.from_numpy(C_ref)





theta = torch.zeros(nb_train, dim)
C = torch.zeros(nb_train,C_ref.size(0)*C_ref.size(1) )
for k in range(nb_train):
    theta[k] = prior.sample()
    C_tmp = simulator(theta[k], resolution, sigma_spot, noisy)
    C_tmp, (LH, HL, HH) = pywt.dwt2(C_tmp, 'bior1.3')
    C_tmp = torch.from_numpy(C_tmp)
    C[k] = C_tmp.reshape((1,C_tmp.size(0)*C_tmp.size(1)))

inference_method = NPE
density_estimator = "nsf"

inference = inference_method(prior, density_estimator=density_estimator)
posterior_estimator = inference.append_simulations(theta, C).train(
        training_batch_size=100
    )
if inference_method==NPE:
    posterior = DirectPosterior(prior=prior, posterior_estimator=posterior_estimator)

if inference_method==NLE:
    posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
                                    mcmc_parameters={"num_chains": 20,
                                    "thin": 5})
    
posterior.set_default_x(C_ref.reshape(1,C_ref.size(0)*C_ref.size(1)))

samples = posterior.sample((nb_posterior_samples,))


fig, axes = plt.subplots(2,2, figsize=(12, 10))

for k, chr in enumerate (list(chr_seq.keys())):

    sns.kdeplot(ax = axes[k//2, k%2], data=samples[:, k])
    
    axes[k//2, k%2].set_xlabel(r"$\theta_{centro}$")
    axes[k//2, k%2].axvline(x=chr_cen[chr], linestyle='--', color="goldenrod", label=rf"$\theta_0$")
    
    axes[k//2, k%2].set_title(rf"$\theta_{{centro}}$ for chr ${k+1}$")
    sec_x = axes[k//2, k%2].secondary_xaxis(location='bottom') #draw the separation between chr    
    sec_x.set_xticks([1, chr_seq[chr]], labels=[])
    sec_x.tick_params('x', length=10, width=1.5)
    extraticks = [1, chr_seq[chr]]
    
    axes[k//2, k%2].set_xticks(list(axes[k//2, k%2].get_xticks()) + extraticks)
    axes[k//2, k%2].set_xlim(1, chr_seq[chr])

plt.suptitle(rf"$p(\theta_{{centro}}|C_{{ref}})$"+f"\n {origin} data - noisy : {noisy}"+fr" - $\sigma^2$ {sigma_spot} - res {resolution}" + f"\n {inference_method.__name__} - {density_estimator} - wavelets")
plt.show()
               

    
     
    