import numpy as np

from itertools import combinations

import matplotlib.pyplot as plt
import pickle
from simulator import *






chr_seq = {"chr01": 230209, "chr02": 813179, "chr03": 316619}
chr_cen = {'chr01': 151584, 'chr02': 238325, 'chr03': 114499}

resolution = 32000
nb_bead_chr, start_bead_chr, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)
C_ref =  np.load(f"ref/3_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")



# with open(f'ref/ref_{resolution}_norm_HiC_duan_intra_all.npy', 'rb') as f:
#             C_ref_all = np.load(f)
# start = 0
# nb_tot_bead_exp = nb_bead_chr["chr01"]+nb_bead_chr["chr02"]+nb_bead_chr["chr03"]
# C_ref = np.zeros((nb_tot_bead_exp, nb_tot_bead_exp))

# for (chr_row, chr_col) in combinations(chr_seq_exp.keys(),r=2):
        
#         start_row = start_bead_chr[chr_row]
#         end_row = start_row + nb_bead_chr[chr_row]
#         start_col = start_bead_chr[chr_col]
#         end_col = start_col + nb_bead_chr[chr_col]
#         C_ref[start_row: end_row,start_col: end_col] = C_ref_all[start_row: end_row,start_col: end_col]
if 0:
    C_ref =  np.load(f"ref/3_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")
    #plot_C_genome(np.log(C_ref+1), resolution, 2, 100, chr_cen)
    plot_C_genome(C_ref, resolution, 2, 100, chr_cen)

    C_simu = np.zeros((nb_tot_bead,nb_tot_bead))

    for (chr_row, chr_col) in combinations(chr_seq.keys(),r=2):
        n_row = chr_seq[chr_row]//resolution
        n_col = chr_seq[chr_col]//resolution

        c_i_simu = chr_cen[chr_row]//resolution
        c_j_simu = chr_cen[chr_col]//resolution
        sig_2_simu = 1
        intensity_simu = 100
        
        
        
        C_simu[start_bead_chr[chr_row]:start_bead_chr[chr_row]+nb_bead_chr[chr_row]-1, start_bead_chr[chr_col]:start_bead_chr[chr_col]+nb_bead_chr[chr_col]-1] = simulator(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=1)
            
    plot_C_genome(C_simu, resolution, sig_2_simu, intensity_simu, chr_cen)
    print(0.5*(Correlation_inter_upper_average(C_simu, C_ref, resolution, Pearson_correlation_row)+Correlation_inter_upper_average(C_simu, C_ref, resolution, Pearson_correlation_col)))
    print(Correlation_inter_upper_average(C_simu, C_ref, resolution, Pearson_correlation_vector))

#simulation theta/param
if 0:
    p_corr_row_col = []
    p_corr_vector = []
    theta = []
    param = []
    for k in range(1000):
        print(k)
        ############# simulate theta, sig_2, intensity ##########
        centro = {}
        sig_2 = random.uniform(0.1, 10)
        
        intensity = random.choice(range(1, 1001))
        for chr in chr_seq.keys():
            c = pdist.Uniform(low=1, high=chr_seq[chr]-1).sample()
            centro[chr]=int(c.detach().item())

        theta.append(centro)
        param.append((sig_2, intensity))
        
        #########################################################
        C_simu = np.zeros((nb_tot_bead,nb_tot_bead))

        for (chr_row, chr_col) in combinations(chr_seq.keys(),r=2):
            n_row = chr_seq[chr_row]//resolution
            n_col = chr_seq[chr_col]//resolution
            c_i_simu = centro[chr_row]//resolution
            c_j_simu = centro[chr_col]//resolution
            sig_2_simu, intensity_simu = param[k]
            
            
            C_simu[start_bead_chr[chr_row]:start_bead_chr[chr_row]+nb_bead_chr[chr_row]-1, start_bead_chr[chr_col]:start_bead_chr[chr_col]+nb_bead_chr[chr_col]-1] = simulator(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=1)
        # plot_C_genome(C_ref, resolution, sig_2_simu, intensity_simu, chr_cen)
        # plot_C_genome(C_simu, resolution, sig_2_simu, intensity_simu, centro)
        
        p_corr_row_col.append(0.5*(Correlation_inter_upper_average(C_simu, C_ref, resolution, Pearson_correlation_row)+Correlation_inter_upper_average(C_simu, C_ref, resolution, Pearson_correlation_col)))
        p_corr_vector.append(Correlation_inter_upper_average(C_simu, C_ref, resolution, Pearson_correlation_vector))

    with open('simulation_little_genome/true/theta', 'wb') as f:
            pickle.dump(theta, f)
    with open('simulation_little_genome/true/param', 'wb') as f:
            pickle.dump(param, f)
    with open('simulation_little_genome/true/P_corr_inter_vector', 'wb') as f:
            pickle.dump(p_corr_vector, f)
    with open('simulation_little_genome/true/P_corr_inter_row_col', 'wb') as f:
            pickle.dump(p_corr_row_col, f)


#simulation reprend theta param existants
with open('simulation_little_genome/true/res_32000/clear/theta', 'rb') as f:   
      theta=pickle.load(f)
with open('simulation_little_genome/true/res_32000/clear/param', 'rb') as f:   
      param=pickle.load(f)

s_corr_row = []
s_corr_col = []
s_corr_row_col = []
s_corr_vector = []

for k in range(1000):
    print(k)
    C_simu = np.zeros((nb_tot_bead,nb_tot_bead))
    for (chr_row, chr_col) in combinations(chr_seq.keys(),r=2):
        n_row = chr_seq[chr_row]//resolution
        n_col = chr_seq[chr_col]//resolution
        
        c_i_simu = theta[k][chr_row]//resolution
        c_j_simu = theta[k][chr_col]//resolution
        sig_2_simu, intensity_simu = param[k]
        
        
        C_simu[start_bead_chr[chr_row]:start_bead_chr[chr_row]+nb_bead_chr[chr_row]-1, start_bead_chr[chr_col]:start_bead_chr[chr_col]+nb_bead_chr[chr_col]-1] = simulator(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=0)
    # plot_C_genome(C_ref, resolution, sig_2_simu, intensity_simu, chr_cen)
    # plot_C_genome(C_simu, resolution, sig_2_simu, intensity_simu, centro)

    corr_row = Correlation_inter_upper_average(C_simu, C_ref, resolution, Spearman_correlation_row)
    corr_col = Correlation_inter_upper_average(C_simu, C_ref, resolution, Spearman_correlation_col)
    s_corr_row.append(corr_row)
    s_corr_col.append(corr_col)
    s_corr_row_col.append(0.5*(corr_row+corr_col))
    s_corr_vector.append(Correlation_inter_upper_average(C_simu, C_ref, resolution, Spearman_correlation_vector))
    
   
    with open('simulation_little_genome/true/res_32000/clear/S_corr_inter_vector', 'wb') as f:
            pickle.dump(s_corr_vector, f)
    with open('simulation_little_genome/true/res_32000/clear/S_corr_inter_row_col', 'wb') as f:
            pickle.dump(s_corr_row_col, f)
    with open('simulation_little_genome/true/res_32000/clear/S_corr_inter_row', 'wb') as f:
            pickle.dump(s_corr_row, f)
    with open('simulation_little_genome/true/res_32000/clear/S_corr_inter_col', 'wb') as f:
            pickle.dump(s_corr_col, f)
      

