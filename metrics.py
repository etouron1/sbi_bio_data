from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from contact_matrix import read_contact_matrix_all_chr_from_simu, read_contact_matrix_all_chr_from_file,read_contact_matrix_all_chr_from_pickle
from simulators import chr_seq_yeast, chr_cen_yeast, sep_yeast, get_num_beads_and_start
# from normalization import downsample_matrix
from sbi.utils.metrics import c2st, unbiased_mmd_squared
from normalize_expected import *
from normalization import *

# def pearson_corr(contact_matrix_ref, contact_matrix_simu):
#     '''Compute the Pearson correlation'''
    
#     pear_corr = 0
    
#     for i in range(len(contact_matrix_ref)):
       
        
#         row_ref = np.delete(contact_matrix_ref[i], i)
#         non_nan_idx = np.invert(np.isnan(row_ref))
#         row_ref = row_ref[non_nan_idx]

#         row_simu = np.delete(contact_matrix_simu[i], i)
#         row_simu = row_simu[non_nan_idx]

#         corr= pearsonr(row_ref, row_simu).statistic
#         if np.isnan(corr):
#             pass
#         else:
#             pear_corr += corr
#     return pear_corr/len(contact_matrix_ref)

def pearson_corr_manuel(contact_matrix_ref, contact_matrix_simu):
    """
    Compute the Average row-based Pearson correlation between 2 contact matrices
    """

    n = len(contact_matrix_ref[0])
    corr = 0
    for i in range(len(contact_matrix_ref)):
        sum_exp_ref = 0
        sum_exp_simu = 0
        sum_exp_exp = 0
        sum_exp_carre_simu = 0
        sum_exp_carre_ref = 0
        for j in range(len(contact_matrix_ref[0])):
            if j!=i:
                sum_exp_exp += contact_matrix_ref[i,j]*contact_matrix_simu[i,j]
                sum_exp_ref += contact_matrix_ref[i,j]
                sum_exp_simu += contact_matrix_simu[i,j]
                sum_exp_carre_ref += contact_matrix_ref[i,j]**2
                sum_exp_carre_simu += contact_matrix_simu[i,j]**2
        num = n*sum_exp_exp - sum_exp_ref*sum_exp_simu
        denom = np.sqrt(n*sum_exp_carre_ref - sum_exp_ref**2) * np.sqrt(n*sum_exp_carre_simu - sum_exp_simu**2)
        
        if denom == 0 or np.isnan(denom):
            corr_tmp = 0
        else:
            corr_tmp = num/denom

        corr += corr_tmp
            
    return corr/len(contact_matrix_ref)

def pearson_correlation_vector(C_ref, C_simu):
    """
    Compute the upper matrix-based Pearson correlation between 2 contact matrices
    -> the upper part of the symmetric matrice is vectorized
    """
    C_ref_upper_vector = np.array((C_ref[np.triu_indices(len(C_ref))]))
    C_simu_upper_vector = np.array(C_simu[np.triu_indices(len(C_simu))])
    return pearsonr(C_ref_upper_vector[np.invert(np.isnan(C_ref_upper_vector))], C_simu_upper_vector[np.invert(np.isnan(C_ref_upper_vector))]).statistic

# from scipy import stats
# def compute_row_correlations(sim_counts, real_counts):
#     correlations = np.zeros(sim_counts.shape[0]) 
#     for i in range(real_counts.shape[0]):
#         row_1, row_2 = sim_counts[i], real_counts[i]

#         row_1 = np.delete(row_1, i)
#         row_2 = np.delete(row_2, i)

#         to_rm = np.invert((np.isnan(row_1) | np.isinf(row_1) | np.isnan(row_2) | np.isinf(row_2)))
        
#         if not np.any(to_rm):
            
#             continue
#         row_1_real = row_1[to_rm]
#         row_2_real = row_2[to_rm]

#         correlations[i] = stats.pearsonr(row_1[to_rm], row_2[to_rm])[0]
#         if np.isnan(correlations[i]) :
#             correlations[i] = 0
    
#     return np.mean(correlations)

def corr_intra(type, simu, ref, resolution):
    """
    Compute the Average row-based Pearson correlation for each intra-chr contact matrix
    Return : the average of all intra-chr correlations
    """
    if type=='y':
        num_chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq_yeast, resolution)

        corr_intra = 0
        for chr in chr_seq_yeast.keys():
            start = chr_bead_start[chr]
            end = chr_bead_start[chr] + num_chr_num_bead[chr]
            intra_ref = ref[start:end, start:end]
            intra_simu = simu[start:end, start:end]
            corr_intra += pearson_corr_manuel(intra_ref, intra_simu)
        return corr_intra/len(chr_seq_yeast.keys())
    
from itertools import combinations
from  scipy.special import binom
def corr_inter(type, simu, ref, resolution):
    """ 
    Compute the Average row-based Pearson correlation for each inter-chr contact matrix
    Return : the average of all inter-chr correlations
    """
    if type=='y':
        num_chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq_yeast, resolution)

        corr_inter = 0
        
        for (chr_1, chr_2) in combinations(chr_seq_yeast.keys(), r=2):
            
            start_1, end_1 = chr_bead_start[chr_1], chr_bead_start[chr_1] + num_chr_num_bead[chr_1] #get the start bead id and the end bead id for the chr
            start_2, end_2 = chr_bead_start[chr_2], chr_bead_start[chr_2] + num_chr_num_bead[chr_2] #get the start bead id and the end bead id for the chr
            inter_simu = simu[start_1:end_1, start_2:end_2] 
            inter_ref = ref[start_1:end_1, start_2:end_2] 
            corr_inter += pearson_corr_manuel(inter_ref, inter_simu)
        return corr_inter/binom(len(chr_seq_yeast), 2)
    
def mean_contact_per_genomic_distance(type, res, ref, simu):
    if type=='y':
        chr_seq = chr_seq_yeast
    else:
        print("unknow type")
        return
    num_bead_per_chr, bead_start_per_chr, nbead = get_num_beads_and_start(chr_seq_yeast, res)
    mapping_ref = get_mapping(ref, np.array(list(num_bead_per_chr.values())), verbose=True)
    x_ref = np.log(mapping_ref[0, 2:]*res)
    y_ref = np.log(mapping_ref[1, 2:])

    mapping_simu = get_mapping(simu, np.array(list(num_bead_per_chr.values())), verbose=True)
    x_simu = np.log(mapping_simu[0, 2:]*res)
    y_simu = np.log(mapping_simu[1, 2:])

    fig, ax = plt.subplots(figsize=(12, 12))

  
    a_ref, b_ref = np.polyfit(x_ref,y_ref, deg=1)
    print(a_ref, b_ref)
    a_simu, b_simu = np.polyfit(x_simu,y_simu, deg=1)
    print(a_simu, b_simu)
    plt.scatter(x_ref,y_ref, color='red', marker = '+', s=5)
    plt.scatter(x_simu,y_simu, color='blue', marker = '+', s=5)
    # plt.plot(x_ref, a_ref*x_ref+b_ref, linestyle='--',color = 'tomato', label=f'y_ref = {a_ref:.1f}x +{b_ref:.1f}')
    plt.plot(x_ref, a_ref*x_ref+b_ref, linestyle='--',color = 'tomato', label=rf'$y_{{ref}} \sim e ^{{{b_ref:.1f}}} x^{{{a_ref:.1f}}}$')
    plt.plot(x_simu, a_simu*x_simu+b_simu, linestyle='--',color = 'lightskyblue', label=rf'$y_{{simu}} \sim e^{{{b_simu:.1f}}} x^{{{a_simu:.1f}}}$')
    plt.xlabel(r"Genomic distance ($log_{10} bp$)")
    plt.ylabel(r"Mean contact frequency ($log_{10} bp$)")
    plt.title(f"Power-law fits to {res//1000} kb aggregated contact frequencies")
    plt.legend()
    plt.show()
    
# contact_matrix_ref = np.load("yeast_Cerevisiae_info/ref_10000_norm_HiC_nelle.npy") #1216, 1216
# #downsampled_matrix_ref = downsample_matrix('y', 10000, 32000, contact_matrix_ref)

# nb_simu=75
# contact_matrix_simu = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu, 3200, 'n', 'tjong')
# downsampled_matrix_simu = downsample_matrix('y', 3200, 10000, contact_matrix_simu)
# #downsampled_matrix_simu = downsample_matrix('y', 10000, 32000, downsampled_matrix_simu)

# mean_contact_per_genomic_distance('y', 10000, contact_matrix_ref, downsampled_matrix_simu)

#@jit(nopython=True)
from scipy.special import kl_div
def mean_kl(matrix_ref, matrix_simu):
    kl = 0
    for i in range(len(matrix_ref)):
        # kl_i = sum_j ref[i,j]*log(ref[i,j]/simu[i,j]) - ref[i,j] + simu[i,j]
        kl+= np.sum(kl_div(matrix_ref[i]/np.sum(matrix_ref[i]), matrix_simu[i]/np.sum(matrix_simu[i])))
    #mean_i kl_i
    return 1.0/len(matrix_ref)*kl
    
    
# if __name__ == "__main__":
    

#     contact_matrix_ref = np.load("yeast_Cerevisiae_info/duan.SC.10000.normed.npy")
#     nb_simu=200
#     contact_matrix = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu)
#     downsampled_matrix = downsample_matrix('y', 3200, 10000, contact_matrix )
#     print(pearson_correlation_vector(contact_matrix_ref, downsampled_matrix))

    # x = torch.ones((1000,1000))
    # y = 2*torch.ones((1000,1000))
    # print(unbiased_mmd_squared(x,y,0.01))

    # x = torch.tensor(np.ones((10,10)))
    # y = torch.tensor(np.zeros((10,10)))
    # print(c2st(x,y))

    # x = torch.ones((10,10))
    # y = torch.zeros((10,10))
    # print(c2st(x,y))

    # fig, ax = plt.subplots()
    # for nb_simu in range(1,200,10):
    #     print(nb_simu)
    #     contact_matrix_simu = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu)
    #     downsampled_matrix = downsample_matrix('y', 3200, 10000, contact_matrix_simu)

    #     norm_matrix_simu = np.log(downsampled_matrix+1)
    #     min_mat, max_mat = norm_matrix_simu.min(), norm_matrix_simu.max()
    #     norm_matrix_simu = (norm_matrix_simu - min_mat)/(max_mat-min_mat)
        

    #     #print(contact_matrix_simu)
    #     #contact_matrix_ref = read_contact_matrix_all_chr_from_file(chr_seq_yeast, sep_yeast, "yeast_Cerevisiae_info/contact_matrix.matrix")

    #     contact_matrix_ref = np.load("yeast_Cerevisiae_info/duan.SC.10000.normed.npy")
        
    #     norm_matrix_ref = np.log(contact_matrix_ref+1)
    #     min_mat, max_mat = norm_matrix_ref.min(), norm_matrix_ref.max()
    #     norm_matrix_ref = (norm_matrix_ref - min_mat)/(max_mat-min_mat)
      

    #     #pearson_correlation = pearson_corr(contact_matrix_ref, downsampled_matrix)
    #     #print("auto", pearson_correlation)
    #     #pearson_correlation_manuel = pearson_corr_manuel(contact_matrix_ref, contact_matrix_simu)
    #     #print("manuel", pearson_correlation_manuel)
    #     #plt.scatter(nb_simu*100, pearson_correlation, marker='+', color = "black")
    #     #plt.scatter(nb_simu*1000, pearson_correlation_manuel, marker='+', color = "red")
    #     plt.scatter(nb_simu*100,c2st(torch.tensor(norm_matrix_ref), torch.tensor(norm_matrix_simu)), marker='+', color = "red")
        
    #     #plt.scatter(nb_simu*100,mean_kl(contact_matrix_ref, downsampled_matrix), marker='+', color = "red")
    #     #plt.scatter(nb_simu*100,unbiased_mmd_squared(torch.tensor(contact_matrix_ref), torch.tensor(downsampled_matrix), scale=0.1), marker='+', color = "red")
    # plt.xlabel("nb of simulations")
    # #plt.ylabel("Pearson correlation")
    # plt.ylabel("C2ST")
    # #plt.title("Pearson correlation between normalized ref matrix and raw simu matrix")
    # plt.title("C2ST between 0-1 normed HiC normalized ref matrix and 0-1 normed simu matrix")
    # #ax.set_ylim(0,1)
    # plt.show()

    # contact_matrix_ref = np.load("yeast_Cerevisiae_info/duan.SC.10000.normed.npy")
    # contact_matrix_simu, nb_simu = read_contact_matrix_all_chr_from_pickle()
    # pearson_correlation = pearson_corr(contact_matrix_ref, contact_matrix_simu)
    # print("auto", pearson_correlation)

    # contact_matrix_ref = np.load("yeast_Cerevisiae_info/duan.SC.10000.normed.npy")
    # norm_matrix_ref = np.log(contact_matrix_ref+1)
    # # min_mat, max_mat = norm_matrix.min(), norm_matrix.max()
    # # norm_matrix_ref = (norm_matrix - min_mat)/(max_mat-min_mat)
    
    # sep_yeast=10000
    # nb_simu=150
    # num_bead_per_chr, bead_start_per_chr, nbead = get_num_beads_and_start(chr_seq_yeast, sep_yeast)
    # contact_matrix = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu)
    # downsampled_matrix = downsample_matrix('y', 3200, 10000, contact_matrix )

    # norm_matrix_simu = np.log(downsampled_matrix+1)
    # # min_mat, max_mat = norm_matrix.min(), norm_matrix.max()
    # # norm_matrix_simu = (norm_matrix - min_mat)/(max_mat-min_mat)


    # pearson_correlation = pearson_corr(contact_matrix_ref, downsampled_matrix)
    # print("auto", pearson_correlation)
    # pearson_correlation = pearson_corr(norm_matrix_ref, norm_matrix_simu)
    # print("auto", pearson_correlation)
    # # pearson_correlation_manuel = pearson_corr_manuel(contact_matrix_ref, downsampled_matrix)
    # # print("manuel", pearson_correlation_manuel)
# import pyro.distributions as pdist
# X = pdist.Normal(loc=0, scale=1).sample((100,100))
# print(X)
# Y = pdist.Normal(loc=0, scale=0.1).sample((100,100))
# print(Y)
# print(pearson_corr(X,-X))