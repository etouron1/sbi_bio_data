import numpy as np
from iced import normalization
from simulators import get_num_beads_and_start, chr_seq_yeast_mean, chr_seq_yeast_duan, sep_yeast
from contact_matrix import read_contact_matrix_all_chr_from_simu
from itertools import combinations


def normalization_contact_matrix(C):
    """
    Normalization of the contact matrix by Tjong et al. (supp mat)
    """
    C_norm = np.zeros_like(C)
    
    sum_j_C_i_j = np.sum(C, axis=1)
    sum_i_C_i_j = np.sum(C, axis=0)
    sum_C_upper_strict = np.sum(np.triu(C, k=1))

    for i in range (len(C)):
        for j in range(len(C[0])):

            if sum_i_C_i_j[j]!=0 and sum_j_C_i_j[i]!=0:
                
                C_norm[i,j] = C[i,j] * sum_C_upper_strict/(sum_i_C_i_j[j]*sum_j_C_i_j[i])
            else:
                C_norm[i,j] = C[i,j]

    return C_norm

def HiC_iterative_normalization_contact_matrix(C):
    return normalization.ICE_normalization(C)

# import matplotlib.pyplot as plt

# nb_simu = 100
# contact_matrix = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu)
# print(contact_matrix)
# hic_norm = normalization.ICE_normalization(contact_matrix)
# print("norm")
# fig, ax = plt.subplots()
# m = ax.matshow(np.log(hic_norm+1), cmap="RdBu_r", vmin=-3, vmax=3)
# ax.set_title(f"Simu Observed / Expected\n resolution {10000}")
# fig.colorbar(m)
# plt.show()

# def downsample_contact_matrix(counts, lengths, factor=2, normalize=False):

#     """
#     Downsamples the resolution of a matrix

#     Parameters
#     ----------
#     counts : ndarray (N, N)
#         contact counts matrix to downsample

#     lengths : ndarray (L, )
#         chromosomes lengths

#     coef : int, optionnal, default: 2
#         downsample resolution of the counts matrix by `coef`

#     Returns
#     -------
#     target_counts, target_lengths : ndarray
#     """

#     if factor == 1:
#         return counts, lengths
#     # FIXME there is probably a better way to do this
#     target_lengths = np.ceil(lengths.astype(float) / factor).astype(int)
    
#     print(lengths)
#     print(target_lengths)
#     target_counts = np.zeros((target_lengths.sum(),
#                               target_lengths.sum()))
#     begin_i, end_i = 0, 0
#     target_begin_i, target_end_i = 0, 0
#     for i, length_i in enumerate(lengths):
#         end_i += length_i
#         target_end_i += target_lengths[i]
#         begin_j, end_j = 0, 0
#         target_begin_j, target_end_j = 0, 0
#         for j, length_j in enumerate(lengths):
#             end_j += length_j
#             target_end_j += target_lengths[j]

#             sub_counts = counts[begin_i:end_i, begin_j:end_j]
#             sub_target_counts = target_counts[target_begin_i:target_end_i,
#                                               target_begin_j:target_end_j]
#             d = np.zeros(sub_target_counts.shape)
#             for i_start in range(factor):
#                 for j_start in range(factor):
#                     s = sub_counts[i_start::factor, j_start::factor]
#                     d[:s.shape[0], :s.shape[1]] += np.invert(np.isnan(s))
#                     s[np.isnan(s)] = 0
#                     sub_target_counts[:s.shape[0], :s.shape[1]] += s
#             if normalize:
#                 sub_target_counts /= d
        
#             begin_j = end_j
#             target_begin_j = target_end_j
#         begin_i = end_i
#         target_begin_i = target_end_i
#     return target_counts, target_lengths
            
# num_bead_per_chr, bead_start_per_chr, nbead = get_num_beads_and_start(chr_seq_yeast, 10000)
# print(num_bead_per_chr.values())
# num_bead_per_chr, bead_start_per_chr, nbead = get_num_beads_and_start(chr_seq_yeast, sep_yeast)
# nb_simu = 1
# contact_matrix = read_contact_matrix_all_chr_from_simu('y', nb_simu)

#downsample, new_length = downsample_contact_matrix(contact_matrix, np.array(list(num_bead_per_chr.values())), 3)
# print(np.shape(contact_matrix))
# print(np.shape(downsample))
# print(contact_matrix)
# print(downsample)



def downsample_matrix(type, resolution_in, resolution_out, contact_matrix, mode):
    """
    Downsample a contact matrix to a wish resolution
    The mean kernel used is of size E(resolution_out/resolution_in)
    The matrice is reshaped such that each block has the good size
    Each pixel of the output matrix is the average of the pixels in the kernel neighborhood of the input matrix
    """
    if type=='y_nelle':
        chr_seq = chr_seq_yeast_mean
    elif type=='y_duan':
        chr_seq = chr_seq_yeast_duan
    else:
        print("type unknown")
        return
    
    ratio = resolution_out//resolution_in
    num_bead_per_chr_in, bead_start_per_chr_in, nbead_in = get_num_beads_and_start(chr_seq, resolution_in)
    num_bead_per_chr_out, bead_start_per_chr_out, nbead_out = get_num_beads_and_start(chr_seq, resolution_out)
    downsampled_matrix = np.zeros((nbead_out, nbead_out))
    for chr in chr_seq.keys():
        start = bead_start_per_chr_in[chr]
        end = bead_start_per_chr_in[chr] + num_bead_per_chr_in[chr]
        
        output_matrix = np.zeros((num_bead_per_chr_out[chr], num_bead_per_chr_out[chr]))
        N = num_bead_per_chr_in[chr]
        n = num_bead_per_chr_out[chr]
        q = N-ratio*n

        if q%2==0:
            i_start = q//2
            i_end = q//2
        else:
            i_start = q//2
            i_end = q//2+1
        input_matrix = contact_matrix[start+i_start:end-i_end, start+i_start:end-i_end]
        
        for i in range(len(output_matrix)):
            for j in range(len(output_matrix[0])):
                i_input = ratio*i+1
                j_input = ratio*j+1
                if mode =="mean":
                    output_matrix[i,j] = 1.0/ratio**2*np.sum(input_matrix[i_input-1:i_input+2,j_input-1:j_input+2])
                else:
                    output_matrix[i,j] = np.sum(input_matrix[i_input-1:i_input+2,j_input-1:j_input+2])
        start = bead_start_per_chr_out[chr]
        end = bead_start_per_chr_out[chr] + num_bead_per_chr_out[chr]
        downsampled_matrix[start:end, start:end]=output_matrix

    for (chr_1,chr_2) in combinations(chr_seq.keys(), r=2):
        start_1,start_2 = bead_start_per_chr_in[chr_1],bead_start_per_chr_in[chr_2]
        end_1, end_2 = bead_start_per_chr_in[chr_1] + num_bead_per_chr_in[chr_1], bead_start_per_chr_in[chr_2] + num_bead_per_chr_in[chr_2]
        
        output_matrix = np.zeros((num_bead_per_chr_out[chr_1], num_bead_per_chr_out[chr_2]))
        N = num_bead_per_chr_in[chr_1]
        M = num_bead_per_chr_in[chr_2]
        n = num_bead_per_chr_out[chr_1]
        m = num_bead_per_chr_out[chr_2]
        q = N-ratio*n
        r = M-ratio*m
        if q%2==0:
            i_start = q//2
            i_end = q//2
        else:
            i_start = q//2
            i_end = q//2+1
        if r%2==0:
            j_start = r//2
            j_end = r//2
        else:
            j_start = r//2
            j_end = r//2+1
        input_matrix = contact_matrix[start_1+i_start:end_1-i_end, start_2+j_start:end_2-j_end]
        
        for i in range(len(output_matrix)):
            for j in range(len(output_matrix[0])):
                i_input = ratio*i+1
                j_input = ratio*j+1
                if mode=="mean":
                    output_matrix[i,j] = 1.0/ratio**2*np.sum(input_matrix[i_input-1:i_input+2,j_input-1:j_input+2])
                else:
                    output_matrix[i,j] = np.sum(input_matrix[i_input-1:i_input+2,j_input-1:j_input+2])
        start_1,start_2 = bead_start_per_chr_out[chr_1],bead_start_per_chr_out[chr_2]
        end_1,end_2 = bead_start_per_chr_out[chr_1] + num_bead_per_chr_out[chr_1],bead_start_per_chr_out[chr_2] + num_bead_per_chr_out[chr_2]
        downsampled_matrix[start_1:end_1, start_2:end_2]=output_matrix
        downsampled_matrix[start_2:end_2, start_1:end_1]=np.transpose(output_matrix)

    return downsampled_matrix
    


if __name__=="__main__":

    from normalize_expected import *
    from metrics import *
    from contact_matrix import read_contact_matrix_all_chr_from_file, construct_ref_matrix_from_file
    import matplotlib.pyplot as plt

    

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    #contact_matrix_ref = np.load("yeast_Cerevisiae_info/ref_32000_norm_HiC_duan_intra_20_kb.npy") #1216, 1216
    #contact_matrix_ref = np.load("yeast_Cerevisiae_info/ref_3200_norm_HiC_duan_intra_all.npy") #1216, 1216

    #contact_matrix_ref_nelle = np.load("yeast_Cerevisiae_info/ref_10000_norm_HiC_nelle.npy") #1216, 1216 
    #contact_matrix_ref_nelle = downsample_matrix('y_nelle', 10000, 32000, contact_matrix_ref_nelle, mode="mean")

    contact_matrix_ref_duan = np.load("yeast_Cerevisiae_info/ref_32000_norm_HiC_duan_intra_all.npy") #1216, 1216
    
    
    

    #np.fill_diagonal(contact_matrix_ref, contact_matrix_ref.max())
    #contact_matrix_ref = np.load("yeast_Cerevisiae_info/ref_32000_norm_HiC_duan_intra_20_kb.npy") #1216, 1216
    #contact_matrix_ref = np.load("yeast_Cerevisiae_info/ref_10000_raw_nelle.npy") #1216, 1216


    resolution_corr =32000
    # downsampled_matrix_ref = downsample_matrix('y', 3200, 10000, contact_matrix_ref)
    # downsampled_matrix_ref = downsample_matrix('y', 10000, resolution_corr, downsampled_matrix_ref)
    #contact_matrix_ref = normalization_contact_matrix(contact_matrix_ref)


    # num_bead_per_chr, bead_start_per_chr, nbead = get_num_beads_and_start(chr_seq_yeast, resolution_corr)

    # mapping_ref = get_mapping(contact_matrix_ref, np.array(list(num_bead_per_chr.values())), verbose=True)
    # c_expected_ref = get_expected(contact_matrix_ref, np.array(list(num_bead_per_chr.values())), mapping=mapping_ref)
    # observed_over_expected_ref = contact_matrix_ref / c_expected_ref

    #contact_matrix_ref = read_contact_matrix_all_chr_from_file(chr_seq_yeast, 10000, "yeast_Cerevisiae_info/contact_matrix.matrix")
    #contact_matrix_ref = normalization_contact_matrix(contact_matrix_ref)
    #downsampled_matrix_ref = contact_matrix_ref

    # correlation_intra_nelle_nelle = []
    # correlation_total_nelle_nelle = []
    # correlation_inter_nelle_nelle = []

    # correlation_intra_tjong_nelle = []
    # correlation_total_tjong_nelle = []
    # correlation_inter_tjong_nelle = []

    correlation_intra_tjong_duan = []
    correlation_total_tjong_duan = []
    correlation_inter_tjong_duan = []

    # correlation_intra_tjong_duan_faux = []
    # correlation_total_tjong_duan_faux = []
    # correlation_inter_tjong_duan_faux = []

    for i in range(1,101):
        print(i)
        if i==1:
            # contact_matrix_simu_nelle_nelle = read_contact_matrix_all_chr_from_simu('y', 1, i, 3200, 'n', "nelle")
            # contact_matrix_simu_tjong_nelle = read_contact_matrix_all_chr_from_simu('y', 1, i, 3200, 'n', "tjong")
            contact_matrix_simu_tjong_duan = read_contact_matrix_all_chr_from_simu('y', 1, i, 3200, 'n', "duan", plot=0)
            #contact_matrix_simu_tjong_duan_faux = read_contact_matrix_all_chr_from_simu('y', 1, i, 3200, 'n', "tjong_faux_centro", plot=0)
        else:
            # contact_matrix_simu_nelle_nelle += read_contact_matrix_all_chr_from_simu('y', 1, i, 3200, 'n', "nelle")
            # contact_matrix_simu_tjong_nelle += read_contact_matrix_all_chr_from_simu('y', 1, i, 3200, 'n', "tjong")
            contact_matrix_simu_tjong_duan += read_contact_matrix_all_chr_from_simu('y', 1, i, 3200, 'n', "duan", plot=0)
            #contact_matrix_simu_tjong_duan_faux += read_contact_matrix_all_chr_from_simu('y', 1, i, 3200, 'n', "tjong_faux_centro", plot=0)
        #contact_matrix_simu = normalization_contact_matrix(contact_matrix_simu)
       
      
        # downsampled_matrix_simu_nelle_nelle = downsample_matrix('y_nelle', 3200, 10000, contact_matrix_simu_nelle_nelle )
        # downsampled_matrix_simu_nelle_nelle = downsample_matrix('y_nelle', 10000, resolution_corr, downsampled_matrix_simu_nelle_nelle)
        
        # downsampled_matrix_simu_tjong_nelle = downsample_matrix('y_nelle', 3200, 10000, contact_matrix_simu_tjong_nelle )
        # downsampled_matrix_simu_tjong_nelle = downsample_matrix('y_nelle', 10000, resolution_corr, downsampled_matrix_simu_tjong_nelle)

        downsampled_matrix_simu_tjong_duan = downsample_matrix('y_duan', 3200, 10000, contact_matrix_simu_tjong_duan, "mean" )
        downsampled_matrix_simu_tjong_duan = downsample_matrix('y_duan', 10000, resolution_corr, downsampled_matrix_simu_tjong_duan, "mean")

        #downsampled_matrix_simu_tjong_duan_faux = downsample_matrix('y_duan', 3200, 10000, contact_matrix_simu_tjong_duan_faux, "mean" )
        #downsampled_matrix_simu_tjong_duan_faux = downsample_matrix('y_duan', 10000, resolution_corr, downsampled_matrix_simu_tjong_duan_faux, "mean")

        
        #downsampled_matrix_simu_vrai = normalization_contact_matrix(downsampled_matrix_simu_vrai)


        # mapping_simu = get_mapping(downsampled_matrix_simu_vrai, np.array(list(num_bead_per_chr.values())), verbose=True)
        # c_expected_simu = get_expected(downsampled_matrix_simu_vrai, np.array(list(num_bead_per_chr.values())), mapping=mapping_simu)
        # observed_over_expected_simu = downsampled_matrix_simu_vrai / c_expected_simu

        #downsampled_matrix_simu_tjong = downsample_matrix('y', 3200, 10000, contact_matrix_simu_tjong)
        #downsampled_matrix_simu_faux = downsample_matrix('y', 3200, 10000, contact_matrix_simu_faux)
        # downsampled_matrix_simu_tjong = downsample_matrix('y', 10000, resolution_corr, downsampled_matrix_simu_tjong)
        # downsampled_matrix_simu_faux = downsample_matrix('y', 10000, resolution_corr, downsampled_matrix_simu_faux)


        # num_bead_per_chr, bead_start_per_chr, nbead = get_num_beads_and_start(chr_seq_yeast, sep_yeast)
        # mapping_simu = get_mapping(contact_matrix_simu, np.array(list(num_bead_per_chr.values())), verbose=True)
        # c_expected_simu = get_expected(contact_matrix_simu, np.array(list(num_bead_per_chr.values())), mapping=mapping_simu)
        # expected_downsampled = downsample_matrix('y', 3200, 32000, c_expected_simu )

        # correlation_intra_vrai.append(corr_intra('y', observed_over_expected_ref, observed_over_expected_simu, resolution_corr))
        # correlation_inter_vrai.append(corr_inter('y', observed_over_expected_ref, observed_over_expected_simu, resolution_corr))
        # correlation_total_vrai.append(pearson_corr_manuel(observed_over_expected_ref, observed_over_expected_simu))
        
        # correlation_intra_nelle_nelle.append(corr_intra('y', contact_matrix_ref_nelle, downsampled_matrix_simu_nelle_nelle, resolution_corr))
        # correlation_inter_nelle_nelle.append(corr_inter('y', contact_matrix_ref_nelle, downsampled_matrix_simu_nelle_nelle, resolution_corr))
        # correlation_total_nelle_nelle.append(pearson_corr_manuel(contact_matrix_ref_nelle, downsampled_matrix_simu_nelle_nelle))

        # correlation_intra_tjong_nelle.append(corr_intra('y', contact_matrix_ref_nelle, downsampled_matrix_simu_tjong_nelle, resolution_corr))
        # correlation_inter_tjong_nelle.append(corr_inter('y', contact_matrix_ref_nelle, downsampled_matrix_simu_tjong_nelle, resolution_corr))
        # correlation_total_tjong_nelle.append(pearson_corr_manuel(contact_matrix_ref_nelle, downsampled_matrix_simu_tjong_nelle))
        
        correlation_intra_tjong_duan.append(corr_intra('y', contact_matrix_ref_duan, downsampled_matrix_simu_tjong_duan, resolution_corr))
        correlation_inter_tjong_duan.append(corr_inter('y', contact_matrix_ref_duan, downsampled_matrix_simu_tjong_duan, resolution_corr))
        correlation_total_tjong_duan.append(pearson_corr_manuel(contact_matrix_ref_duan, downsampled_matrix_simu_tjong_duan))
        
        # correlation_intra_tjong_duan_faux.append(corr_intra('y', contact_matrix_ref_duan, downsampled_matrix_simu_tjong_duan_faux, resolution_corr))
        # correlation_inter_tjong_duan_faux.append(corr_inter('y', contact_matrix_ref_duan, downsampled_matrix_simu_tjong_duan_faux, resolution_corr))
        # correlation_total_tjong_duan_faux.append(pearson_corr_manuel(contact_matrix_ref_duan, downsampled_matrix_simu_tjong_duan_faux))

        # print(correlation_intra_tjong)
        # print(correlation_inter_tjong)
        
  
    ax4.set_axis_off()
    plt.xticks(range(0,10001, 100))
    #ax1.plot(range(0, 5001, 100)[1:], correlation_intra_tjong_duan_faux, marker="o", color="salmon", linestyle="--", label="tjong_faux_centro")
    # ax1.plot(range(0, 5001, 100)[1:], correlation_intra_nelle_nelle, marker="o", color="sandybrown", linestyle="--", label="simu nelle data nelle")
    # ax1.plot(range(0, 5001, 100)[1:], correlation_intra_tjong_nelle, marker="*", color="salmon", linestyle="--", label="simu tjong data nelle")
    ax1.plot(range(0, 10001, 100)[1:], correlation_intra_tjong_duan, marker="+", color="red", linestyle="--", label="simu tjong data duan")
    ax1.set_xlabel("nb simulations")
    ax1.set_ylabel("Av. Row-b. Pearson corr.")
    ax1.set_title("Intra_chr contacts")
    ax1.legend()
    #ax2.plot(range(0, 5001, 100)[1:], correlation_inter_tjong_duan_faux, marker="o", color="lightskyblue", linestyle="--", label="tjong_faux_centro")
    # ax2.plot(range(0, 5001, 100)[1:], correlation_inter_nelle_nelle, marker="o", color="lightskyblue", linestyle="--",label="simu nelle data nelle")
    # ax2.plot(range(0, 5001, 100)[1:], correlation_inter_tjong_nelle, marker="*", color="blue", linestyle="--",label="simu tjong data nelle")
    ax2.plot(range(0, 10001, 100)[1:], correlation_inter_tjong_duan, marker="+", color="cornflowerblue", linestyle="--",label="simu tjong data duan")
    ax2.set_xlabel("nb simulations")
    ax2.set_ylabel("Av. Row-b. Pearson corr.")
    ax2.set_title("Inter_chr contacts")
    ax2.legend()
    #ax3.plot(range(0, 5001, 100)[1:],correlation_total_tjong_duan_faux, marker="o", color="grey", linestyle="--", label="tjong_faux_centro" )
    # ax3.plot(range(0, 5001, 100)[1:],correlation_total_nelle_nelle, marker="o", color="silver", linestyle="--", label="simu nelle data nelle")
    # ax3.plot(range(0, 5001, 100)[1:],correlation_total_tjong_nelle, marker="*", color="slategrey", linestyle="--", label="simu tjong data nelle")
    ax3.plot(range(0, 10001, 100)[1:],correlation_total_tjong_duan, marker="+", color="black", linestyle="--", label="simu tjong data duan")
    ax3.set_xlabel("nb simulations")
    ax3.set_title("All contacts")
    ax3.set_ylabel("Av. Row-b. Pearson corr.")
    ax3.legend()

        #print(pearson_corr_manuel(contact_matrix_ref, downsampled_matrix))
        # print(pearson_corr(contact_matrix_ref, downsampled_matrix))
        # print(compute_row_correlations(downsampled_matrix, contact_matrix_ref))
        # print(corr_intra('y', contact_matrix_ref, downsampled_matrix, 10000))
        # print(corr_inter('y', contact_matrix_ref, downsampled_matrix, 10000))
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    plt.show()

    # sep_yeast=10000
    # num_bead_per_chr, bead_start_per_chr, nbead = get_num_beads_and_start(chr_seq_yeast, sep_yeast)

    # contact_matrix_ref = np.load("yeast_Cerevisiae_info/duan.SC.10000.normed.npy")
    # mapping_ref = get_mapping(contact_matrix_ref, np.array(list(num_bead_per_chr.values())), verbose=True)
    # c_expected_ref = get_expected(contact_matrix_ref, np.array(list(num_bead_per_chr.values())), mapping=mapping_ref)
    # observed_over_expected_ref = contact_matrix_ref / c_expected_ref

    # nb_simu=50
    # contact_matrix = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu, 10000, 'n')
    # #downsampled_matrix = downsample_matrix('y', 3200, 10000, contact_matrix )
    # mapping_simu = get_mapping(contact_matrix, np.array(list(num_bead_per_chr.values())), verbose=True)
    # c_expected_simu = get_expected(contact_matrix, np.array(list(num_bead_per_chr.values())), mapping=mapping_simu)
    # observed_over_expected_simu = contact_matrix / c_expected_simu

    # diff = observed_over_expected_ref-observed_over_expected_simu
    # fig, ax = plt.subplots() 
    # draw = ax.matshow(np.log(diff), cmap="RdBu_r", vmin=-3, vmax=3)
    # ax.set_title(f"Diff o/e ref/simu ({10000} pb)")
    # fig.colorbar(draw)
    # plt.show()
    
    # print("code origine sans nucleolus 10000 pb")
    # print(pearson_corr_manuel(observed_over_expected_ref, observed_over_expected_simu))
    # print(compute_row_correlations(observed_over_expected_ref, observed_over_expected_simu))
    # #print(pearson_corr(observed_over_expected_ref, observed_over_expected_simu))

    # nb_simu=100
    # contact_matrix = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu, 10000, 'n')
    # #norm_matrix = HiC_iterative_normalization_contact_matrix(contact_matrix)
    # #downsampled_matrix = downsample_matrix('y', 3200, 10000, contact_matrix )
    # mapping_simu = get_mapping(contact_matrix, np.array(list(num_bead_per_chr.values())), verbose=True)
    # c_expected_simu = get_expected(contact_matrix, np.array(list(num_bead_per_chr.values())), mapping=mapping_simu)
    # observed_over_expected_simu = contact_matrix / c_expected_simu
    # print("nucleolus 10000 pb")
    # print(pearson_corr_manuel(observed_over_expected_ref, observed_over_expected_simu))
    # print(compute_row_correlations(observed_over_expected_ref, observed_over_expected_simu))
    # #print(pearson_corr(observed_over_expected_ref, observed_over_expected_simu))

    # #print(pearson_correlation_vector(observed_over_expected_ref, observed_over_expected_simu)) 

    # nb_simu=200
    # contact_matrix = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu, 3200, 'n')
    # #norm_matrix = HiC_iterative_normalization_contact_matrix(contact_matrix)
    # downsampled_matrix = downsample_matrix('y', 3200, 10000, contact_matrix )
    # mapping_simu = get_mapping(downsampled_matrix, np.array(list(num_bead_per_chr.values())), verbose=True)
    # c_expected_simu = get_expected(downsampled_matrix, np.array(list(num_bead_per_chr.values())), mapping=mapping_simu)
    # observed_over_expected_simu = downsampled_matrix / c_expected_simu
    # #print(pearson_correlation_vector(observed_over_expected_ref, observed_over_expected_simu)) 
    # print("nucleolus 3200 pb downsampled to 10000 pb")
    # print(pearson_corr_manuel(observed_over_expected_ref, observed_over_expected_simu))
    # print(compute_row_correlations(observed_over_expected_ref, observed_over_expected_simu))

    # #print(pearson_corr(observed_over_expected_ref, observed_over_expected_simu))

    # nb_simu=50
    # contact_matrix = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu, 3200, 'sn')
    # #norm_matrix = HiC_iterative_normalization_contact_matrix(contact_matrix)
    # downsampled_matrix = downsample_matrix('y', 3200, 10000, contact_matrix )
    # mapping_simu = get_mapping(downsampled_matrix, np.array(list(num_bead_per_chr.values())), verbose=True)
    # c_expected_simu = get_expected(downsampled_matrix, np.array(list(num_bead_per_chr.values())), mapping=mapping_simu)
    # observed_over_expected_simu = downsampled_matrix / c_expected_simu
    # #print(pearson_correlation_vector(observed_over_expected_ref, observed_over_expected_simu)) 
    # print("without nucleolus 3200 pb downsampled to 10000 pb")
    # print(pearson_corr_manuel(observed_over_expected_ref, observed_over_expected_simu))
    # print(compute_row_correlations(observed_over_expected_ref, observed_over_expected_simu))

    #print(pearson_corr(observed_over_expected_ref, observed_over_expected_simu))

    # from normalize_expected import *
    # nb_simu = 100
    # contact_matrix = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu)
    # resolution_in = 3200
    # resolution_out = 10000
    # downsampled_contact_matrix = downsample_matrix('y', resolution_in, resolution_out, contact_matrix )
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # m = ax.matshow(np.log(contact_matrix+1), cmap="Blues")
    # ax.set_title(f"Raw contact counts\n resolution ({resolution_out} pb)")
    # fig.colorbar(m)
    # plt.show()

    # sep_yeast=10000
    # num_bead_per_chr, bead_start_per_chr, nbead = get_num_beads_and_start(chr_seq_yeast, sep_yeast)
    # mapping = get_mapping(downsampled_contact_matrix, np.array(list(num_bead_per_chr.values())), verbose=True)
    # print("mapping", mapping)

    # c_expected = get_expected(downsampled_contact_matrix, np.array(list(num_bead_per_chr.values())), mapping=mapping)
    # print("matrix expected", c_expected)
    # fig, ax = plt.subplots()
    # m = ax.matshow(np.log(c_expected), cmap="Blues")
    # ax.set_title(f"Expected contact counts\n resolution {resolution_out} pb")
    # fig.colorbar(m)
    # plt.show()

    # observed_over_expected = downsampled_contact_matrix / c_expected
    # fig, ax = plt.subplots()
    # m = ax.matshow(np.log(observed_over_expected), cmap="RdBu_r", vmin=-3, vmax=3)
    # ax.set_title(f"Simu Observed / Expected\n resolution {resolution_out}")
    # fig.colorbar(m)
    # plt.show()
    
    # ref_contact_matrix = np.load("yeast_Cerevisiae_info/duan.SC.10000.normed.npy")

    # mapping_ref = get_mapping(ref_contact_matrix, np.array(list(num_bead_per_chr.values())), verbose=True)
    # print("mapping", mapping_ref)

    # c_expected_ref = get_expected(ref_contact_matrix, np.array(list(num_bead_per_chr.values())), mapping=mapping_ref)
    # print("matrix expected", c_expected_ref)
    # fig, ax = plt.subplots()
    # m = ax.matshow(np.log(c_expected_ref), cmap="Blues")
    # ax.set_title(f"Expected contact counts\n resolution {resolution_out} pb")
    # fig.colorbar(m)
    # plt.show()


    # observed_over_expected_ref = ref_contact_matrix / c_expected_ref
    # fig, ax = plt.subplots()
    # m = ax.matshow(np.log(observed_over_expected_ref), cmap="RdBu_r", vmin=-3, vmax=3)
    # ax.set_title(f"Ref Observed / Expected\n resolution {resolution_out} pb")
    # fig.colorbar(m)
    # plt.show()