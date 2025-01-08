
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
from scipy.sparse import triu
import os
from scipy.sparse import csr_matrix
from pycirclize import Circos
from pycirclize.utils import ColorCycler
from itertools import product

from simulators import get_num_beads_and_start, chr_seq_parasite, chr_cen_parasite, sep_parasite, chr_seq_yeast, chr_cen_yeast, sep_yeast
from contact_matrix import get_config_3D_pop, get_contact_matrix_inter_chr, get_contact_matrix_all_chr 
from contact_matrix import read_contact_matrix_inter_chr_from_pickle, read_contact_matrix_all_chr_from_pickle 
from contact_matrix import read_contact_matrix_all_chr_from_simu
from contact_matrix import read_contact_matrix_all_chr_from_file, read_contact_matrix_inter_chr_from_file, construct_ref_matrix_from_file
from contact_matrix import get_start_bead_end_bead_per_chr, read_contact_matrix_norm
from normalization import normalization_contact_matrix, HiC_iterative_normalization_contact_matrix
from normalize_expected import *
from downsample import downsample_matrix

NB_SIMU = 26

###----------------------------- plot contact matrix between 2 chromosomes -----------------------------###

def plot_contact_matrix_inter_chr(type, sep, contact_matrix, chr_row, chr_col, origin, thresh):
    """ plot the contact matrix of two chr """
    if type=='y':
        chr_cen = chr_cen_yeast

    elif type=='p':
        chr_cen = chr_cen_parasite

    else:
        print("unknown type")
        return
    centro_bead_row = chr_cen[chr_row] // sep #get the bead id of the centromere
    centro_bead_col = chr_cen[chr_col] // sep #get the bead id of the centromere

    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
    
    #ax = axes[1]
    # ax.text(-0.05, 1.02, "B", transform=ax.transAxes, fontweight="bold",
    # fontsize="large")
    # if origin=='c':
    #   np.fill_diagonal(contact_matrix, 0)  

    norm_matrix = np.log(contact_matrix+1)
    min_mat, max_mat = norm_matrix.min(), norm_matrix.max()
    norm_matrix = (norm_matrix - min_mat)/(max_mat-min_mat)
    contact_matrix = norm_matrix

    draw = ax.matshow(
        #np.log(contact_matrix+1),
        #contact_matrix/contact_matrix.max(),
        contact_matrix,
        #norm=colors.SymLogNorm(thresh),#represent positive and negative values logarithmically, scale is linear around zero within a specified range (linthresh)
        #cmap="YlGnBu",
        cmap="YlOrRd",
        #extent=(0, length, 0, length),
        #vmax=14000,
        origin="lower")
    
    ax.axvline(centro_bead_col, linestyle="--", color="navy", alpha=0.5)
    ax.axhline(centro_bead_row, linestyle="--", color="navy", alpha=0.5)
    # ax.set_xticks([])
    # ax.set_yticks([])
    chr_id_row = int(chr_row[chr_row.find("chr") + 3:])
    chr_id_col = int(chr_col[chr_col.find("chr") + 3:])
    
    ax.tick_params(left=True, bottom=True, top=False, right=False, labelbottom=True, labeltop=False)
    ax.set_xlabel(f"Chr {chr_id_col}\n ({sep/1000} kbp)", fontweight="bold", fontsize="small")

    ax.set_ylabel(f"Chr {chr_id_row}\n ({sep/1000} kbp)", fontweight="bold", fontsize="small")
    fig.colorbar(draw, location='right')
    if type=='p':
        if origin=='c':
            title = f"P. falciparum-trophozoite stage\n simulated matrix (pop : {NB_SIMU})"
        if origin=='rr':
            title = f"P. falciparum-trophozoite stage\n reference matrix raw"
        if origin=='rn':
            title = f"P. falciparum-trophozoite stage\n reference matrix normed"
    if type=='y':
        if origin=='c':
            title = f"Yeast Cerevisiae\n simulated matrix (pop : {NB_SIMU})"
        if origin=='rr':
            title = f"Yeast Cerevisiae\n reference matrix raw"
        if origin=='rn':
            title = f"Yeast Cerevisiae\n reference matrix normed"
            
    plt.title(title, fontweight="bold", fontsize="medium")
    
    plt.show()
    # if origin == 'c':
    #     plt.savefig('/home/etouron/SBI_bio/plot/yeast_Cerevisiae/comparaison/simu_'+chr_col+'_'+chr_row)
    # if origin == 'rr':
    #     plt.savefig('/home/etouron/SBI_bio/plot/yeast_Cerevisiae/comparaison/ref_raw_'+chr_col+'_'+chr_row)
    # if origin == 'rn':
    #     plt.savefig('/home/etouron/SBI_bio/plot/yeast_Cerevisiae/comparaison/ref_norm_'+chr_col+'_'+chr_row)

# config_3D_pop = get_config_3D_pop(NB_SIMU)
# chr_col = "chr03"
# chr_row = "chr02"
# contact_matrix = get_contact_matrix_inter_chr(chr_seq, sep, config_3D_pop, chr_col, chr_row)
# plot_contact_matrix_inter_chr("p", contact_matrix,chr_cen, sep, chr_col, chr_row, 'c')

# NB_SIMU = 100
# config_3D_pop, simu_path = get_config_3D_pop(NB_SIMU, 'y')

# # #for chr_col, chr_row in zip(chr_seq_yeast.keys(), chr_seq_yeast.keys()):

# chr_col = "chr02"
# chr_row = "chr02"
# contact_matrix = get_contact_matrix_inter_chr(chr_seq_yeast, sep_yeast, config_3D_pop, chr_col, chr_row)
# plot_contact_matrix_inter_chr("y", contact_matrix,chr_cen_yeast, sep_yeast, chr_col, chr_row, 'c', 0.05)

# chr_row = "chr01"
# chr_col = "chr01"

# contact_matrix = read_contact_matrix_inter_chr_from_file(chr_seq_yeast, sep_yeast, "yeast_Cerevisiae_info/contact_matrix.matrix", chr_row,chr_col)
# plot_contact_matrix_inter_chr("y", contact_matrix, chr_cen_yeast, sep_yeast, chr_row, chr_col,'rr',0.05)

# chr_row = "chr01"
# chr_col = "chr01"

# contact_matrix = read_contact_matrix_norm(chr_seq_yeast, sep_yeast, "yeast_Cerevisiae_info/duan.SC.10000.normed.npy", chr_row,chr_col)
# plot_contact_matrix_inter_chr("y", contact_matrix, chr_cen_yeast, sep_yeast, chr_row, chr_col,'rn',0.05)


# plt.show()

# chr_row = "chr07"
# chr_col = "chr07"
# contact_matrix = read_contact_matrix_inter_chr_from_file(chr_seq, sep_yeast, "parasite_Falciparum_info/schizonts_10000_raw.matrix", chr_row,chr_col)
# plot_contact_matrix_inter_chr("p", contact_matrix, chr_cen, sep_yeast, chr_row, chr_col,'r',3)

# chr_y = 'chr16'
# chr_x = 'chr01'
# nb_simu = 90
# contact_matrix = read_contact_matrix_inter_chr_from_pickle('y',nb_simu, chr_y, chr_x)
# NB_SIMU=nb_simu*1000
# plot_contact_matrix_inter_chr("y", contact_matrix, chr_y, chr_x, 'c', 0.05)


# contact_matrix = read_contact_matrix_norm(chr_seq_yeast, sep_yeast, "yeast_Cerevisiae_info/duan.SC.10000.normed.npy", chr_y, chr_x)
# plot_contact_matrix_inter_chr("y", contact_matrix, chr_y, chr_x,'rn',0.05)

# for chr in chr_seq_yeast.keys():
#     chr_col = chr
#     chr_row = chr
#     nb_simu = 90
#     contact_matrix = read_contact_matrix_inter_chr_from_pickle('y',nb_simu, chr_col, chr_row)
#     NB_SIMU=nb_simu*1000
#     plot_contact_matrix_inter_chr("y", contact_matrix, chr_col, chr_row, 'c', 0.05)


#     contact_matrix = read_contact_matrix_norm(chr_seq_yeast, sep_yeast, "yeast_Cerevisiae_info/duan.SC.10000.normed.npy", chr_row, chr_col)
#     plot_contact_matrix_inter_chr("y", contact_matrix, chr_col, chr_row,'rn',0.05)


def plot_observed_over_expected_inter_chr(type, sep, obs_exp, chr_row, chr_col, origin):
    """ plot the contact matrix of two chr """
    if type=='y':
        chr_cen = chr_cen_yeast

    elif type=='p':
        chr_cen = chr_cen_parasite

    else:
        print("unknown type")
        return
    centro_bead_row = chr_cen[chr_row] // sep #get the bead id of the centromere
    centro_bead_col = chr_cen[chr_col] // sep #get the bead id of the centromere

    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
    

    

    draw = ax.matshow(np.log(obs_exp), cmap="RdBu_r", vmin=-3, vmax=3, origin="lower")

    
    # ax.axvline(centro_bead_col, linestyle="--", color="navy", alpha=0.5)
    # ax.axhline(centro_bead_row, linestyle="--", color="navy", alpha=0.5)
    plt.scatter(centro_bead_col, centro_bead_row, color="black", s=0.5)

    # ax.set_xticks([])
    # ax.set_yticks([])
    chr_id_row = int(chr_row[chr_row.find("chr") + 3:])
    chr_id_col = int(chr_col[chr_col.find("chr") + 3:])
    
    ax.tick_params(left=True, bottom=True, top=False, right=False, labelbottom=True, labeltop=False)
    ax.set_xlabel(f"Chr {chr_id_col}\n ({sep/1000} kbp)", fontweight="bold", fontsize="small")

    ax.set_ylabel(f"Chr {chr_id_row}\n ({sep/1000} kbp)", fontweight="bold", fontsize="small")
    fig.colorbar(draw, location='right')
    if type=='p':
        if origin=='c':
            title = f"P. falciparum-trophozoite stage\n simulated matrix (pop : {NB_SIMU})"
        if origin=='rr':
            title = f"P. falciparum-trophozoite stage\n reference matrix raw"
        if origin=='rn':
            title = f"P. falciparum-trophozoite stage\n reference matrix normed"
    if type=='y':
        if origin=='c':
            title = f"Yeast Cerevisiae\n obs/exp simulated matrix (pop : {NB_SIMU})"
        if origin=='rr':
            title = f"Yeast Cerevisiae\n obs/exp reference matrix raw"
        if origin=='rn':
            title = f"Yeast Cerevisiae\n obs/exp reference matrix normed"
            
    plt.title(title, fontweight="bold", fontsize="medium")
    
    plt.show()


# sep_yeast=10000
# num_bead_per_chr, bead_start_per_chr, nbead = get_num_beads_and_start(chr_seq_yeast, sep_yeast)
# start = bead_start_per_chr["chr07"]
# end = start + num_bead_per_chr["chr07"]

# contact_matrix_ref = np.load("yeast_Cerevisiae_info/ref_10000_norm_HiC_nelle.npy")
# mapping_ref = get_mapping(contact_matrix_ref, np.array(list(num_bead_per_chr.values())), verbose=True)
# c_expected_ref = get_expected(contact_matrix_ref, np.array(list(num_bead_per_chr.values())), mapping=mapping_ref)
# observed_over_expected_ref = contact_matrix_ref / c_expected_ref

# nb_simu=75
# contact_matrix = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu, 3200, 'n', 'tjong')
# downsampled_matrix = downsample_matrix('y', 3200, 10000, contact_matrix )
# mapping_simu = get_mapping(downsampled_matrix, np.array(list(num_bead_per_chr.values())), verbose=True)
# c_expected_simu = get_expected(downsampled_matrix, np.array(list(num_bead_per_chr.values())), mapping=mapping_simu)
# observed_over_expected_simu = downsampled_matrix / c_expected_simu

# NB_SIMU = 7500
# plot_observed_over_expected_inter_chr('y', sep_yeast, observed_over_expected_ref[start:end, start:end], "chr07", "chr07", 'c')

###----------------------------------------------------------------------------------------------------###


###----------------------------- plot contact matrix for all the chromosomes -----------------------------###

   

def plot_contact_matrix_all_chr(type, sep, contact_matrix, origin):
    """ plot the contact matrix for all the chr """

    if type=='p':
        chr_seq=chr_seq_parasite
        chr_cen = chr_cen_parasite

    elif type=='y':
        chr_seq=chr_seq_yeast
        chr_cen = chr_cen_yeast


    num_chr_num_bead, chr_bead_start, nbead = get_num_beads_and_start(chr_seq, sep) #get the number of bead and the bead start id for each chr

    nb_chr =len(chr_seq.keys())
    
    fig, ax = plt.subplots(figsize=(nb_chr*16, nb_chr*16), tight_layout=True)

    norm_matrix = np.log(contact_matrix+1)
    min_mat, max_mat = norm_matrix.min(), norm_matrix.max()
    norm_matrix = (norm_matrix - min_mat)/(max_mat-min_mat)
    contact_matrix = norm_matrix

    sparse_matrix = csr_matrix(contact_matrix)
    non_zero_contact_bead_id = sparse_matrix.nonzero()
    non_zero_contact = sparse_matrix.data
    ##upper_sparse_matrix = triu(sparse_matrix, k=1)
    ##non_zero_contact_upper = upper_sparse_matrix.data
    #draw=plt.scatter(non_zero_contact_bead_id[0], non_zero_contact_bead_id[1], c=non_zero_contact, cmap="YlOrRd", s=0.7) #, norm = colors.SymLogNorm(0.05))
    ##draw=plt.scatter(upper_sparse_matrix.nonzero()[1], upper_sparse_matrix.nonzero()[0], c=non_zero_contact_upper, cmap="YlOrRd", s=0.7, norm = colors.SymLogNorm(0.05))
    
    draw = ax.matshow(
        #np.log(contact_matrix+1),
        #contact_matrix/contact_matrix.max(),
        #np.log(contact_matrix),
        contact_matrix,
        #norm=colors.SymLogNorm(thresh),#represent positive and negative values logarithmically, scale is linear around zero within a specified range (linthresh)
        #cmap="YlGnBu",
        cmap="YlOrRd",
        #extent=(0, length, 0, length),
        #vmax=14000,
        origin="upper")

    #draw = ax.matshow(np.log(contact_matrix), cmap="RdBu_r", vmin=-3, vmax=3)

    
    chr_name = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']
    
    pos_chr_label = np.array(list(chr_bead_start.values()))+ 0.5*np.array(list(num_chr_num_bead.values()))
    
    sec_x = ax.secondary_xaxis(location='top') #draw the separation between chr    
    sec_x.set_xticks(list(chr_bead_start.values()), labels=[])
    sec_x.tick_params('x', length=20, width=1.5)

    sec_2_x = ax.secondary_xaxis(location='top') #write the chr name    
    sec_2_x.set_xticks(pos_chr_label, labels = chr_name, ) #labels=list(chr_bead_start.keys()))
    sec_2_x.tick_params('x', length=0)
    
    sec_y = ax.secondary_yaxis(location=0) #draw the separation between chr   
    sec_y.set_yticks(list(chr_bead_start.values()), labels=[])
    sec_y.tick_params('y', length=20, width=1.5)
    
    sec_2_y = ax.secondary_yaxis(location=0) #write the chr name    
    sec_2_y.set_yticks(pos_chr_label, labels=chr_name) #labels=list(chr_bead_start.keys()))
    sec_2_y.tick_params('y', length=0)

    #plot the grid to separate chr
    ax.vlines(np.array(list(chr_bead_start.values())), ymin=0, ymax=len(contact_matrix)-1, linestyle=(0,(1,1)), color="black", alpha=0.3, lw=0.7)
    ax.hlines(np.array(list(chr_bead_start.values())), xmin=0, xmax=len(contact_matrix)-1, linestyle=(0,(1,1)), color="black", alpha=0.3, lw=0.7)

    for chr in chr_cen.keys():
        centro_bead_row = chr_cen[chr] // sep #get the bead id of the centromere
        plt.scatter(chr_bead_start[chr] + centro_bead_row, chr_bead_start[chr]+centro_bead_row, color="black", s=0.5)
    for (chr_row,chr_col) in product(chr_cen.keys(), repeat=2):
        centro_bead_row = chr_cen[chr_row] // sep #get the bead id of the centromere
        centro_bead_col = chr_cen[chr_col] // sep #get the bead id of the centromere
        plt.scatter(chr_bead_start[chr_col] + centro_bead_col, chr_bead_start[chr_row]+centro_bead_row, color="black", s = 0.5)


        
    ax.set_xticks([]) #no x, y axis labels
    ax.set_yticks([])
   
    #ax.set_facecolor("gainsboro")#change the background color


    fig.colorbar(draw, location='right')
    if type=='p':
        if origin=='c':
            title = f"P. falciparum-trophozoite stage\npopulation number : {NB_SIMU}"
        if origin=='r':
            title = f"P. falciparum-trophozoite stage : reference matrix"
    if type=='y':
        if origin=='c':
            title = f"Yeast Cerevisiae\npopulation number : {NB_SIMU}"
        if origin=='rr':
            title = f"Yeast Cerevisiae : raw reference matrix"
        if origin=='rn':
            title = f"Yeast Cerevisiae : normalized reference matrix"
    plt.suptitle(title, fontweight="bold", fontsize="medium")

    ax.set_xlim(0, nbead)
    ax.set_ylim(0, nbead)
    ax.set_box_aspect(1)
    plt.axis([0, nbead, nbead, 0])
    plt.show()


# NB_SIMU=100
# config_3D_pop, pickle_path = get_config_3D_pop(NB_SIMU, 'y')
# contact_matrix = get_contact_matrix_all_chr('y', config_3D_pop)
# plot_contact_matrix_all_chr('y', contact_matrix, 'c')

# contact_matrix = read_contact_matrix_all_chr_from_file(chr_seq_yeast, sep_yeast, "yeast_Cerevisiae_info/contact_matrix.matrix")
# plot_contact_matrix_all_chr('y', contact_matrix, 'rr')

# contact_matrix = np.load("yeast_Cerevisiae_info/duan.SC.10000.normed.npy")
# downsampled_matrix = downsample_matrix('y', 10000, 32000, contact_matrix )
# plot_contact_matrix_all_chr('y', 32000, downsampled_matrix, 'rn')

# contact_matrix, nb_simu = read_contact_matrix_all_chr_from_pickle()
# print(contact_matrix)
# NB_SIMU = nb_simu*100
# plot_contact_matrix_all_chr('y', sep_yeast, contact_matrix, origin='c')

# sep_yeast = 10000
# contact_matrix_ref_raw = read_contact_matrix_all_chr_from_file(chr_seq_yeast, 10000, "yeast_Cerevisiae_info/contact_matrix.matrix")
# print(contact_matrix_ref_raw)
# norm_matrix = HiC_iterative_normalization_contact_matrix(contact_matrix_ref_raw)
# # with open('ref_10000_HiC.npy', 'wb') as f:
# #     np.save(f, norm_matrix)
# plot_contact_matrix_all_chr('y', sep_yeast, norm_matrix, origin='rn')

# from simulators import chr_seq_yeast_duan
# file_intra_1 ='/home/etouron/SBI_bio/yeast_Cerevisiae_info/intra/interactions_EcoRI_MseI_before_FDR_intra_all.txt'
# file_intra_2 ='/home/etouron/SBI_bio/yeast_Cerevisiae_info/intra/interactions_EcoRI_MspI_before_FDR_intra_all.txt'
# file_intra_3 ='/home/etouron/SBI_bio/yeast_Cerevisiae_info/intra/interactions_HindIII_MseI_before_FDR_intra_all.txt'
# file_intra_4 ='/home/etouron/SBI_bio/yeast_Cerevisiae_info/intra/interactions_HindIII_MspI_before_FDR_intra_all.txt'

file_intra_1 ='/home/etouron/SBI_bio/yeast_Cerevisiae_info/intra_20_kb/interactions_EcoRI_MseI_before_FDR_intra.txt'
file_intra_2 ='/home/etouron/SBI_bio/yeast_Cerevisiae_info/intra_20_kb/interactions_EcoRI_MspI_before_FDR_intra.txt'
file_intra_3 ='/home/etouron/SBI_bio/yeast_Cerevisiae_info/intra_20_kb/interactions_HindIII_MseI_before_FDR_intra.txt'
file_intra_4 ='/home/etouron/SBI_bio/yeast_Cerevisiae_info/intra_20_kb/interactions_HindIII_MspI_before_FDR_intra.txt'

file_inter_1 = '/home/etouron/SBI_bio/yeast_Cerevisiae_info/inter/interactions_HindIII_MspI_before_FDR_inter.txt'
file_inter_2 = '/home/etouron/SBI_bio/yeast_Cerevisiae_info/inter/interactions_HindIII_MseI_before_FDR_inter.txt'
file_inter_3 = '/home/etouron/SBI_bio/yeast_Cerevisiae_info/inter/interactions_EcoRI_MspI_before_FDR_inter.txt'
file_inter_4 = '/home/etouron/SBI_bio/yeast_Cerevisiae_info/inter/interactions_EcoRI_MseI_before_FDR_inter.txt'

file_intra = [file_intra_1, file_intra_2, file_intra_3, file_intra_4]
file_inter = [file_inter_1, file_inter_2, file_inter_3, file_inter_4]

# contact_matrix_ref = construct_ref_matrix_from_file(chr_seq_yeast, 3200, file_intra, file_inter)
# from normalization import downsample_matrix
# downsampled_matrix_ref = downsample_matrix('y', 3200, 10000, contact_matrix_ref)
# downsampled_matrix_ref = downsample_matrix('y', 10000, 32000, downsampled_matrix_ref)

# norm_matrix = HiC_iterative_normalization_contact_matrix(downsampled_matrix_ref)

# # # # # #sep_yeast = 32000
# with open('ref_32000_duan_intra_20_kb.npy', 'wb') as f:
#     np.save(f, downsampled_matrix_ref)
# with open('ref_32000_norm_HiC_duan_intra_20_kb.npy', 'wb') as f:
#     np.save(f, norm_matrix)

# contact_matrix_ref = np.load("yeast_Cerevisiae_info/ref_32000_duan_intra_all.npy")
# norm_matrix_ref = normalization_contact_matrix(contact_matrix_ref)
# plot_contact_matrix_all_chr('y', 32000, norm_matrix_ref, origin='rr')
# # contact_matrix = downsample_matrix('y', 10000, 32000, contact_matrix)



# # contact_matrix = np.load("yeast_Cerevisiae_info/ref_32000_duan_intra_all.npy")
# plot_contact_matrix_all_chr('y', 32000, downsampled_matrix_ref, origin='rr')
# plot_contact_matrix_all_chr('y', 32000, norm_matrix, origin='rn')

# with open('ref_3200_norm_HiC_duan_intra_20_kb.npy', 'wb') as f:
#     np.save(f, contact_matrix)
# plot_contact_matrix_all_chr('y', sep_yeast, norm_matrix, origin='rn')

# contact_matrix_ref = read_contact_matrix_all_chr_from_file(chr_seq_yeast, 10000, "yeast_Cerevisiae_info/contact_matrix.matrix")
# with open('ref_10000_norm_0_1.npy', 'wb') as f:
#     np.save(f, contact_matrix_ref)
# contact_matrix_ref = np.load("yeast_Cerevisiae_info/ref_10000_raw_nelle.npy")
# downsampled_matrix_ref = downsample_matrix('y', 10000, 32000, contact_matrix_ref)
# norm_matrix_ref = normalization_contact_matrix(downsampled_matrix_ref)
# plot_contact_matrix_all_chr('y', 32000, norm_matrix_ref, 'rr')

# contact_matrix_simu = read_contact_matrix_all_chr_from_simu('y', 1, 75, 3200, 'n')
# downsampled_matrix_simu = downsample_matrix('y', 3200, 10000, contact_matrix_simu )
# downsampled_matrix_simu = downsample_matrix('y', 10000, 32000, downsampled_matrix_simu )
# norm_matrix_simu = normalization_contact_matrix(downsampled_matrix_simu)
# NB_SIMU=75*100
# plot_contact_matrix_all_chr('y', 32000, norm_matrix_simu, origin='c')



for nb_simu in [50]:
    #num_bead_per_chr, bead_start_per_chr, nbead = get_num_beads_and_start(chr_seq_yeast, sep_yeast)
    contact_matrix_simu = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu, 3200, 'n', 'duan', plot=1)
    
    # with open("raw_ref_10000.npy",'wb') as f:
    #     np.save(f, contact_matrix)
    print(contact_matrix_simu)
    #contact_matrix_norm = HiC_iterative_normalization_contact_matrix(contact_matrix)
    NB_SIMU = nb_simu*100
    plot_contact_matrix_all_chr('y', sep_yeast, contact_matrix_simu, origin='c')

    #downsample, new_length = downsample_contact_matrix(contact_matrix, np.array(list(num_bead_per_chr.values())),2)
    downsampled_matrix_simu = downsample_matrix('y', 3200, 10000, contact_matrix_simu )
    downsampled_matrix_simu = downsample_matrix('y', 10000, 32000, downsampled_matrix_simu )
    #downsampled_matrix_simu = normalization_contact_matrix(downsampled_matrix_simu)

    sep_yeast=32000
    #sep_yeast=10000
    plot_contact_matrix_all_chr('y', sep_yeast, downsampled_matrix_simu, origin='c')


# contact_matrix = np.load("yeast_Cerevisiae_info/duan.SC.10000.normed.npy")
# contact_matrix_norm = normalization_contact_matrix(contact_matrix)

# plot_contact_matrix_all_chr('y', sep_yeast, contact_matrix_norm, 'rn')


# fig, ax = plt.subplots() 
# draw = ax.matshow(np.log(observed_over_expected_ref), cmap="RdBu_r", vmin=-3, vmax=3)
# ax.set_title(f"Diff o/e ref/simu ({10000} pb)")
# fig.colorbar(draw)
# plt.show()

# nb_simu = 1
# contact_matrix = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu)
# resolution_in = 3200
# resolution_out = 10000

# downsampled_matrix = downsample_matrix('y', resolution_in, resolution_out, contact_matrix )
# plot_contact_matrix_all_chr('y', 10000, downsampled_matrix, origin='c')


def plot_observed_over_expected_all_chr(type, sep, obs_exp, origin):
    """ plot the contact matrix for all the chr """

    if type=='p':
        chr_seq=chr_seq_parasite
        chr_cen = chr_cen_parasite

    elif type=='y':
        chr_seq=chr_seq_yeast
        chr_cen = chr_cen_yeast


    num_chr_num_bead, chr_bead_start, nbead = get_num_beads_and_start(chr_seq, sep) #get the number of bead and the bead start id for each chr

    nb_chr =len(chr_seq.keys())
    
    fig, ax = plt.subplots(figsize=(nb_chr*16, nb_chr*16), tight_layout=True)

    draw = ax.matshow(np.log(obs_exp), cmap="RdBu_r", vmin=-3, vmax=3)
    
    chr_name = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']
    
    pos_chr_label = np.array(list(chr_bead_start.values()))+ 0.5*np.array(list(num_chr_num_bead.values()))
    
    sec_x = ax.secondary_xaxis(location='top') #draw the separation between chr    
    sec_x.set_xticks(list(chr_bead_start.values()), labels=[])
    sec_x.tick_params('x', length=20, width=1.5)

    sec_2_x = ax.secondary_xaxis(location='top') #write the chr name    
    sec_2_x.set_xticks(pos_chr_label, labels = chr_name, ) #labels=list(chr_bead_start.keys()))
    sec_2_x.tick_params('x', length=0)
    
    sec_y = ax.secondary_yaxis(location=0) #draw the separation between chr   
    sec_y.set_yticks(list(chr_bead_start.values()), labels=[])
    sec_y.tick_params('y', length=20, width=1.5)
    
    sec_2_y = ax.secondary_yaxis(location=0) #write the chr name    
    sec_2_y.set_yticks(pos_chr_label, labels=chr_name) #labels=list(chr_bead_start.keys()))
    sec_2_y.tick_params('y', length=0)

    #plot the grid to separate chr
    ax.vlines(np.array(list(chr_bead_start.values())), ymin=0, ymax=len(obs_exp)-1, linestyle=(0,(1,1)), color="black", alpha=0.5, lw=1)
    ax.hlines(np.array(list(chr_bead_start.values())), xmin=0, xmax=len(obs_exp)-1, linestyle=(0,(1,1)), color="black", alpha=0.5, lw=1)

    for chr in chr_cen.keys():
        centro_bead_row = chr_cen[chr] // sep #get the bead id of the centromere
        plt.scatter(chr_bead_start[chr] + centro_bead_row, chr_bead_start[chr]+centro_bead_row, color="black", s=0.5)
    for (chr_row,chr_col) in product(chr_cen.keys(), repeat=2):
        centro_bead_row = chr_cen[chr_row] // sep #get the bead id of the centromere
        centro_bead_col = chr_cen[chr_col] // sep #get the bead id of the centromere
        plt.scatter(chr_bead_start[chr_col] + centro_bead_col, chr_bead_start[chr_row]+centro_bead_row, color="black", s = 0.5)

    ax.set_xticks([]) #no x, y axis labels
    ax.set_yticks([])
   
    fig.colorbar(draw, location='right')
    if type=='p':
        if origin=='c':
            title = f"P. falciparum-trophozoite stage\npopulation number : {NB_SIMU}"
        if origin=='r':
            title = f"P. falciparum-trophozoite stage : observed/expected reference"
    if type=='y':
        if origin=='c':
            title = f"Yeast Cerevisiae : observed/expected simulation \npopulation number : {NB_SIMU}"
        if origin=='rr':
            title = f"Yeast Cerevisiae : raw observed/expected reference"
        if origin=='rn':
            title = f"Yeast Cerevisiae : ICE normalized observed/expected reference"
    plt.suptitle(title, fontweight="bold", fontsize="medium")

    ax.set_xlim(0, nbead)
    ax.set_ylim(0, nbead)
    ax.set_box_aspect(1)
    plt.axis([0, nbead, nbead, 0])
    plt.show()


# sep_yeast=10000
# num_bead_per_chr, bead_start_per_chr, nbead = get_num_beads_and_start(chr_seq_yeast, sep_yeast)
# # contact_matrix_ref = np.load("yeast_Cerevisiae_info/ref_10000_norm_HiC_nelle.npy")
# # mapping_ref = get_mapping(contact_matrix_ref, np.array(list(num_bead_per_chr.values())), verbose=True)
# # c_expected_ref = get_expected(contact_matrix_ref, np.array(list(num_bead_per_chr.values())), mapping=mapping_ref)
# # observed_over_expected_ref = contact_matrix_ref / c_expected_ref

# nb_simu=75
# contact_matrix = read_contact_matrix_all_chr_from_simu('y', 1, nb_simu, 3200, 'n', 'tjong')
# downsampled_matrix = downsample_matrix('y', 3200, 10000, contact_matrix )
# mapping_simu = get_mapping(downsampled_matrix, np.array(list(num_bead_per_chr.values())), verbose=True)
# c_expected_simu = get_expected(downsampled_matrix, np.array(list(num_bead_per_chr.values())), mapping=mapping_simu)
# observed_over_expected_simu = downsampled_matrix / c_expected_simu
# plot_observed_over_expected_all_chr('y', sep_yeast, observed_over_expected_simu, 'c')

###-------------------------------------------------------------------------------------------------------###


###----------------------------- plot circos diagram for all the chromosomes -----------------------------###


def find_chromosome(chr_num_bead, chr_start_bead, bead):
        """ Returns a chromosome id given a bead number"""

        for i in chr_num_bead.keys():
            if bead >= chr_start_bead[i] and bead <chr_start_bead[i] + chr_num_bead[i] :
                return i
                

def write_genomic_info_all_bed_file(chr_seq, sep, bed_file):
    """write start and end bead for each chr in a bed file"""

    chr_num_bead, chr_start_bead, nbead = get_num_beads_and_start(chr_seq, sep) #get the number of bead and the bead start id for each chr
    chr_end_bead = np.array(list(chr_start_bead.values())) + np.array(list(chr_num_bead.values()))-1
    data = np.transpose(np.array([list(chr_start_bead.keys()), list(chr_start_bead.values()),  list(chr_end_bead)]))#write in column
    data_df = pd.DataFrame(data)
    fichier = open(bed_file, "w")
    fichier.write("#chrom\tchromStart\tchromEnd\n")#separation with tabulation
    fichier.write(data_df.to_csv(header=False, index=False, sep="\t"))
    fichier.close()


def plot_circos_diagram_all_chr(chr_seq, sep, contact_matrix, bed_file):
    """ plot the circos diagram for all the chr """

    chr_num_bead, chr_start_bead, nbead = get_num_beads_and_start(chr_seq, sep) #get the number of bead and the bead start id for each chr

    write_genomic_info_all_bed_file(chr_seq, sep, bed_file) #write start+end bead of all the chr in a bed file

    #create the diagram + legend
    circos = Circos.initialize_from_bed(bed_file, space=3)
    circos.text(f"Chr pop \n (1 bead = {sep} pb)", deg=315, r=150, size=12)
    circos.add_cytoband_tracks((95, 100), bed_file)

    # {chr_id : color}
    ColorCycler.set_cmap("hsv")
    chr_names = chr_start_bead.keys()
    colors = ColorCycler.get_color_list(len(chr_names))
    chr_name2color = {name: color for name, color in zip(chr_names, colors)}
 
    for sector in circos.sectors:
        
        sector.text(sector.name, r=120, size=10, color=chr_name2color[sector.name])
        sector.get_track("cytoband").xticks_by_interval(
            100,
            label_size=8,
            label_orientation="vertical",
            label_formatter=lambda v: f"{v :.0f}",
        )
    
    # plot chromosome link
    sparse_matrix = csr_matrix(contact_matrix)
    rows, cols = sparse_matrix.nonzero()
    for bead_1, bead_2 in zip(rows, cols):
        
        chr_num_1 = find_chromosome(chr_num_bead, chr_start_bead, bead_1)
        chr_num_2 = find_chromosome(chr_num_bead, chr_start_bead, bead_2)

        if chr_num_1 != chr_num_2: #only inter-chr

            region1 = (chr_num_1, bead_1, bead_1)
            region2 = (chr_num_2, bead_2, bead_2)
            color = chr_name2color[chr_num_1]
            circos.link(region1, region2, color=color)
        
    circos.plotfig()
    plt.show()
    os.remove(bed_file)

# config_3D_pop = get_config_3D_pop(NB_SIMU)
# contact_matrix = get_contact_matrix_all_chr(chr_seq, sep, config_3D_pop)
# plot_circos_diagram_all_chr(chr_seq, sep, contact_matrix, "chr_pop.bed")


###-------------------------------------------------------------------------------------------------------###


###----------------------------- plot circos diagram of 2 chromosomes -----------------------------###


def write_genomic_info_inter_chr_bed_file(chr_seq, sep, chr_num_1, chr_num_2, ):
    """write start and end bead for each chr in a bed file"""
    
    chr_num_bead, chr_start_bead, nbead = get_num_beads_and_start(chr_seq, sep) #get the number of bead and the bead start id for each chr
    chr_end_bead = [chr_start_bead[chr_num_1] + chr_num_bead[chr_num_1]-1, chr_start_bead[chr_num_2] + chr_num_bead[chr_num_2]-1]
    data = np.transpose(np.array([[chr_num_1, chr_num_2], [chr_start_bead[chr_num_1], chr_start_bead[chr_num_2]], chr_end_bead])) #write in column

    data_df = pd.DataFrame(data)
    fichier = open(f"{chr_num_1}_{chr_num_2}.bed", "w")
    fichier.write("#chrom\tchromStart\tchromEnd\n")#separation with tabulation
    fichier.write(data_df.to_csv(header=False, index=False, sep='\t'))
    fichier.close()


def plot_circos_diagram_inter_chr(chr_seq, chr_cen, sep, contact_matrix, chr_num_1, chr_num_2):
    """ plot the circos diagram of two chr """
    
    chr_num_bead, chr_start_bead, nbead = get_num_beads_and_start(chr_seq, sep) #get the number of bead and the bead start id for each chr

    write_genomic_info_inter_chr_bed_file(chr_seq, sep, chr_num_1, chr_num_2) #write start+end bead of all the chr in a bed file

    bed_file = chr_num_1+"_"+chr_num_2+".bed"
    #create the diagram + legend
    circos = Circos.initialize_from_bed(bed_file, space=3)
    circos.text(f"{chr_num_1}-{chr_num_2} \n (1 bead = {sep} pb)", deg=315, r=150, size=12)
    
    

    circos.add_cytoband_tracks((95, 100), bed_file)

    # {chr_id : color}
    chr_name2color = {chr_num_1 : "red", chr_num_2 : "royalblue"}
 
    for sector in circos.sectors:
        
        sector.text(sector.name, r=120, size=10, color=chr_name2color[sector.name])
        sector.get_track("cytoband").xticks_by_interval(
            10,
            label_size=8,
            label_orientation="vertical",
            label_formatter=lambda v: f"{v :.0f}",
            
        )
        
        track = sector.get_track("cytoband")
        
        track.rect(chr_start_bead[sector.name] + chr_cen[sector.name] // sep -0.5 ,chr_start_bead[sector.name] + chr_cen[sector.name] // sep+0.5,fc=chr_name2color[sector.name] )
            
    
    # plot chromosome link
    
    sparse_matrix = csr_matrix(contact_matrix)
    nb_of_black_levels = 12
    rows, cols = sparse_matrix.nonzero()
    
    for bead_1, bead_2 in zip(rows, cols):

        region1 = (chr_num_1, bead_1+chr_start_bead[chr_num_1], bead_1+chr_start_bead[chr_num_1])
        region2 = (chr_num_2, bead_2+chr_start_bead[chr_num_2], bead_2+chr_start_bead[chr_num_2])
        
        #color = chr_name2color[chr_num_1]
        transparency = (sparse_matrix[bead_1, bead_2] / (max(sparse_matrix.data) + 1))//(1/nb_of_black_levels) + 1 
        #transparency = (sparse_matrix[bead_1, bead_2] - min(sparse_matrix.data))/ (max(sparse_matrix.data)-min(sparse_matrix.data))
        
        circos.link(region1, region2, color="black",  alpha = transparency/nb_of_black_levels, height_ratio=0.1, linewidth=2*transparency/nb_of_black_levels)
        
    circos.plotfig()
    #plt.show()
    plt.savefig('plot/yeast_Cerevisiae/reference/'+chr_num_1+'_'+chr_num_2+'_circos.pdf')
    os.remove(bed_file)
    

# chr_1 = "chr05"
# chr_2 = "chr07"
# config_3D_pop = get_config_3D_pop(NB_SIMU)
# contact_matrix = get_contact_matrix_inter_chr(chr_seq, sep, config_3D_pop, chr_1, chr_2)
# plot_circos_diagram_inter_chr(chr_seq, chr_cen, sep, contact_matrix, chr_1, chr_2)

# chr_1="chr12"
# chr_2="chr12"
# contact_matrix = read_contact_matrix_inter_chr_from_file(chr_seq_yeast, sep_yeast, "yeast_Cerevisiae_info/contact_matrix.matrix", chr_1,chr_2)
# plot_circos_diagram_inter_chr(chr_seq_yeast, chr_cen_yeast, sep_yeast, contact_matrix, chr_1,chr_2)


###------------------------------------------------------------------------------------------------###


