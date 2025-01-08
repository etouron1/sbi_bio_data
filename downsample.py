import numpy as np
from itertools import combinations

chr_seq_yeast = {'chr01': 240000, 'chr02': 820000, 'chr03': 320000, 'chr04': 1540000, 
                 'chr05': 580000, 'chr06': 280000, 'chr07': 1100000, 'chr08': 570000, 
                 'chr09': 440000, 'chr10': 750000, 'chr11': 670000, 'chr12': 1080000, 
                 'chr13': 930000, 'chr14': 790000, 'chr15': 1100000, 'chr16': 950000}

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

def downsample_matrix(type, resolution_in, resolution_out, contact_matrix, mode):
    """
    Parameters:
    - type : 'y' for the yeast
    - resolution_in : resolution of the input matrix
    - resolution_out : resolution of the output matrix
    - contact matrix 
    Return:
    - the downsampled contact matrix at the resolution resolution_out
    """

    if type=='y':
        chr_seq = chr_seq_yeast
    else:
        print("type unknown")
        return
    
    ratio = resolution_out//resolution_in
    num_bead_per_chr_in, bead_start_per_chr_in, nbead_in = get_num_beads_and_start(chr_seq, resolution_in)
    num_bead_per_chr_out, bead_start_per_chr_out, nbead_out = get_num_beads_and_start(chr_seq, resolution_out)
    downsampled_matrix = np.zeros((nbead_out, nbead_out))
    #intra-chr : squared matrix
    for chr in chr_seq.keys():
        start = bead_start_per_chr_in[chr]
        end = bead_start_per_chr_in[chr] + num_bead_per_chr_in[chr]
        
        output_matrix = np.zeros((num_bead_per_chr_out[chr], num_bead_per_chr_out[chr]))
        N = num_bead_per_chr_in[chr]
        n = num_bead_per_chr_out[chr]
        #nb of row and col to remove to have a nb of pixels multiple of 9
        q = N-ratio*n
        #remove row and col symmetrically
        if q%2==0:
            i_start = q//2
            i_end = q//2
        else:
            i_start = q//2
            i_end = q//2+1
        #extract the input intra-chr matrix
        input_matrix = contact_matrix[start+i_start:end-i_end, start+i_start:end-i_end]
        
        for i in range(len(output_matrix)):
            for j in range(len(output_matrix[0])):
                #apply the kernel mean (average over 9 pixels)
                i_input = ratio*i+1
                j_input = ratio*j+1
                if mode =="mean":
                    output_matrix[i,j] = 1.0/ratio**2*np.sum(input_matrix[i_input-1:i_input+2,j_input-1:j_input+2])
                else:
                    output_matrix[i,j] = np.sum(input_matrix[i_input-1:i_input+2,j_input-1:j_input+2])

        start = bead_start_per_chr_out[chr]
        end = bead_start_per_chr_out[chr] + num_bead_per_chr_out[chr]
        #fill the output matrix
        downsampled_matrix[start:end, start:end]=output_matrix
    #inter-chr matrix : rectangular matrix
    for (chr_1,chr_2) in combinations(chr_seq.keys(), r=2):
        start_1,start_2 = bead_start_per_chr_in[chr_1],bead_start_per_chr_in[chr_2]
        end_1, end_2 = bead_start_per_chr_in[chr_1] + num_bead_per_chr_in[chr_1], bead_start_per_chr_in[chr_2] + num_bead_per_chr_in[chr_2]
        
        output_matrix = np.zeros((num_bead_per_chr_out[chr_1], num_bead_per_chr_out[chr_2]))
        
        N = num_bead_per_chr_in[chr_1]
        M = num_bead_per_chr_in[chr_2]
        n = num_bead_per_chr_out[chr_1]
        m = num_bead_per_chr_out[chr_2]
        #nb of row and col to remove
        q = N-ratio*n
        r = M-ratio*m
        #remove row and col symmetrically
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
        #extract the input matrix
        input_matrix = contact_matrix[start_1+i_start:end_1-i_end, start_2+j_start:end_2-j_end]
        
        for i in range(len(output_matrix)):
            for j in range(len(output_matrix[0])):
                #apply the mean kernel
                i_input = ratio*i+1
                j_input = ratio*j+1
                if mode=="mean":
                    output_matrix[i,j] = 1.0/ratio**2*np.sum(input_matrix[i_input-1:i_input+2,j_input-1:j_input+2])
                else:
                    output_matrix[i,j] = np.sum(input_matrix[i_input-1:i_input+2,j_input-1:j_input+2])

        start_1,start_2 = bead_start_per_chr_out[chr_1],bead_start_per_chr_out[chr_2]
        end_1,end_2 = bead_start_per_chr_out[chr_1] + num_bead_per_chr_out[chr_1],bead_start_per_chr_out[chr_2] + num_bead_per_chr_out[chr_2]
        #fill in the output matrix
        downsampled_matrix[start_1:end_1, start_2:end_2]=output_matrix
        #complete the matrix by symmetry
        downsampled_matrix[start_2:end_2, start_1:end_1]=np.transpose(output_matrix)
    
    return downsampled_matrix
    
