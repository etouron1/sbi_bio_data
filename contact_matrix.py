import pickle
import numpy as np
from sklearn.metrics import euclidean_distances
from scipy.sparse import triu
from simulators import get_num_beads_and_start, chr_seq_parasite, sep_parasite, chr_seq_yeast, sep_yeast
from itertools import combinations

###----------------------------- compute contact matrix from DNA population -----------------------------###

def get_config_3D(pickle_file, type):
    """ print the set of bead's 3D coord for a DNA configuration"""
    
    with open(pickle_file, 'rb') as f:
        config_3D = pickle.load(f)
    return config_3D

#print_config_3D('/home/etouron/SBI_bio/data/outputs/debug/2/artifacts/pickle/DNA_config')

def get_config_3D_pop(pop_size, type):
    """ return a list of the set of bead's 3D coord for each DNA configuration in the pop_size population"""
    if type=='p':
        pickle_file = '/home/etouron/SBI_bio/data/outputs/debug/P_Falciparum/'
    elif type=='y':
        pickle_file = '/home/etouron/SBI_bio/data/outputs/debug/S_Cerevisiae/'
    else:
        print("unknown type")
        return

    config_3D_pop = [] #list of 3D config
    for i in range(pop_size):
        current_file = pickle_file + str(i+1) + '/artifacts/pickle/DNA_config'
        with open(current_file, 'rb') as f:
            config_3D_pop.append(pickle.load(f)) #load the set of coor
    return config_3D_pop, pickle_file+str(pop_size)

def get_contact_matrix_inter_chr(type, sep, config_3D_pop, num_chr_1, num_chr_2):
    """ create the contact matrix between two chromosomes from a population of 3D DNA structures"""

    thresh = 45 #nm each bead at distance <= thresh nm are in contact
    if type=='y':
        chr_seq = chr_seq_yeast

    elif type=='p':
        chr_seq = chr_seq_parasite

    else:
        print("unknown type")
        return
    
    num_chr_num_bead, chr_bead_start, nbead = get_num_beads_and_start(chr_seq, sep) #get the number of bead and the bead start id for each chr
    
    start_bead_chr_1, end_bead_chr_1 = chr_bead_start[num_chr_1], chr_bead_start[num_chr_1] + num_chr_num_bead[num_chr_1] - 1 #get the start bead id and the end bead id for the chr 1
    start_bead_chr_2, end_bead_chr_2 = chr_bead_start[num_chr_2], chr_bead_start[num_chr_2] + num_chr_num_bead[num_chr_2] - 1 #get the start bead id and the end bead id for the chr 2

    contact_matrix = np.zeros((num_chr_num_bead[num_chr_1],num_chr_num_bead[num_chr_2]))
    for config_3D in config_3D_pop:
        
        coor_list_chr_1 = config_3D[start_bead_chr_1: end_bead_chr_1+1] # get the 3D coor for the chr
        #coor_list_chr_1 = [i[1:] for i in coor_list_chr_1] # delete the num_chr from the coor list

        coor_list_chr_2 = config_3D[start_bead_chr_2: end_bead_chr_2+1] # get the 3D coor for the chr
        #coor_list_chr_2 = [i[1:] for i in coor_list_chr_2] # delete the num_chr from the coor list

        distance_matrix = euclidean_distances(coor_list_chr_1, coor_list_chr_2) # compute the distances between each bead

        contact_matrix += (distance_matrix<=thresh).astype(int) #add the contact
    
    contact_matrix=contact_matrix
    
    return contact_matrix

def get_contact_matrix_all_chr(type, sep, config_3D_pop):
    """ return the contact matrix for all the chr from a population of DNA configurations"""
    if type=='y':
        chr_seq = chr_seq_yeast

    elif type=='p':
        chr_seq = chr_seq_parasite

    else:
        print("unknown type")
        return
    num_chr_num_bead, chr_bead_start, nbead = get_num_beads_and_start(chr_seq, sep) #get the number of bead and the bead start id for each chr
    contact_matrix = np.zeros((nbead,nbead))

    for num_chr in chr_seq.keys():
        #num_chr = 'chr0' + str(chr_id)
        start_bead_chr, end_bead_chr = chr_bead_start[num_chr], chr_bead_start[num_chr] + num_chr_num_bead[num_chr] - 1 #get the start bead id and the end bead id for the chr
        contact_matrix[start_bead_chr:end_bead_chr+1, start_bead_chr:end_bead_chr+1] = np.triu(get_contact_matrix_inter_chr(type, sep, config_3D_pop, num_chr, num_chr))

    for (num_chr_1,num_chr_2) in combinations(chr_seq.keys(), r=2):
        #num_chr_1, num_chr_2 = 'chr0' + str(chr_id_1), 'chr0' + str(chr_id_2)
        start_bead_chr_1, end_bead_chr_1 = chr_bead_start[num_chr_1], chr_bead_start[num_chr_1] + num_chr_num_bead[num_chr_1] - 1 #get the start bead id and the end bead id for the chr
        start_bead_chr_2, end_bead_chr_2 = chr_bead_start[num_chr_2], chr_bead_start[num_chr_2] + num_chr_num_bead[num_chr_2] - 1 #get the start bead id and the end bead id for the chr
        contact_matrix[start_bead_chr_1:end_bead_chr_1+1, start_bead_chr_2:end_bead_chr_2+1] = get_contact_matrix_inter_chr(type, sep, config_3D_pop, num_chr_1, num_chr_2)
    return contact_matrix + np.transpose(np.triu(contact_matrix, k=1))

###------------------------------------------------------------------------------------------------------###

###--------------------------------- read contact matrix from experiment ---------------------------------###
def get_start_bead_end_bead_per_chr(file, nb_chr):
    """return two dict from a bed_file : {chr_num:start bead}, {chr_num:end_bead}"""

    fichier = open(file, "r")
    chr_bead_start, chr_bead_end = {}, {}

    line = fichier.readline()
    chr_num = line.split()[0]
    chr_bead_start[chr_num] = int(line.split()[3])
    for _ in range(1, nb_chr+1):
        while line.split()[0] == chr_num:
            line_prec=line
            line = fichier.readline()
            if line =='':
                chr_bead_end[chr_num] = int(line_prec.split()[3])
                return chr_bead_start, chr_bead_end

        chr_bead_end[chr_num] = int(line_prec.split()[3])
        chr_bead_start[line.split()[0]] = int(line.split()[3])
        chr_num = line.split()[0]

def construct_ref_matrix_from_file(chr_seq, sep, file_intra, file_inter):

    num_chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq, sep)
    contact_matrix = np.zeros((nb_bead, nb_bead))

    for f in file_intra:
        fichier = open(f, "r")
        i=0
        for line in fichier.readlines():
            if i!=0:
                if int(line.split()[0]) <=9:
                    chr = "chr0" + line.split()[0]
                else:
                    chr = "chr" + line.split()[0]
                
                
                contact_matrix[chr_bead_start[chr] + int(line.split()[1])//sep, chr_bead_start[chr] + int(line.split()[3])//sep] += float(line.split()[4])
            i+=1
        fichier.close()
    for f in file_inter:
        fichier = open(f, "r")
        
        i=0
        for line in fichier.readlines():
            if i!=0:
                if int(line.split()[0]) <=9:
                    chr_row = "chr0" + line.split()[0]
                else:
                    chr_row = "chr" + line.split()[0]
                if int(line.split()[2]) <=9:
                    chr_col = "chr0" + line.split()[2]
                else:
                    chr_col = "chr" + line.split()[2]
                contact_matrix[chr_bead_start[chr_row] + int(line.split()[1])//sep, chr_bead_start[chr_col] + int(line.split()[3])//sep] += float(line.split()[4])
            i+=1
        fichier.close()
    return contact_matrix + np.transpose(np.triu(contact_matrix, k=1))


def read_contact_matrix_all_chr_from_file(chr_seq, sep, file):
    """ return the contact matrix for all chromosomes from a matrix file"""
    num_chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq, sep)
    fichier = open(file, "r")
    contact_matrix = np.zeros((nb_bead, nb_bead))
    for line in fichier.readlines():
        
        contact_matrix[int(line.split()[0]), int(line.split()[1])] = float(line.split()[2])
    fichier.close()
    return contact_matrix + np.transpose(np.triu(contact_matrix, k=1))


def read_contact_matrix_inter_chr_from_file(chr_seq, sep, file, chr_row, chr_col):
    """return the contact matrix for 2 chromosomes from a matrix file"""
    chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq, sep)
    # if chr_num_1 > chr_num_2:
    #     chr_temp = chr_num_1
    #     chr_num_1 = chr_num_2
    #     chr_num_2 = chr_temp
    contact_matrix_all_chr = read_contact_matrix_all_chr_from_file(chr_seq, sep, file) 
    
    #chr_bead_start, chr_bead_end = get_start_bead_end_bead_per_chr("mapping_chr_pb_10000.bed", len(chr_bead_start.keys()))
    # start_1, end_1 = chr_bead_start[chr_num_1],chr_bead_end[chr_num_1]
    # start_2, end_2 = chr_bead_start[chr_num_2],chr_bead_end[chr_num_2]

    start_1, start_2 = chr_bead_start[chr_row],chr_bead_start[chr_col]
    end_1, end_2 = chr_bead_start[chr_row] + chr_num_bead[chr_row]-1, chr_bead_start[chr_col] + chr_num_bead[chr_col]-1
    contact_matrix_inter_chr =  contact_matrix_all_chr[start_1:end_1+1, start_2:end_2+1]
    if chr_row==chr_col:
        print("matrix intra-chr")
        # i_j_upper = np.triu_indices(end_1-start_1+1, k=1)
        # i_j_lower = np.tril_indices(end_1-start_1+1, k=-1)
        #contact_matrix_inter_chr[i_j_lower] = contact_matrix_inter_chr[i_j_upper]
 
    return contact_matrix_inter_chr

def read_contact_matrix_norm(chr_seq, sep, np_file, chr_row, chr_col):
    #fichier = open(file, "r")
    chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq, sep)
    contact_matrix = np.load(np_file)
    start_1, start_2 = chr_bead_start[chr_row],chr_bead_start[chr_col]
    end_1, end_2 = chr_bead_start[chr_row] + chr_num_bead[chr_row]-1, chr_bead_start[chr_col] + chr_num_bead[chr_col]-1
    return contact_matrix[start_1:end_1+1, start_2:end_2+1]

#print(read_contact_matrix_inter_chr_from_file("contact_matrix.matrix", 1216,16,"chr01", "chr02"))
#read_contact_matrix_norm("duan.SC.10000.normed.npy")
#  
import os  
def read_contact_matrix_all_chr_from_pickle():
    #pickle_dir = '/home/etouron/SBI_bio/Simulation/data/S_Cerevisiae_cluster_nucleolus/'
    #pickle_dir = '/home/etouron/SBI_bio/code_origine/data'
    pickle_dir = '/home/etouron/SBI_bio/Simulation_cluster/data'
    list_dir = os.listdir(pickle_dir)
    for i in range(len(list_dir)):
        print(list_dir[i])
        current_pickle = pickle_dir+f"/{list_dir[i]}/artifacts/pickle/raw_contact_matrix"
        with open(current_pickle, 'rb') as f:
            if i==0:
                contact_matrix = (pickle.load(f)) #load the raw contact matrix
            else:
                contact_matrix += (pickle.load(f)) #load the raw contact matrix

    return contact_matrix, len(list_dir)

def read_contact_matrix_all_chr_from_simu(type, start_simu, nb_simu, resolution, nucleolus, author, plot):
    if type=='y':
        if nucleolus=='n':
            if resolution==10000:
                pickle_dir = '/home/etouron/SBI_bio/Simulation/data/10000_S_Cerevisiae_nelle_donnees_moyennes'
            elif resolution==3200:
                if author=="duan":
                    pickle_dir = '/home/etouron/SBI_bio/Simulation/data/3200_S_Cerevisiae_tjong_donnees_duan'
                elif author=="nelle":
                    pickle_dir = '/home/etouron/SBI_bio/Simulation/data/3200_S_Cerevisiae_nelle_donnees_moyennes'
                #pickle_dir = '/home/etouron/SBI_bio/Simulation/data/S_Cerevisiae_tjong_3200_donnees_precises'
                elif author=="tjong":
                    pickle_dir = '/home/etouron/SBI_bio/Simulation/data/3200_S_Cerevisiae_tjong_donnees_moyennes'
                elif author=="tjong_faux_centro":
                    pickle_dir = '/home/etouron/SBI_bio/Simulation/data/3200_S_Cerevisiae_tjong_donnees_duan_faux_centro'
                else:
                    print ("unknow author")
                    return
            elif resolution==100000:
                pickle_dir = '/home/etouron/SBI_bio/Simulation/data/S_Cerevisiae_cluster_nucleolus_100000'
            else:
                print("no data")
                return
        elif nucleolus=='sn':
            #print("unknown resolution")
            if resolution==3200:
                pickle_dir = '/home/etouron/SBI_bio/Simulation/data/S_Cerevisiae_cluster_sans_nucleolus_3200'
            else:
                print("no data")
                return
            #pickle_dir = '/home/etouron/SBI_bio/code_origine/data'
    else:
        print("unknown type")
        return
                        ######### plot #########
    if plot:
        for i in range(start_simu, nb_simu+1):
            
            current_pickle = pickle_dir+f"/{i}/artifacts/pickle/raw_contact_matrix"
            with open(current_pickle, 'rb') as f:
                if i==start_simu:
                    contact_matrix = (pickle.load(f)) #load the set of coor
                else:
                    contact_matrix += (pickle.load(f)) #load the set of coor
                        ######### plot #########
    else:

                        ######### correlation #########
        current_pickle = pickle_dir+f"/{nb_simu}/artifacts/pickle/raw_contact_matrix"
        with open(current_pickle, 'rb') as f:
            
            contact_matrix = (pickle.load(f)) #load the set of coor
                        ######### correlation #########
           
    return contact_matrix

# contact_matrix, nb_sim = read_contact_matrix_all_chr_from_pickle()
# print(contact_matrix)

# with open('/home/etouron/SBI_bio/data/outputs/debug/1/artifacts/pickle/raw_contact_matrix', 'rb') as f:
#             contact_matrix = (pickle.load(f))
# print(contact_matrix)

def read_contact_matrix_inter_chr_from_pickle(type, sep, nb_simu, chr_row, chr_col):
    contact_matrix_all_chr = read_contact_matrix_all_chr_from_simu(type, nb_simu)
    if type=='y':
        chr_seq = chr_seq_yeast

    else:
        print("unknown type")
        return
    num_chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq, sep)
    start_1, start_2 = chr_bead_start[chr_row],chr_bead_start[chr_col]
    end_1, end_2 = chr_bead_start[chr_row] + num_chr_num_bead[chr_row]-1, chr_bead_start[chr_col] + num_chr_num_bead[chr_col]-1
    contact_matrix_inter_chr =  contact_matrix_all_chr[start_1:end_1+1, start_2:end_2+1]
    if chr_row==chr_col:
        print("matrix intra-chr")
    return contact_matrix_inter_chr




###-------------------------------------------------------------------------------------------------------###

def get_chr_seq(file):
    """get the dict {chr_num:num of pb} from an info file"""

    fichier = open(file, "r")
    chr_seq = {}

    line = fichier.readline()
    while line != '':
        chr_num = line.split()[0]
        start_pb = int(line.split()[1])
        while line.split()[0] == chr_num:
            line_prec=line
            line = fichier.readline()
            if line =='':
                end_pb = int(line_prec.split()[2])
                chr_seq[chr_num] = end_pb-start_pb+1
                return chr_seq
        end_pb = int(line_prec.split()[2])
        chr_seq[chr_num] = end_pb-start_pb+1
        
