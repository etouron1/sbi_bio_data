import matplotlib.pyplot as plt
import pickle
import random

def plot_evolution_pixel(type, nb_simu):
    if type=='y':
        pickle_dir = '/home/etouron/SBI_bio/Simulation/data/S_Cerevisiae_cluster_nucleolus_3200'
    else:
        print("unknown type")
        return
    pixel_diag = []
    pixel_diag_sup_1 =[]
    pixel_diag_sup_2 =[]
    pixel_diag_inf_1 =[]
    pixel_diag_inf_2 =[]
    pixel_out_diag_1 =[]
    pixel_out_diag_2 =[]
    pixel_out_diag_3 =[]
    pixel_out_diag_4 =[]
    for i in range(1, nb_simu+1):
        current_pickle = pickle_dir+f"/{i}/artifacts/pickle/raw_contact_matrix"
        with open(current_pickle, 'rb') as f:
            if i==1:
                contact_matrix = (pickle.load(f)) #load the set of coor
                idx_diag=random.choice(range(0, len(contact_matrix)))


            else:
                contact_matrix += (pickle.load(f)) #load the set of coor
            pixel_diag.append(contact_matrix[idx_diag, idx_diag])
            pixel_diag_sup_1.append(contact_matrix[idx_diag, idx_diag+2])
            pixel_diag_sup_2.append(contact_matrix[idx_diag, idx_diag+5])
            pixel_diag_inf_1.append(contact_matrix[idx_diag-2, idx_diag])
            pixel_diag_inf_2.append(contact_matrix[idx_diag-5, idx_diag])
            pixel_out_diag_1.append(contact_matrix[idx_diag, idx_diag+100])
            pixel_out_diag_2.append(contact_matrix[idx_diag, idx_diag+200])
            pixel_out_diag_3.append(contact_matrix[idx_diag, idx_diag+500])
            pixel_out_diag_4.append(contact_matrix[idx_diag, idx_diag+700])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
    ax1.plot(range(1, nb_simu+1), pixel_diag, label=f"({idx_diag}, {idx_diag})")
    ax1.set_xlabel("kilo nb simulations")
    ax1.set_ylabel("nb of contacts")
    ax1.legend(bbox_to_anchor=(1, 1.1))
    ax1.set_title("diagonal pixel")
   
    ax2.plot(range(1, nb_simu+1), pixel_out_diag_1, label=f"({idx_diag}, {idx_diag+100})")
    ax2.plot(range(1, nb_simu+1), pixel_out_diag_2, label=f"({idx_diag}, {idx_diag+200})")
    ax2.plot(range(1, nb_simu+1), pixel_out_diag_3, label=f"({idx_diag}, {idx_diag+500})")
    ax2.plot(range(1, nb_simu+1), pixel_out_diag_4, label=f"({idx_diag}, {idx_diag+700})")
    ax2.set_xlabel("kilo nb simulations")
    ax2.set_ylabel("nb of contacts")
    ax2.legend(bbox_to_anchor=(1, 1.1))
    ax2.set_title("off diagonal pixel")
    
    ax3.plot(range(1, nb_simu+1), pixel_diag_sup_1, label=f"({idx_diag}, {idx_diag+2})")
    ax3.plot(range(1, nb_simu+1), pixel_diag_sup_2, label=f"({idx_diag}, {idx_diag+5})")
    ax3.set_xlabel("kilo nb simulations")
    ax3.set_ylabel("nb of contacts")
    ax3.legend(bbox_to_anchor=(1, 1.1))
    ax3.set_title("upper diagonal pixel")

    fig.subplots_adjust(hspace=1)
    fig.subplots_adjust(wspace=1)
    plt.show()
    
plot_evolution_pixel('y', 200)