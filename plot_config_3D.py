import matplotlib.pyplot as plt
import numpy as np
import yaml
from pycirclize.utils import ColorCycler
from simulators import get_num_beads_and_start, chr_seq_parasite, chr_cen_parasite, sep_parasite, chr_seq_yeast, chr_cen_yeast, sep_yeast
from distinctipy import get_colors, color_swatch
from contact_matrix import get_config_3D_pop, get_config_3D

def get_nuclear_info(simu):
    """ return a dict with nuclear info"""
    info_file = open(simu + '/metadata/config.yaml', 'r')
    config = yaml.safe_load(info_file)
    return config['simulator']['args']



#get_nuclear_info('/home/etouron/SBI_bio/data/outputs/debug/6')


def plot_DNA(type, sep, config_3D, simu_path):
    """ plot the 3D DNA configuration """
    if type=='y':
        chr_seq = chr_seq_yeast
        chr_cen = chr_cen_yeast

    elif type=='p':
        chr_seq = chr_seq_parasite
        chr_cen = chr_cen_parasite

    else:
        print("unknown type")
        return
    chr_num_bead, chr_start_bead, nbead = get_num_beads_and_start(chr_seq, sep) #get the number of bead and the bead start id for each chr
    nuclear_info_dict = get_nuclear_info(simu_path)
    #nb_chr=int(list(chr_seq.keys())[-1][list(chr_seq.keys())[-1].find("chr")+3:])
    nb_chr = len(chr_seq.keys())
    
    #colors = get_colors(nb_chr, pastel_factor=0.1)

    # {chr_id : color}
    ColorCycler.set_cmap("jet")
    colors = ColorCycler.get_color_list(nb_chr)
    chr_name2color = {name: color for name, color in zip(chr_seq.keys(), colors)}

    #{organite : color}
    organite = {"nucleus env": ("lightgray", 0.2), "telomere env": ("silver", 0.1), 
                "centromere env": ("gray", 0.3), "nucleolus env": ("lightsteelblue", 0.5)}

    constraints_not_satis = ''
    ax = plt.figure().add_subplot(projection='3d')
    #plot the nuclear sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x_n = nuclear_info_dict['nuclear_rad']*np.cos(u)*np.sin(v)
    y_n = nuclear_info_dict['nuclear_rad']*np.sin(u)*np.sin(v)
    z_n = nuclear_info_dict['nuclear_rad']*np.cos(v)
    ax.plot_surface(x_n, y_n, z_n, color=organite['nucleus env'][0], alpha = organite['nucleus env'][1])

    #plot the telomere sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x_n = (nuclear_info_dict['nuclear_rad']-nuclear_info_dict['telo_thick'])*np.cos(u)*np.sin(v)
    y_n = (nuclear_info_dict['nuclear_rad']-nuclear_info_dict['telo_thick'])*np.sin(u)*np.sin(v)
    z_n = (nuclear_info_dict['nuclear_rad']-nuclear_info_dict['telo_thick'])*np.cos(v)
    ax.plot_surface(x_n, y_n, z_n, color=organite['telomere env'][0], alpha = organite['telomere env'][1])

    #plot the centromere sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = nuclear_info_dict['centro_rad']*np.cos(u)*np.sin(v) + nuclear_info_dict['nuclear_rad'] - nuclear_info_dict['centro_rad']
    y = nuclear_info_dict['centro_rad']*np.sin(u)*np.sin(v)
    z = nuclear_info_dict['centro_rad']*np.cos(v)
    ax.plot_surface(x, y, z, color=organite['centromere env'][0], alpha = organite['centromere env'][1])
   
    #plot the nucleolus
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = nuclear_info_dict['nucleolus_rad']*np.cos(u)*np.sin(v) + nuclear_info_dict['nucleolus_center_x'] 
    y = nuclear_info_dict['nucleolus_rad']*np.sin(u)*np.sin(v)
    z = nuclear_info_dict['nucleolus_rad']*np.cos(v)

    
    mask_inter = np.square(x) + np.square(y) + np.square(z) <= nuclear_info_dict['nuclear_rad']**2

    # x = np.reshape(x[mask_inter], (-1,2))
    # y = np.reshape(y[mask_inter], (-1,2))
    # z = np.reshape(z[mask_inter], (-1,2))
    
    ax.scatter(x[mask_inter], y[mask_inter], z[mask_inter], color=organite['nucleolus env'][0], alpha = organite['nucleolus env'][1], marker='s', s=10)
    #ax.plot_surface(np.asmatrix(x[mask_inter]), np.asmatrix(y[mask_inter]), np.asmatrix(z[mask_inter]), color="lightsteelblue", alpha = 0.5)
    #ax.plot_surface(x,y,z, color="lightsteelblue")
    for chr_num in chr_seq.keys():
        
        xs, ys, zs=[], [], [] 
        for bead_id in range (chr_start_bead[chr_num], chr_start_bead[chr_num] + chr_num_bead[chr_num]):
            #x,y,z = config_3D[bead_id][1:]
            x,y,z = config_3D[bead_id]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            #ax.scatter(xs=x, ys=y, zs=z, color= chr_name2color[chr_num])
        plt.plot(xs=xs, ys=ys, zs=zs, color= chr_name2color[chr_num], marker='o', markersize = 0.1, linestyle='-', linewidth=2)

        centro_telo = {"centromere": ("o", 10), "telomere":('s', 7)}

        #telomeres
        #x_t,y_t,z_t = config_3D[chr_start_bead[chr_num]][1:]
        x_t,y_t,z_t = config_3D[chr_start_bead[chr_num]]
        if (nuclear_info_dict['nuclear_rad']-nuclear_info_dict['telo_thick']) **2 <= x_t**2 + y_t**2+z_t**2 <= nuclear_info_dict['nuclear_rad']**2:
            edgecolor = "green"
        else:
            edgecolor = "red"
            print(f"constraint not satisfied on telomere for {chr_num}")
            constraints_not_satis += f"_t_{chr_num[chr_num.find('chr')+3:]}"

        plt.plot(xs=x_t, ys=y_t, zs=z_t, mfc= chr_name2color[chr_num], marker=centro_telo['telomere'][0], markersize=centro_telo['telomere'][1], color=edgecolor)
        #x_t,y_t,z_t = config_3D[chr_start_bead[chr_num]+chr_num_bead[chr_num]-1][1:]
        x_t,y_t,z_t = config_3D[chr_start_bead[chr_num]+chr_num_bead[chr_num]-1]
        if (nuclear_info_dict['nuclear_rad']-nuclear_info_dict['telo_thick']) **2 <= x_t**2 + y_t**2+z_t**2 <= nuclear_info_dict['nuclear_rad']**2:
            edgecolor = "green"
        else:
            edgecolor = "red"
            print(f"constraint not satisfied on telomere for {chr_num}")
            constraints_not_satis += f"_t_{chr_num[chr_num.find('chr')+3:]}"


        plt.plot(xs=x_t, ys=y_t, zs=z_t, mfc= chr_name2color[chr_num], marker=centro_telo['telomere'][0], markersize=centro_telo['telomere'][1], color=edgecolor)

        #centromeres
        bead_centro = chr_start_bead[chr_num] + chr_cen[chr_num]//sep
        #x_c,y_c,z_c = config_3D[bead_centro][1:]
        x_c,y_c,z_c = config_3D[bead_centro]
        if (x_c-(nuclear_info_dict['nuclear_rad'] - nuclear_info_dict['centro_rad']))**2 + y_c**2+z_c**2 <= nuclear_info_dict['centro_rad']**2:
            edgecolor = "green"
        else:
            edgecolor="red"
            print(f"constraint not satisfied on centromere for {chr_num}")
            constraints_not_satis += f"_c_{chr_num[chr_num.find('chr')+3:]}"

        plt.plot(xs=x_c, ys=y_c, zs=z_c, mfc= chr_name2color[chr_num], marker=centro_telo['centromere'][0], markersize=centro_telo['centromere'][1], color=edgecolor)

    
    #legend
    markers = []
    for chr_num in chr_seq.keys():
        width = chr_num_bead[chr_num]/100
        color=chr_name2color[chr_num]
        markers.append(plt.Line2D([0],[0],color=color, linestyle='-', linewidth=width))
    #centromere
    markers.append(plt.Line2D([0],[0],color="black", marker=centro_telo['centromere'][0], markersize=centro_telo['centromere'][1], linestyle=''))
    #telomere
    markers.append(plt.Line2D([0],[0],color="black", marker=centro_telo['telomere'][0], markersize=centro_telo['telomere'][1], linestyle=''))
    #centromere sphere
    markers.append(plt.Line2D([0],[0],color=organite['centromere env'][0], alpha=organite['centromere env'][1], marker='s', markersize = 5,linestyle=''))
    #nucleus
    markers.append(plt.Line2D([0],[0],color=organite['nucleus env'][0], alpha=organite['nucleus env'][1],  marker='s', markersize=5, linestyle=''))
    #nucleolus
    markers.append(plt.Line2D([0],[0],color=organite['nucleolus env'][0], alpha=organite['nucleolus env'][1], marker='s', linestyle=''))
    labels = list(chr_name2color.keys())+["centromere", "telomere", "centromere sphere", "nuclear sphere", "nucleolus membrane"]
    ax.legend(markers, labels, bbox_to_anchor=(1.1, 1.1))

    
    plt.title(f"3D DNA configuration for the {nb_chr} chromosomes")
    ax.grid(False)
    #plt.axis('off')
    plt.savefig(f"/home/etouron/SBI_bio/Simulation/plot/nucleolus/resolution_3200/DNA_config_{NB_SIMU}"+constraints_not_satis)
    plt.show()

    

# NB_SIMU=1
# config_3Ds, simu_path = get_config_3D_pop(NB_SIMU, 'p')
# plot_DNA('p', chr_seq, chr_cen, sep, config_3Ds[0], simu_path )

# for i in range(1,2):
#     #simu_path = '/home/etouron/SBI_bio/data/outputs/debug/S_Cerevisiae/'+str(i)+'/'
#     simu_path = '/home/etouron/SBI_bio/data/outputs/debug/1/'
#     config_3D = get_config_3D(simu_path+'artifacts/pickle/DNA_config')
#     plot_DNA('y', chr_seq_yeast, chr_cen_yeast, sep_yeast, config_3D, simu_path)

# NB_SIMU=100
# simu_path = f'/home/etouron/SBI_bio/data/outputs/debug/S_Cerevisiae/{NB_SIMU}'
# pickle_file = simu_path+'/artifacts/pickle/DNA_config'
# config_3D = get_config_3D(pickle_file, 'y')
# plot_DNA('y', config_3D, simu_path)

# for NB_SIMU in range(1, 51):
#     simu_path = f'/home/etouron/SBI_bio/Simulation/data/S_Cerevisiae_cluster/{NB_SIMU}'
#     pickle_file = simu_path+'/artifacts/pickle/DNA_config'
#     config_3D = get_config_3D(pickle_file, 'y')
#     plot_DNA('y', sep_yeast, config_3D, simu_path)

# NB_SIMU=1
# sep_yeast=100000
for i in [50]:
    config_3D = get_config_3D(f'/home/etouron/SBI_bio/Simulation/data/3200_S_Cerevisiae_tjong_donnees_duan/{i}/artifacts/pickle/DNA_config', 'y')
    NB_SIMU=i
    simu_path = f'/home/etouron/SBI_bio/Simulation/data/3200_S_Cerevisiae_tjong_donnees_duan/{i}/'
    plot_DNA('y', sep_yeast, config_3D, simu_path )


# sep_yeast = 3200
# config_3D = get_config_3D("/home/etouron/SBI_bio/Simulation_cluster/data/1/artifacts/pickle/DNA_config", "y")
# simu_path = f'/home/etouron/SBI_bio/Simulation_cluster/data/1/'
# plot_DNA('y', sep_yeast, config_3D, simu_path )

