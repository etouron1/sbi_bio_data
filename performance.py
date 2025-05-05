import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from itertools import combinations

import pandas as pd
import seaborn as sns
import torch

from scipy import ndimage

from sbi.utils.metrics import unbiased_mmd_squared


chr_seq = {"chr01": 230209, "chr02": 813179, "chr03": 316619}
chr_cen = {'chr01': 151584, 'chr02': 238325, 'chr03': 114499}

# chr_seq = {"chr01": 230209, "chr02": 813179, "chr03": 316619, "chr04": 1531918,
#            "chr05": 576869, "chr06": 270148, "chr07": 1090947, "chr08": 562644,
#             "chr09": 439885, "chr10": 745746, "chr11": 666455, "chr12": 1078176,
#             "chr13": 924430, "chr14": 784334, "chr15": 1091290, "chr16": 948063}

# chr_cen = {'chr01': 151584, 'chr02': 238325, 'chr03': 114499, 'chr04': 449819,
#                  'chr05': 152103, 'chr06': 148622, 'chr07': 497042, 'chr08': 105698,
#                  'chr09': 355742, 'chr10': 436418, 'chr11': 439889, 'chr12': 150946,
#                  'chr13': 268149, 'chr14': 628877, 'chr15': 326703, 'chr16': 556070}

def get_theta_corr(theta, corr):
    """ return a dict {theta : corr}"""
    theta_corr = {}
    for i, t in enumerate(theta):
        t_keys = tuple(t.values())
        theta_corr[t_keys] = corr[i]
    return theta_corr

def get_theta_pertinent(theta_corr, eps_theta):
    """ return theta st ||theta-theta_0|| <= eps_theta"""
    theta_pertinence = []
    for j, theta in enumerate(theta_corr.keys()):
        d = 0
        for i in range(len(theta)):
            d+= (theta[i] - theta_ref[i])**2
        d = np.sqrt(d/len(theta))
        distance[j] = d
        if d <= eps_theta:
            theta_pertinence.append(1)
        else:
            theta_pertinence.append(0)
    return theta_pertinence

def get_theta_selectionnes(theta_corr, eps_corr):
    """ return theta st corr(theta, theta_0) >= eps_corr"""
    theta_selectionnes = []
    for theta, corr in theta_corr.items():
        if corr >= eps_corr:
            theta_selectionnes.append(1)

        else:
            theta_selectionnes.append(0)
    return theta_selectionnes

def get_distance_theta_theta_ref(theta_corr, theta_ref):
    """ return a list [ ||theta-theta_0|| for all theta ]"""
    distance = np.zeros(len(theta_corr))
    for j, theta in enumerate(theta_corr.keys()):
        d= 0
        for i in range(len(theta)):
            d+= (theta[i] - theta_ref[i])**2
        d = np.sqrt(d/len(theta))
        distance[j] = d
    return distance


def get_t_p_f_n_t_n_f_p(theta_corr, theta_ref, eps_theta):
    t_p_f_n = 0
    t_n_f_p = 0

    for theta in theta_corr.keys():
        d = 0
        for i in range(len(theta)):
            d+= (theta[i] - theta_ref[i])**2
        d = np.sqrt(d/len(theta))
        if d<= eps_theta:
            t_p_f_n += 1
        else:
            t_n_f_p += 1

    return t_p_f_n, t_n_f_p

def get_t_p_f_p(theta_corr, eps_corr):
    t_p_f_p = 0

    for corr in theta_corr.values():
        if corr >= eps_corr:
                t_p_f_p += 1
    return t_p_f_p


def get_t_p(theta_corr, theta_ref, eps_theta, eps_corr):
    t_p = 0

    for theta, corr in theta_corr.items():
        d = 0
        for i in range(len(theta)):
            d+= (theta[i] - theta_ref[i])**2
        d = np.sqrt(d/len(theta))

        if d<= eps_theta and corr >= eps_corr:
            t_p += 1
    return t_p

def get_f_p(theta_corr, theta_ref, eps_theta, eps_corr):
    f_p = 0
    for theta, corr in theta_corr.items():
        d = 0
        for i in range(len(theta)):
            d+= (theta[i] - theta_ref[i])**2
        d = np.sqrt(d/len(theta))
        if corr >= eps_corr and d > eps_theta:
                f_p += 1
    return f_p



def get_precision(theta_corr, theta_ref, eps_theta, eps_corr):
    t_p = get_t_p(theta_corr, theta_ref, eps_theta, eps_corr)
    t_p_f_p = get_t_p_f_p(theta_corr, eps_corr)
    return t_p/t_p_f_p

def get_recall(theta_corr, theta_ref, eps_theta, eps_corr):
      t_p = get_t_p(theta_corr, theta_ref, eps_theta, eps_corr)
      t_p_f_n = get_t_p_f_n_t_n_f_p(theta_corr, theta_ref, eps_theta)[0]
      return t_p/t_p_f_n

def get_fpr(theta_corr, theta_ref, eps_theta, eps_corr):
    f_p = get_f_p(theta_corr, theta_ref, eps_theta, eps_corr)
    t_n_f_p = get_t_p_f_n_t_n_f_p(theta_corr, theta_ref, eps_theta)[1]
    return f_p/t_n_f_p

def plot_precision_recall(eps_theta_list, type_corr, noisy):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, eps_theta in enumerate(eps_theta_list):
        eps_corr_min = min(theta_corr.values())
        eps_corr_max = max(theta_corr.values())
        precision = []
        recall = []
        for eps_corr in np.linspace(eps_corr_min,eps_corr_max,100):

            precision.append(get_precision(theta_corr, theta_ref, eps_theta, eps_corr))
            recall.append(get_recall(theta_corr, theta_ref, eps_theta, eps_corr))

        # sorted_indices = np.argsort(np.array(recall))
        # recall = np.array(recall)[sorted_indices]
        # precision = np.array(precision)[sorted_indices]
        #color = "blue", marker="+")

        axes[i//2, i%2].step(recall, precision, where='post', color="blue", linestyle="-", linewidth=2, zorder=1)
        surf=axes[i//2, i%2].scatter(recall, precision, c=np.linspace(eps_corr_min,eps_corr_max,100), cmap="coolwarm", s=10, zorder=2)#, vmin = -0.5, vmax = 1)

        axes[i//2, i%2].set_xlabel("recall")
        axes[i//2, i%2].set_ylabel("precision")
        axes[i//2, i%2].set_title(rf"$\epsilon_{{\theta}}$ = {eps_theta:.0f}")
    plt.suptitle(f"Precision-Recall curve for {type_corr}-based corr \n"+r"inference $(c_1, c_2, c_3)$," +f"{noisy} data")
    plt.tight_layout()
    # plt.show()
    fig.subplots_adjust(top=0.83)
    cbar_ax = fig.add_axes([0.15, 0.9, 0.7, 0.02])  # [left, bottom, width, height]
    fig.colorbar(surf, cax=cbar_ax, orientation='horizontal')
    plt.show()
    #plt.savefig(f"{type_corr}_precision_recall_corr_{noisy}.svg")

def plot_ROC(eps_theta_list, type_corr, noisy):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, eps_theta in enumerate(eps_theta_list):
        eps_corr_min = min(theta_corr.values())
        eps_corr_max = max(theta_corr.values())
        tpr = []
        fpr = []
        for eps_corr in np.linspace(eps_corr_min,eps_corr_max,100):

            tpr.append(get_recall(theta_corr, theta_ref, eps_theta, eps_corr))
            fpr.append(get_fpr(theta_corr, theta_ref, eps_theta, eps_corr))

            #axes[i//2, i%2].scatter(fpr, tpr, color = "blue", marker="+")
        # sorted_indices = np.argsort(np.array(fpr))
        # fpr = np.array(fpr)[sorted_indices]
        # tpr = np.array(tpr)[sorted_indices]
        axes[i//2, i%2].step(fpr, tpr, where='post', color="blue", linestyle="-", linewidth=2, zorder=1)
        surf=axes[i//2, i%2].scatter(fpr, tpr, c=np.linspace(eps_corr_min,eps_corr_max,100), cmap="coolwarm", s=10, zorder=2)#, vmin = -0.5, vmax = 1)
        axes[i//2, i%2].set_xlabel("fpr")
        axes[i//2, i%2].set_ylabel("tpr")
        axes[i//2, i%2].set_title(rf"$\epsilon_{{\theta}}$ = {eps_theta:.0f}")
    plt.suptitle(f"ROC curve for {type_corr}-based corr\n"+r"inference $(c_1, c_2, c_3)$,"+f"{noisy} data")
    plt.tight_layout()
    # plt.show()
    fig.subplots_adjust(top=0.83)
    cbar_ax = fig.add_axes([0.15, 0.9, 0.7, 0.02])  # [left, bottom, width, height]
    fig.colorbar(surf, cax=cbar_ax, orientation='horizontal')
    plt.show()
    #plt.savefig(f"{type_corr}_ROC_corr_{noisy}.svg")

def plot_precision_recall_vs_threshold(eps_theta_list, type_corr, noisy):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, eps_theta in enumerate(eps_theta_list):
        eps_corr_min = min(theta_corr.values())
        eps_corr_max = max(theta_corr.values())
        precision = []
        recall = []
        for eps_corr in np.linspace(eps_corr_min,eps_corr_max,100):

            precision.append(get_precision(theta_corr, theta_ref, eps_theta, eps_corr))
            recall.append(get_recall(theta_corr, theta_ref, eps_theta, eps_corr))

            #axes[i//2, i%2].scatter(recall, precision, color = "blue", marker="+")
        # sorted_indices = np.argsort(np.array(recall))
        # recall = np.array(recall)[sorted_indices]
        # precision = np.array(precision)[sorted_indices]
        axes[i//2, i%2].step(np.linspace(eps_corr_min,eps_corr_max,100), precision, where='post', color="blue", linestyle="-", linewidth=2, zorder=1, label = 'precision')
        axes[i//2, i%2].step(np.linspace(eps_corr_min,eps_corr_max,100), recall, where='post', color="red", linestyle="-", linewidth=2, zorder=1, label='recall')

        axes[i//2, i%2].set_xlabel(r"$\epsilon_{corr}$")
        axes[i//2, i%2].set_ylabel("precision-recall")
        axes[i//2, i%2].set_title(rf"$\epsilon_{{\theta}}$ = {eps_theta:.0f}")
    plt.suptitle(f"Precision-recall VS $\epsilon_{{corr}}$ for {type_corr}-based corr\n"+r"inference $(c_1, c_2, c_3)$,"+f"{noisy} data")
    plt.tight_layout()
    plt.legend()
    # plt.show()
    fig.subplots_adjust(top=0.83)

    plt.savefig(f"{type_corr}_corr_precision-recall_vs_thresh_{noisy}.svg")

def mean_distance_theta(theta_corr, theta_ref, eps_corr_list):

    d_mean_list = []
    for eps_corr in eps_corr_list:

        d_mean = 0
        theta_sel = 0
        for theta, corr in theta_corr.items():
            if corr >= eps_corr:
                theta_sel += 1
                # d_mean += np.abs(theta-theta_ref)

                d = 0
                for i in range(len(theta)):
                    d+= (theta[i] - theta_ref[i])**2
                d_mean += np.sqrt(d/len(theta))


        d_mean_list.append(d_mean/theta_sel)
    return d_mean_list

def mean_distance_closest_theta(theta_dist, theta_ref,eps_dist_list):

    d_mean_list = []
    for eps_dist in eps_dist_list:

        d_mean = 0
        theta_sel = 0
        for theta, dist in theta_dist.items():
            if dist <= eps_dist:
                theta_sel += 1

                d = 0
                for i in range(len(theta)):
                    d+= (theta[i] - theta_ref[i])**2

                d_mean += np.sqrt(d/len(theta))



        d_mean_list.append(d_mean/theta_sel)
    return d_mean_list

def mean_distance_best_thetas(thetas_accepted, theta_ref):

    d_mean = 0

    for theta in thetas_accepted:

        d = 0
        for i in range(len(theta)):
            d+= (theta[i] - theta_ref[i])**2
        d_mean += np.sqrt(d/len(theta))



    return d_mean/len(thetas_accepted)


def mean_distance_theta_given_prop(theta_accepted, theta_ref):

    d_mean = 0
    for j in range(len(theta_accepted[0])):
        theta = theta_accepted[:,j]


        d_mean += torch.sqrt(torch.sum(theta - theta_ref)**2/len(theta))

    return d_mean/len(theta_accepted[0])

def wasserstein_distance(theta_ref, theta_accepted):
      theta_accepted = theta_accepted.float()
      mean = torch.mean(theta_accepted, dim = 0)
      var = torch.mean(theta_accepted**2, dim = 0) - mean**2

      return torch.sum(var) + torch.mean((mean-theta_ref)**2)

def f_mmd(a, b, diag=False):
        if diag:
            return torch.sum((a[None, ...] - b[:, None, :]) ** 2, dim=-1).reshape(-1)
        else:
            m, n = a.shape[0], b.shape[0] #50,50
            ix = torch.tril_indices(m, n, offset=-1) #indices non nuls mat triang lower strict (n(n-1)/2)

            #a : 50,3 a[None, ...] : 1,50,3
            #b : 50,3 b[:, None, :] : 50,1,3
            #(a-b)^2 : 50,50
            return torch.sum(
                (a[None, ...] - b[:, None, :]) ** 2, dim=-1, keepdim=False
            )[ix[0, :], ix[1, :]].reshape(-1)
###################################################################
###################################################################

#plot mean distance VS correlation for 1 bloc
if 0 :
    fig, axes = plt.subplots(1,3, figsize=(10, 8))
    #color = {"row": "red", "col": "blue", "vector": "green"}
    color = {"27_27": "red", "27_105": "blue", "105_27": "green"}
    theta_ref = {"27_27":(256000, 544000), "27_105": (256000, 928000), "105_27":(256000, 640000)} #c_i, c_j
    # theta_ref = {"27_27":256000, "27_105": 256000, "105_27":256000} #c_i
    #theta_ref = {"27_27":544000, "27_105": 928000, "105_27":640000} #c_j
    #for i, metric in enumerate(["row", "col", "vector"]):
    for i, metric in enumerate(["row", "col", "vector"]):
        for shape in ["27_27", "27_105", "105_27"]:
            d_mean_list = {}
            with open(f"matrix_{shape}/1000_theta/inference_c_i_c_j/theta_P_corr_{metric}_mean", 'rb') as f:
                        theta_corr = pickle.load(f)
                        print(shape, len(theta_corr))

            eps_corr_min = min(theta_corr.values())
            eps_corr_max = max(theta_corr.values())

            #print(shape, metric, eps_corr_min, eps_corr_max, color[shape])
            eps_corr_list = np.linspace(eps_corr_min+(eps_corr_max- eps_corr_min)*0.75,eps_corr_max,100)
            theta_corr_sorted = dict(sorted(theta_corr.items(), key=lambda item: item[1]))
            start = int(len(theta_corr_sorted)*0.75)
            eps_corr_start = list(theta_corr_sorted.values())[start]

            # eps_corr_list = np.linspace(eps_corr_min+(eps_corr_max- eps_corr_min)*0.75,eps_corr_max,100)
            eps_corr_list = np.linspace(eps_corr_start,eps_corr_max,100)

            d_mean_list = mean_distance_theta(theta_corr, theta_ref[shape], eps_corr_list)
            #print(d_mean_list)

            axes[i].scatter(eps_corr_list, d_mean_list, marker="+", color = color[shape], label=f"shape {shape}")
            axes[i].legend()
            axes[i].set_title(f"P. {metric}-based")
            axes[i].set_xlabel(r"$\epsilon_{corr}$")
            axes[i].set_ylabel(r"$mean(||\theta_i - \theta_{ref}||, \theta_i$ s.t. $corr(C_{\theta_i}, C_{\theta_{ref}}) \geq \epsilon_{corr}$ (bp)")
            axes[i].axhline(y=32000, linestyle="--", linewidth = 0.7, color="black")
            if metric != "row":
                axes[i].set_ylim((0,100000))
    plt.suptitle(r"Inference of 1 centromere $c_i$")
    plt.tight_layout()

    plt.show()
#mean distance multiple setup clear/noisy sigma fixe/variable
if 0:
    fig, ax = plt.subplots(2,2, figsize=(10, 8))
    sigmas = ["fixe", "variable"]
    noisys = ["clear", "noisy"]
    color = {"row": "red", "col": "blue", "row_col": "green", "vector": "goldenrod"}
    theta_ref = (151584, 238325, 114499) #c_i, c_j
    for k, noisy in enumerate(noisys):
        for j, sigma in enumerate(sigmas):

            for i, metric in enumerate(["row", "col", "row_col", "vector"]):

                    d_mean_list = {}
                    with open(f"simulation_little_genome/true/res_32000/{noisy}/sigma_{sigma}/0_theta_S_corr_inter_{metric}", 'rb') as f:
                                theta_corr = pickle.load(f)
                                print(len(theta_corr))

                    eps_corr_min = min(theta_corr.values())
                    eps_corr_max = max(theta_corr.values())


                    theta_corr_sorted = dict(sorted(theta_corr.items(), key=lambda item: item[1]))
                    start = int(len(theta_corr_sorted)*0.75) #3eme quartile (75%)
                    eps_corr_start = list(theta_corr_sorted.values())[start]

                    print(len(list(theta_corr_sorted.values())[start:]))

                    #eps_corr_list = np.linspace(eps_corr_min+(eps_corr_max- eps_corr_min)*0.75,eps_corr_max,100)
                    eps_corr_list = np.linspace(eps_corr_start,eps_corr_max,100)

                    d_mean_list = mean_distance_theta(theta_corr, theta_ref, eps_corr_list)
                    #print(d_mean_list)
                    ax[j,k].axhline(32000, linestyle = "--", color = "black", linewidth = 0.7)

                    ax[j,k].scatter(eps_corr_list, d_mean_list, marker="+", color = color[metric], label=f"S. {metric}-based")
                    ax[j,k].legend()
                    ax[j,k].set_title(fr"{noisy} - $\sigma^2$ {sigma}")
                    ax[j,k].set_xlabel(r"$\epsilon_{corr}$")
                    ax[j,k].set_ylabel(r"$mean(||\theta_i - \theta_{ref}||, \theta_i$ s.t. $corr(C_{\theta_i}, C_{\theta_{ref}}) \geq \epsilon_{corr}$ (bp)")
    plt.suptitle(fr"Inference of 3 centromeres $c_1, c_2, c_3$ using Spearman correlation"+"\n resolution 32000")
    plt.tight_layout()
    plt.show()

#plot mean distance VS correlation for a small genome
if 0:
    fig, ax = plt.subplots(figsize=(10, 8))
    #color = {"inter_row_col": "red", "inter_vector": "blue", "upper_vector": "green"}
    color = {"inter_row_col": "red", "inter_vector": "blue"}

    theta_ref = (151584, 238325, 114499) #c_i, c_j
    #theta_ref = {"27_27":256000, "27_105": 256000, "105_27":256000} #c_i
    #theta_ref = {"27_27":544000, "27_105": 928000, "105_27":640000} #c_j
    #for i, metric in enumerate(["row", "col", "vector"]):
    for i, metric in enumerate(["inter_row_col", "inter_vector"]):#, "upper_vector"]):


        d_mean_list = {}
        with open(f"simulation_little_genome/true/res_32000/theta_P_corr_{metric}", 'rb') as f:
                    theta_corr = pickle.load(f)
                    #print(shape, len(theta_corr))
                    #print(theta_corr)

        eps_corr_min = min(theta_corr.values())
        eps_corr_max = max(theta_corr.values())

        print(eps_corr_min, eps_corr_max)
        theta_corr_sorted = dict(sorted(theta_corr.items(), key=lambda item: item[1]))
        start = int(len(theta_corr_sorted)*0.75)
        eps_corr_start = list(theta_corr_sorted.values())[start]
        # print(list(theta_corr_sorted.values()))
        print(len(list(theta_corr_sorted.values())[start:]))
        #print(theta_corr_sorted)

        #print(shape, metric, eps_corr_min, eps_corr_max, color[shape])
        #eps_corr_list = np.linspace(eps_corr_min+(eps_corr_max- eps_corr_min)*0.75,eps_corr_max,100)
        eps_corr_list = np.linspace(eps_corr_start,eps_corr_max,100)
        #print(eps_corr_list)
        d_mean_list = mean_distance_theta(theta_corr, theta_ref, eps_corr_list)
        #print(d_mean_list)
        #print(eps_corr_list)
        ax.scatter(eps_corr_list, d_mean_list, marker="+", color = color[metric], label=f"P. {metric}-based")
        ax.legend()
        ax.axhline(y=10000, color="black", linestyle='--', linewidth = 1)
        #ax.set_title(f"P. {metric}-based")
        ax.set_xlabel(r"$\epsilon_{corr}$")
        ax.set_ylabel(r"$mean(||\theta_i - \theta_{ref}||, \theta_i$ s.t. $corr(C_{\theta_i}, C_{\theta_{ref}}) \geq \epsilon_{corr}$ (bp)")
    plt.suptitle(r"Inference of 3 centromeres $c_1, c_2, c_3$"+"\n experimental data")
    plt.tight_layout()
    # plt.ylim((10000, 16000))
    # plt.xlim((0.05, 0.6))
    plt.show()

#plot precision/recall + ROC for small genome
if 0:
    for simu_type in ["noisy"]:
        for metric in ["P"]:
            for corr_name in ["vector"]:
                # with open(f"simulation_little_genome/synthetic/{simu_type}/theta", 'rb') as f:
                #         theta = pickle.load(f)
                with open(f'simulation_little_genome/true/res_10000/{simu_type}/theta_{metric}_corr_inter_{corr_name}', 'rb') as f:
                        theta_corr = pickle.load( f)

                #theta_corr=get_theta_corr(theta, corr)
                #theta_corr = {(c_1, c_2, c_3) : corr}
                #theta_ref = {'chr01': 151584, 'chr02': 238325, 'chr03': 114499}
                theta_ref = (151584, 238325, 114499)

                distance = sorted(get_distance_theta_theta_ref(theta_corr, theta_ref), reverse=True)

                start_list = [int(len(theta_corr)*prop) for prop in [0.75,0.8,0.9,0.95]] #[0.5,0.6,0.7,0.8]]#[0.95,0.99,0.995,0.997]]

                #print(min(distance), np.mean(distance), max(distance), distance[start])
                eps_theta_list = [distance[start] for start in start_list]

                type_corr = f"{metric}. {corr_name}"
                plot_precision_recall(eps_theta_list, type_corr, f"{simu_type}")
                plot_ROC(eps_theta_list, type_corr, f"{simu_type}")
                #plot_precision_recall_vs_threshold(eps_theta_list, type_corr, f"{simu_type}")


# theta_pertinents = get_theta_pertinent(theta_corr, eps_theta_list[3])
# theta_selectionnes = np.zeros_like(theta_pertinents, dtype=float)
# for eps_corr in np.linspace(min(theta_corr.values()),max(theta_corr.values()),100):
#     theta_selectionnes += np.array(get_theta_selectionnes(theta_corr,eps_corr))

# theta_selectionnes/=100

# from sklearn.metrics import PrecisionRecallDisplay

# PrecisionRecallDisplay.from_predictions(theta_pertinents, theta_selectionnes)
# plt.title("S.vector-based")
# plt.show()

#correlation VS sigma^2 for 1 bloc sigma^2 only
if 0:
    fig, axes = plt.subplots(1,2, figsize=(10, 8))
    color = {"27_27": "red", "27_105": "blue", "105_27": "green"}

    for i, metric in enumerate(["row_col", "vector"]):
            for shape in ["27_27", "27_105", "105_27"]:
                with open(f"matrix_{shape}/1000_theta/inference_c_i_c_j/sig_2_P_corr_{metric}", 'rb') as f:
                            sig_2_corr = pickle.load(f)
                axes[i].scatter(list(sig_2_corr.keys()), list(sig_2_corr.values()), marker="+", color = color[shape], label=f"shape {shape}")
                axes[i].axvline(2, linestyle = "--", color = "black", linewidth = 0.7)
                axes[i].legend()
                axes[i].set_title(f"P. {metric}-based")
                axes[i].set_xlabel(r"$\sigma^2$")
                axes[i].set_ylabel(r"$corr(C_{\sigma_i^2}, C_{\sigma^2_{ref}})$")
    plt.suptitle(r"Effect of $\sigma^2$ on the correlation when infering $c_i, c_j$"+"\n"+r"$\sigma^2_{ref} = 2$")
    plt.tight_layout()
    plt.show()

#corelation VS sigma^2 for small genome sigma only
if 0:

    fig, ax = plt.subplots(figsize=(10, 8))
    color = {"row_col": "red", "vector": "blue"}

    for i, metric in enumerate(["row_col", "vector"]):

                with open(f"simulation_little_genome/noisy/sig_2_P_corr_inter_{metric}", 'rb') as f:
                            sig_2_corr = pickle.load(f)
                ax.scatter(list(sig_2_corr.keys()), list(sig_2_corr.values()), marker="+", color = color[metric], label=f"P. {metric}-based")
                ax.axvline(2, linestyle = "--", color = "black", linewidth = 0.7)
                ax.legend()
                #ax.set_title(f"P. {metric}-based")
                ax.set_xlabel(r"$\sigma^2$")
                ax.set_ylabel(r"$corr(C_{\sigma_i^2}, C_{\sigma^2_{ref}})$")
    plt.suptitle(r"Effect of $\sigma^2$ on the correlation when infering $c_1, c_2, c_3$"+"\n"+r"$\sigma^2_{ref} = 2$")
    plt.tight_layout()
    plt.show()

#mean distance small genome synthetic/true
if 0:
    fig, ax = plt.subplots(figsize=(10, 8))
    sigma = "variable"
    noisy = "noisy"
    color = {"row": "red", "col": "blue", "row_col": "green", "vector": "goldenrod"}
    theta_ref = (151584, 238325, 114499) #c_i, c_j

    for i, metric in enumerate(["row", "col", "row_col", "vector"]):

            d_mean_list = {}
            with open(f"simulation_little_genome/true/res_32000/{noisy}/sigma_{sigma}/0_theta_S_corr_inter_{metric}", 'rb') as f:
                        theta_corr = pickle.load(f)
                        print(len(theta_corr))

            eps_corr_min = min(theta_corr.values())
            eps_corr_max = max(theta_corr.values())


            theta_corr_sorted = dict(sorted(theta_corr.items(), key=lambda item: item[1]))
            start = int(len(theta_corr_sorted)*0.75) #3eme quartile (75%)
            eps_corr_start = list(theta_corr_sorted.values())[start]

            print(len(list(theta_corr_sorted.values())[start:]))

            #eps_corr_list = np.linspace(eps_corr_min+(eps_corr_max- eps_corr_min)*0.75,eps_corr_max,100)
            eps_corr_list = np.linspace(eps_corr_start,eps_corr_max,100)

            d_mean_list = mean_distance_theta(theta_corr, theta_ref, eps_corr_list)
            #print(d_mean_list)
            ax.axhline(32000, linestyle = "--", color = "black", linewidth = 0.7)

            ax.scatter(eps_corr_list, d_mean_list, marker="+", color = color[metric], label=f"S. {metric}-based")
            ax.legend()
            #ax.set_title(f"P. {metric}-based")
            ax.set_xlabel(r"$\epsilon_{corr}$")
            ax.set_ylabel(r"$mean(||\theta_i - \theta_{ref}||, \theta_i$ s.t. $corr(C_{\theta_i}, C_{\theta_{ref}}) \geq \epsilon_{corr}$ (bp)")
    plt.suptitle(fr"Inference of 3 centromeres $c_1, c_2, c_3$ - {noisy} - $\sigma^2$ {sigma}")
    plt.tight_layout()
    plt.show()

#plot mean distance VS correlation for a small genome 1 corr per chr
if 0:
    nb_seq = 3
    nb_train = 1000
    fig, ax = plt.subplots(figsize=(10, 8))
    #color = {"inter_row_col": "red", "inter_vector": "blue", "upper_vector": "green"}
    #color = {0: "red", 1: "blue", 2 : "green", 3: "yellow"}
    color = plt.cm.RdYlBu_r(np.linspace(0, 1, nb_seq+1))
    theta_ref = torch.tensor([151584, 238325, 114499]) #c_i, c_j

    for j in range(nb_seq+1):


        d_mean_list = []
        with open(f"simulation_little_genome/true/res_32000/noisy/sigma_fixe/sequential/{j}_theta_P_corr_inter_vector_per_row", 'rb') as f:
                    theta_corr_dict = pickle.load(f)

        prop_list = np.linspace(0.25,0.01,100)
        for prop in prop_list:

            nb_sel = int(prop*nb_train)

            theta_accepted = torch.zeros((len(chr_seq), nb_sel))

            for i, chr in enumerate(chr_seq.keys()):
                start_chr = len(theta_corr_dict[chr])-nb_sel

                chr_corr_sorted= dict(sorted(theta_corr_dict[chr].items(), key=lambda item: item[1])) #sort by values
                chr_accepted = list(dict(list(chr_corr_sorted.items())[start_chr:]).keys()) #take theta:corr_inter accepted
                chr_accepted = torch.tensor(chr_accepted)

                theta_accepted[i] = chr_accepted

            d_mean_list.append(mean_distance_theta_given_prop(theta_accepted, theta_ref))

        ax.scatter(prop_list, d_mean_list, marker="+", color = color[j], label=f"ABC round {j}")
    ax.invert_xaxis()
    ax.legend()
    ax.axhline(y=32000, color="black", linestyle='--', linewidth = 1)

    ax.set_xlabel(r"proportion of selected $\theta$")
    ax.set_ylabel(r"$mean(||\theta_i - \theta_{ref}||, \theta_i$ selected by $corr(C_{\theta_i}, C_{\theta_{ref}})$ (bp)")
    plt.suptitle(r"Inference of 3 centromeres $c_1, c_2, c_3$"+"\n P-vector based per row")
    plt.tight_layout()
    # plt.ylim((10000, 16000))
    # plt.xlim((0.05, 0.6))
    plt.show()

#histogram 1d 1 corr per chr
if 0:
    for nb_seq in range(4):
        with open(f"simulation_little_genome/true/res_32000/noisy/sigma_fixe/sequential/{nb_seq}_theta_P_corr_inter_vector_per_row", 'rb') as f:
                    theta_corr_dict = pickle.load(f)


        fig, axes = plt.subplots(2,2, figsize=(12, 10))
        nb_train = 1000
        prop = 0.05
        nb_sel = int(prop*nb_train)

        for k, chr in enumerate(chr_seq.keys()):
                start_chr = len(theta_corr_dict[chr])-nb_sel

                chr_corr_sorted= dict(sorted(theta_corr_dict[chr].items(), key=lambda item: item[1])) #sort by values
                chr_accepted = list(dict(list(chr_corr_sorted.items())[start_chr:]).keys()) #take theta:corr_inter accepted
                chr_accepted = torch.tensor(chr_accepted)
                df = pd.DataFrame(chr_accepted, columns=["theta"])
                sns.histplot(ax = axes[k//2, k%2], data=df, x="theta", kde=True, alpha=0.3, edgecolor='cornflowerblue')

                axes[k//2, k%2].set_xlabel(r"$\theta_{centro}$")
                axes[k//2, k%2].axvline(x=chr_cen[chr], linestyle='--', color="goldenrod", label=rf"$\theta_0$")
                thetas = list(theta_corr_dict[chr].keys())
                axes[k//2, k%2].scatter(thetas, [0]*len(thetas), marker="+", color='red', label=r"$\theta$ not accepted")
                axes[k//2, k%2].scatter(chr_accepted, [0]*len(chr_accepted), marker='+', color="navy", label = r"$\theta$ accepted")
                axes[k//2, k%2].legend([f"({len(chr_accepted)} / {len(thetas)}) \n {int(len(chr_accepted)/len(thetas)*100)}% accepted", rf"$\theta_0$", r"$\theta$ not accepted", r"$\theta$ accepted"], bbox_to_anchor=(0.85,1.01))
                axes[k//2, k%2].set_title(rf"$\theta_{{centro}}$ for chr ${k+1}$")
                sec_x = axes[k//2, k%2].secondary_xaxis(location='bottom') #draw the separation between chr
                sec_x.set_xticks([1, chr_seq[chr]], labels=[])
                sec_x.tick_params('x', length=10, width=1.5)
                extraticks = [1, chr_seq[chr]]
                axes[k//2, k%2].set_xticks(list(axes[k//2, k%2].get_xticks()) + extraticks)
                axes[k//2, k%2].set_xlim(1, chr_seq[chr])
        axes[1,1].axis("off")
        plt.suptitle(rf"$p(\theta_{{centro}}|C_{{ref}})$ for $\theta$ s.t. with {prop*100} % of highest $corr(C_{{\theta}}, C_{{ref}})$")
        plt.tight_layout()
        plt.show()

#plot mean distance VS correlation for a small genome 1 corr for all chr
if 0:
    nb_seq = 2
    resolution = 10000
    fig, ax = plt.subplots(figsize=(10, 8))
    #color = {"inter_row_col": "red", "inter_vector": "blue", "upper_vector": "green"}
    color = {0: "red", 1: "blue", 2 : "green", 3: "yellow"}
    #color = plt.cm.RdYlBu_r(np.linspace(0, 1, nb_seq+1))
    theta_ref = (151584, 238325, 114499) #c_i, c_j

    for i in range(nb_seq+1):


        d_mean_list = {}
        with open(f"simulation_little_genome/true/res_{resolution}/noisy/sigma_fixe/sequential/{i}_theta_P_corr_inter_vector", 'rb') as f:
                    theta_corr = pickle.load(f)


        eps_corr_min = min(theta_corr.values())
        eps_corr_max = max(theta_corr.values())

        print(eps_corr_min, eps_corr_max)
        theta_corr_sorted = dict(sorted(theta_corr.items(), key=lambda item: item[1]))
        start = int(len(theta_corr_sorted)*0.75)
        eps_corr_start = list(theta_corr_sorted.values())[start]

        print(len(list(theta_corr_sorted.values())[start:]))



        eps_corr_list = np.linspace(eps_corr_start,eps_corr_max,100)

        d_mean_list = mean_distance_theta(theta_corr, theta_ref, eps_corr_list)

        ax.scatter(eps_corr_list, d_mean_list, marker="+", color = color[i], label=f"ABC round {i}")
    ax.legend()
    ax.axhline(y=resolution, color="black", linestyle='--', linewidth = 1)

    ax.set_xlabel(r"$\epsilon_{corr}$")
    ax.set_ylabel(r"$mean(||\theta_i - \theta_{ref}||, \theta_i$ s.t. $corr(C_{\theta_i}, C_{\theta_{ref}}) \geq \epsilon_{corr}$ (bp)")
    plt.suptitle(r"Inference of 3 centromeres $c_1, c_2, c_3$"+"\n P-vector based")
    plt.tight_layout()
    # plt.ylim((10000, 16000))
    # plt.xlim((0.05, 0.6))
    plt.show()

#histogram 1d 1 corr for all chr sequential
if 0:
    resolution = 32000
    for metric in ["vector"]:
        for nb_seq in range(1):
            with open(f"simulation_little_genome/true/res_{resolution}/noisy/sigma_variable/summary_stat/0_theta_dnn", 'rb') as f:
                            theta_corr = pickle.load(f)

            fig, axes = plt.subplots(2,2, figsize=(12, 10))


            for k, chr in enumerate (list(chr_seq.keys())):
                ######### take the first centro ###########
                theta_corr_1 = {}
                for t in theta_corr.keys():
                    theta_corr_1[t[k]] = theta_corr[t] #{centro_1 : corr}
                ###########################################

                for prop in [0.05]:#np.linspace(0.05,0.26, 5):
                    start = int(len(theta_corr_1)*(1-prop))-1


                    theta_corr_1_sorted= dict(sorted(theta_corr_1.items(), key=lambda item: item[1])) #sort by values
                    thetas_corr_1_accepted = dict(list(theta_corr_1_sorted.items())[start:]) #take theta:corr_inter accepted
                    thresh = list(thetas_corr_1_accepted.values())[0] #take the corre corresponding to the prop

                    thetas_accepted = list(thetas_corr_1_accepted.keys()) #take the thetas accepted

                    #df = pd.DataFrame(thetas_corr_accepted.items(), columns=["theta", "corr_inter"])
                    df = pd.DataFrame(thetas_corr_1_accepted.items(), columns=["theta", "corr_inter"])

                    #np.array(thetas)[np.array(corr_inter)>= thresh]
                    sns.histplot(ax = axes[k//2, k%2], data=df, x="theta", kde=True, alpha=0.3, edgecolor='cornflowerblue')

                    #sns.kdeplot(data=thetas_accepted, color="cornflowerblue")
                    print(k)
                    axes[k//2, k%2].set_xlabel(r"$\theta_{centro}$")
                    axes[k//2, k%2].axvline(x=chr_cen[chr], linestyle='--', color="goldenrod", label=rf"$\theta_0$")
                    thetas = list(theta_corr_1.keys())
                    axes[k//2, k%2].scatter(thetas, [0]*len(thetas), marker="+", color='red', label=r"$\theta$ not accepted")
                    axes[k//2, k%2].scatter(thetas_accepted, [0]*len(thetas_accepted), marker='+', color="navy", label = r"$\theta$ accepted")
                    axes[k//2, k%2].legend([f"({len(thetas_accepted)} / {len(thetas)}) \n {int(len(thetas_accepted)/len(thetas)*100)}% accepted", rf"$\theta_0$", r"$\theta$ not accepted", r"$\theta$ accepted"], bbox_to_anchor=(0.85,1.01))
                    axes[k//2, k%2].set_title(rf"$\theta_{{centro}}$ for chr ${k+1}$")
                    sec_x = axes[k//2, k%2].secondary_xaxis(location='bottom') #draw the separation between chr
                    sec_x.set_xticks([1, chr_seq[chr]], labels=[])
                    sec_x.tick_params('x', length=10, width=1.5)
                    extraticks = [1, chr_seq[chr]]
                    axes[k//2, k%2].set_xticks(list(axes[k//2, k%2].get_xticks()) + extraticks)
                    axes[k//2, k%2].set_xlim(1, chr_seq[chr])
            axes[1,1].axis("off")
            plt.suptitle(rf"$p(\theta_{{centro}}|C_{{ref}})$ for $\theta$ s.t. $corr(C_{{\theta}}, C_{{ref}}) \geq \epsilon_{{corr}} = {thresh:.3f}$"+f"\n {metric}-based Spearman correlation ")
            plt.tight_layout()
            plt.show()
            #plt.savefig(f"{prop}.svg")

#kde 2d
if 0:
    with open(f"simulation_little_genome/true/res_10000/noisy/theta_P_corr_inter_vector", 'rb') as f:
                        theta_corr = pickle.load(f)

    fig, axes = plt.subplots(2,2, figsize=(12, 10))

    k=0
    for (chr_1, chr_2) in combinations(chr_seq.keys(),r=2):



        for prop in [0.05]:#np.linspace(0.05,0.26, 5):
            start = int(len(theta_corr)*(1-prop))-1


            theta_corr_sorted= dict(sorted(theta_corr.items(), key=lambda item: item[1])) #sort by values
            thetas_corr_accepted = dict(list(theta_corr_sorted.items())[start:]) #take theta:corr_inter accepted
            thresh = list(thetas_corr_accepted.values())[0] #take the corre corresponding to the prop

            thetas_accepted = list(thetas_corr_accepted.keys()) #take the thetas accepted


            #df = pd.DataFrame(thetas_corr_accepted.items(), columns=["theta", "corr_inter"])
            df_all = pd.DataFrame(list(theta_corr.keys()), columns=["chr01", "chr02", "chr03"])
            df_accepted = pd.DataFrame(thetas_accepted, columns=["chr01", "chr02", "chr03"])


            #np.array(thetas)[np.array(corr_inter)>= thresh]
            #sns.histplot(ax = axes[k//2, k%2], data=df, x="theta", kde=True, alpha=0.3, edgecolor='cornflowerblue')

            sns.kdeplot(ax = axes[k//2, k%2], data=df_accepted, x=f"{chr_1}", y=f"{chr_2}", color="cornflowerblue", fill=True)

            print(k)
            axes[k//2, k%2].set_xlabel(rf"$\theta_{{centro}}$ {chr_1}")
            axes[k//2, k%2].set_ylabel(rf"$\theta_{{centro}}$ {chr_2}")
    #         axes[k//2, k%2].axvline(x=chr_cen[chr], linestyle='--', color="goldenrod", label=rf"$\theta_0$")
    #         thetas = list(theta_corr_1.keys())
            axes[k//2, k%2].scatter(df_all[chr_1], df_all[chr_2], marker='+', color="red", alpha = 0.3, label = r"$\theta$ not accepted")
            axes[k//2, k%2].scatter(df_accepted[chr_1], df_accepted[chr_2], marker='+', color="navy", alpha = 0.5, label = r"$\theta$ accepted")
            axes[k//2, k%2].scatter(x=chr_cen[chr_1],y=chr_cen[chr_2], marker="*", s = 30, color='goldenrod', label=rf"$\theta_{{ref}}$ {chr_1}-{chr_2}")

            axes[k//2, k%2].legend()
    #         axes[k//2, k%2].set_title(rf"$\theta_{{centro}}$ for chr ${k+1}$")
            sec_x = axes[k//2, k%2].secondary_xaxis(location='bottom') #draw the separation between chr
            sec_x.set_xticks([1, chr_seq[chr_1]], labels=[])
            sec_x.tick_params('x', length=10, width=1.5)
            extraticks = [1, chr_seq[chr_1]]
            axes[k//2, k%2].set_xticks(list(axes[k//2, k%2].get_xticks()) + extraticks)
            sec_y = axes[k//2, k%2].secondary_yaxis(location='left') #draw the separation between chr
            sec_y.set_yticks([1, chr_seq[chr_2]], labels=[])
            sec_y.tick_params('y', length=10, width=1.5)
            extraticks = [1, chr_seq[chr_2]]
            axes[k//2, k%2].set_yticks(list(axes[k//2, k%2].get_yticks()) + extraticks)
            axes[k//2, k%2].set_xlim(1, chr_seq[chr_1])
            axes[k//2, k%2].set_ylim(1, chr_seq[chr_2])
            k+=1
    axes[1,1].axis("off")
    plt.suptitle(rf"$p(\theta_{{centro}}|C_{{ref}})$ for $\theta$ s.t. $corr(C_{{\theta}}, C_{{ref}}) \geq \epsilon_{{corr}} = {thresh:.3f}$")
    plt.tight_layout()
    plt.show()

#multiple histogram 1d 1 corr for all chr sequential
if 0:
    background_color = "lightgray"
    # background_color = "white"
    resolution = 3200
    fig, axes = plt.subplots(2,2, figsize=(12, 10))
    nb_seq = 10
    color = plt.cm.RdYlBu_r(np.linspace(0, 1, nb_seq+1))
    sigma = "variable"
    origin = "true"
    noisy = "noisy"
    wavelets = 'wavelets/3_levels/'
    #wavelets = ''
    nb_levels = 3
    metric = "vector"
    for metric in ["vector"]:
        for nb_seq in range(nb_seq+1):
            path = f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{wavelets}{nb_seq}_theta_P_corr_inter_{metric}"

            with open(path, 'rb') as f:
                            theta_corr = pickle.load(f)

            # with open(f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{wavelets}/{nb_levels}_levels/{nb_seq}_weights", 'rb') as f:
            #                 weights = pickle.load(f)

            for k, chr in enumerate (list(chr_seq.keys())):
                ######### take the first centro ###########
                theta_corr_1 = {}
                for t in theta_corr.keys():

                    theta_corr_1[t[k]] = theta_corr[t] #{centro_1 : corr}
                ###########################################
                print("nb_seq", nb_seq)
                for prop in [0.05]:#np.linspace(0.05,0.26, 5):
                    start = int(len(theta_corr_1)*(1-prop))-1


                    theta_corr_1_sorted= dict(sorted(theta_corr_1.items(), key=lambda item: item[1])) #sort by values
                    thetas_corr_1_accepted = dict(list(theta_corr_1_sorted.items())[start:]) #take theta:corr_inter accepted


                    thetas_accepted = list(thetas_corr_1_accepted.keys()) #take the thetas accepted

                    #df = pd.DataFrame(thetas_corr_accepted.items(), columns=["theta", "corr_inter"])
                    df = pd.DataFrame(thetas_corr_1_accepted.items(), columns=["theta", "corr_inter"])

                    #np.array(thetas)[np.array(corr_inter)>= thresh]
                    sns.kdeplot(ax = axes[k//2, k%2], data=df, x="theta", color=color[nb_seq], label=f"ABC round {nb_seq}")

                    #sns.kdeplot(data=thetas_accepted, color="cornflowerblue")
                    print(k)
                    axes[k//2, k%2].set_xlabel(r"$\theta_{centro}$")
                    axes[k//2, k%2].axvline(x=chr_cen[chr], linestyle='--', color="goldenrod", label=rf"$\theta_0$")
                    # thetas = list(theta_corr_1.keys())
                    # axes[k//2, k%2].scatter(thetas, [0]*len(thetas), marker="+", color='red', label=r"$\theta$ not accepted")
                    # axes[k//2, k%2].scatter(thetas_accepted, [0]*len(thetas_accepted), marker='+', color="navy", label = r"$\theta$ accepted")
                    # axes[k//2, k%2].legend([f"({len(thetas_accepted)} / {len(thetas)}) \n {int(len(thetas_accepted)/len(thetas)*100)}% accepted", rf"$\theta_0$", r"$\theta$ not accepted", r"$\theta$ accepted"], bbox_to_anchor=(0.85,1.01))
                    #axes[k//2, k%2].legend()
                    axes[k//2, k%2].set_title(rf"$\theta_{{centro}}$ for chr ${k+1}$")
                    sec_x = axes[k//2, k%2].secondary_xaxis(location='bottom') #draw the separation between chr
                    sec_x.set_xticks([1, chr_seq[chr]], labels=[])
                    sec_x.tick_params('x', length=10, width=1.5)
                    extraticks = [1, chr_seq[chr]]
                    if nb_seq==0:
                        axes[k//2, k%2].set_xticks(list(axes[k//2, k%2].get_xticks()) + extraticks)
                    axes[k//2, k%2].set_xlim(1, chr_seq[chr])
                    axes[k//2, k%2].set_facecolor(background_color)



    theta_ref = (151584, 238325, 114499) #c_i, c_j
    # theta_ref = (151584, 238325, 114499, 449819,
    #              152103, 148622, 497042, 105698,
    #              355742, 436418, 439889, 150946,
    #              268149, 628877, 326703, 556070)

    for i in range(nb_seq+1):
        path = f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{wavelets}{i}_theta_P_corr_inter_{metric}"


        d_mean_list = {}
        with open(path, 'rb') as f:
                    theta_corr = pickle.load(f)


        eps_corr_min = min(theta_corr.values())
        eps_corr_max = max(theta_corr.values())

        print(eps_corr_min, eps_corr_max)
        theta_corr_sorted = dict(sorted(theta_corr.items(), key=lambda item: item[1]))
        start = int(len(theta_corr_sorted)*0.75)
        eps_corr_start = list(theta_corr_sorted.values())[start]

        print(len(list(theta_corr_sorted.values())[start:]))



        eps_corr_list = np.linspace(eps_corr_start,eps_corr_max,100)

        d_mean_list = mean_distance_theta(theta_corr, theta_ref, eps_corr_list)

        axes[1,1].scatter(eps_corr_list, d_mean_list, marker="+", color = color[i], label=f"ABC round {i}")
    axes[1,1].legend()
    axes[1,1].axhline(y=resolution, color="black", linestyle='--', linewidth = 1)
    # axes[1,1].set_ylim(0,1.5*resolution)
    # axes[1,1].set_xlim(0.08,0.13)
    axes[1,1].set_xlabel(r"$\epsilon_{corr}$")
    axes[1,1].set_ylabel(r"$mean(||\theta_i - \theta_{ref}||, \theta_i$ s.t. $corr(C_{\theta_i}, C_{\theta_{ref}}) \geq \epsilon_{corr}$ (bp)")
    axes[1,1].set_title(r"mean distance of $\theta$ accepted to $\theta_{ref}$")
    axes[1,1].set_facecolor(background_color)

    plt.suptitle(rf"$p(\theta_{{centro}}|C_{{ref}})$ for {prop*100} % of $\theta$ with highest $corr(C_{{\theta}}, C_{{ref}})$"+f"\n {origin} data - {noisy} - {metric}-based Pearson correlation"+fr" - $\sigma^2$ {sigma} - res {resolution}" + f"\n {wavelets} - {nb_levels} levels")
    plt.tight_layout()
    plt.show()


#multiple weighted histogram 1d 1 corr for all chr sequential
if 0:
    background_color = "lightgray"
    # background_color = "white"
    resolution = 32000
    fig, axes = plt.subplots(4,5, figsize=(12, 10))
    gs = fig.add_gridspec(4,5)

    ax3 = fig.add_subplot(gs[:, 4])

    nb_seq = 6
    color = plt.cm.RdYlBu_r(np.linspace(0, 1, nb_seq+1))
    sigma = "variable"
    origin = "true"
    noisy = "noisy"
    nb_levels = 3
    #wavelets = f'wavelets/{nb_levels}_levels/'
    wavelets = ''
    prop = 0.05
    for metric in ["vector"]:
        for nb_seq in range(nb_seq+1):
            with open(f"simulation_little_genome/16_chr/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/{wavelets}{nb_seq}_thetas_accepted", 'rb') as f:
                            thetas_accepted = pickle.load(f)
            with open(f"simulation_little_genome/16_chr/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/{wavelets}{nb_seq}_weights", 'rb') as f:
                            weights = pickle.load(f)

            for k, chr in enumerate (list(chr_seq.keys())):

                    df = pd.DataFrame(thetas_accepted[:,k], columns=["theta"])

                    #np.array(thetas)[np.array(corr_inter)>= thresh]
                    sns.kdeplot(ax = axes[k//4, k%4], data=df, x="theta", color=color[nb_seq], label=f"ABC round {nb_seq}")

                    #sns.kdeplot(data=thetas_accepted, color="cornflowerblue")
                    print(k)
                    axes[k//4, k%4].set_xlabel(r"$\theta_{centro}$")
                    axes[k//4, k%4].axvline(x=chr_cen[chr], linestyle='--', color="goldenrod", label=rf"$\theta_0$")
                    # thetas = list(theta_corr_1.keys())
                    # axes[k//2, k%2].scatter(thetas, [0]*len(thetas), marker="+", color='red', label=r"$\theta$ not accepted")
                    # axes[k//2, k%2].scatter(thetas_accepted, [0]*len(thetas_accepted), marker='+', color="navy", label = r"$\theta$ accepted")
                    # axes[k//2, k%2].legend([f"({len(thetas_accepted)} / {len(thetas)}) \n {int(len(thetas_accepted)/len(thetas)*100)}% accepted", rf"$\theta_0$", r"$\theta$ not accepted", r"$\theta$ accepted"], bbox_to_anchor=(0.85,1.01))
                    #axes[k//2, k%2].legend()
                    axes[k//4, k%4].set_title(rf"$\theta_{{centro}}$ for chr ${k+1}$")
                    sec_x = axes[k//4, k%4].secondary_xaxis(location='bottom') #draw the separation between chr
                    sec_x.set_xticks([1, chr_seq[chr]], labels=[])
                    sec_x.tick_params('x', length=10, width=1.5)
                    extraticks = [1, chr_seq[chr]]
                    if nb_seq==0:
                        axes[k//4, k%4].set_xticks(list(axes[k//4, k%4].get_xticks()) + extraticks)
                    axes[k//4, k%4].set_xlim(1, chr_seq[chr])
                    axes[k//4, k%4].set_facecolor(background_color)



    #theta_ref = (151584, 238325, 114499) #c_i, c_j
    theta_ref = (151584, 238325, 114499, 449819,
                 152103, 148622, 497042, 105698,
                 355742, 436418, 439889, 150946,
                 268149, 628877, 326703, 556070)

    for i in range(nb_seq+1):


        d_mean_list = {}
        with open(f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{wavelets}{i}_theta_P_corr_inter_vector", 'rb') as f:
                    theta_corr = pickle.load(f)


        eps_corr_min = min(theta_corr.values())
        eps_corr_max = max(theta_corr.values())

        print(eps_corr_min, eps_corr_max)
        theta_corr_sorted = dict(sorted(theta_corr.items(), key=lambda item: item[1]))
        start = int(len(theta_corr_sorted)*0.75)
        eps_corr_start = list(theta_corr_sorted.values())[start]

        print(len(list(theta_corr_sorted.values())[start:]))

        eps_corr_list = np.linspace(eps_corr_start,eps_corr_max,100)

        d_mean_list = mean_distance_theta(theta_corr, theta_ref, eps_corr_list)

        ax3.scatter(d_mean_list, eps_corr_list, marker="+", color = color[i], label=f"ABC round {i}")
    ax3.legend()
    ax3.axvline(x=resolution, color="black", linestyle='--', linewidth = 1)
    # ax3.set_ylim(0,1.5*resolution)
    # ax3.set_xlim(0.08,0.13)

    for i in range(4):
        axes[i][4].set_yticklabels([])  # Remove y-tick labels
        axes[i][4].set_ylabel("")  # Remove y-axis label
        axes[i][4].set_xticklabels([])  # Remove y-tick labels
        axes[i][4].set_xlabel("")  # Remove y-axis label


    ax3.set_ylabel(r"$\epsilon_{corr}$")
    ax3.set_xlabel(r"$mean(||\theta_i - \theta_{ref}||$," + "\n" + r"$\theta_i$ s.t. $corr(C_{\theta_i}, C_{\theta_{ref}}) \geq \epsilon_{corr}$ (bp)")
    ax3.set_title(r"mean distance of $\theta$" + "\n" r"accepted to $\theta_{ref}$")
    ax3.set_facecolor(background_color)
    plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.07, wspace=0.4, hspace = 0.7)
    plt.suptitle(rf"$p(\theta_{{centro}}|C_{{ref}})$ for {prop*100} % of $\theta$ with highest $corr(C_{{\theta}}, C_{{ref}})$"+f"\n {origin} data - {noisy} - {metric}-based Pearson correlation"+fr" - $\sigma^2$ {sigma} - res {resolution}" + f"\n {wavelets}")
    #plt.tight_layout()
    plt.show()

#multiple histogram 1d 1 corr for all chr with/without summary stat
if 1:
    background_color = "lightgray"
    # background_color = "white"
    resolution = 3200
    theta_ref = (151584, 238325, 114499) #c_i, c_j
    fig, axes = plt.subplots(2,2, figsize=(12, 10))
    nb_seq = 0
    color_stat = plt.cm.Blues_r(np.linspace(0, 1, nb_seq+2))
    color_without_stat = plt.cm.Reds_r(np.linspace(0, 1, nb_seq+2))
    sigma = "variable"
    origin = "true"
    noisy = "noisy"
    #wavelets = 'wavelets'
    wavelets = ''
    nb_levels = 3

    for nb_seq in range(nb_seq+1):
        path_stat = f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/summary_stat/CNN/sigmoid/sequential/{nb_seq}_theta_dnn"
        path_without_stat = f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{nb_seq}_theta_P_corr_inter_vector"
        with open(path_stat, 'rb') as f:
                theta_dist = pickle.load(f)
        with open(path_without_stat, 'rb') as f:
                theta_corr = pickle.load(f)
        print(theta_dist)
        # with open(f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{wavelets}/{nb_levels}_levels/{nb_seq}_weights", 'rb') as f:
        #                 weights = pickle.load(f)

        for k, chr in enumerate (list(chr_seq.keys())):
            ######### take the first centro ###########
            theta_dist_1 = {}
            theta_corr_1 = {}
            for t in theta_dist.keys():
                theta_dist_1[t[k].item()] = theta_dist[t] #{centro_1 : dist to ref}
            for t in theta_corr.keys():
                theta_corr_1[t[k]] = theta_corr[t] #{centro_1 : dist to ref}
            ###########################################

            for prop in [0.05]:#np.linspace(0.05,0.26, 5):
                start_stat = int(len(theta_dist_1)*(1-prop))-1
                start_without_stat = int(len(theta_corr_1)*(1-prop))-1

                theta_dist_1_sorted= dict(sorted(theta_dist_1.items(), key=lambda item: item[1], reverse=True)) #sort by values
                thetas_dist_1_accepted = dict(list(theta_dist_1_sorted.items())[start_stat:]) #take theta:corr_inter accepted

                theta_corr_1_sorted= dict(sorted(theta_corr_1.items(), key=lambda item: item[1])) #sort by values
                thetas_corr_1_accepted = dict(list(theta_corr_1_sorted.items())[start_without_stat:]) #take theta:corr_inter accepted


                df_stat = pd.DataFrame(thetas_dist_1_accepted.items(), columns=["theta", "dist"])
                df_without_stat = pd.DataFrame(thetas_corr_1_accepted.items(), columns=["theta", "corr_inter"])

                sns.kdeplot(ax = axes[k//2, k%2], data=df_stat, x="theta", color=color_stat[nb_seq], label=f"ABC round {nb_seq} summary stat")
                sns.kdeplot(ax = axes[k//2, k%2], data=df_without_stat, x="theta", color=color_without_stat[nb_seq], label=f"ABC round {nb_seq}")

                print(k)
                axes[k//2, k%2].set_xlabel(r"$\theta_{centro}$")
                axes[k//2, k%2].axvline(x=chr_cen[chr], linestyle='--', color="goldenrod", label=rf"$\theta_0$")
                axes[k//2, k%2].set_title(rf"$\theta_{{centro}}$ for chr ${k+1}$")

                sec_x = axes[k//2, k%2].secondary_xaxis(location='bottom') #draw the separation between chr
                sec_x.set_xticks([1, chr_seq[chr]], labels=[])
                sec_x.tick_params('x', length=10, width=1.5)
                extraticks = [1, chr_seq[chr]]
                if nb_seq==0:
                    axes[k//2, k%2].set_xticks(list(axes[k//2, k%2].get_xticks()) + extraticks)
                axes[k//2, k%2].set_xlim(1, chr_seq[chr])
                axes[k//2, k%2].set_facecolor(background_color)

    for i in range(nb_seq+1):
        path_stat = f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/summary_stat/CNN/sigmoid/sequential/{i}_theta_dnn"
        path_without_stat = f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{i}_theta_P_corr_inter_vector"
        with open(path_stat, 'rb') as f:
                theta_dist = pickle.load(f)
        with open(path_without_stat, 'rb') as f:
                theta_corr = pickle.load(f)

        eps_dist_min = min(theta_dist.values())

        print("min dist", eps_dist_min)

        eps_corr_max = max(theta_corr.values())
        print("max corr", eps_corr_max)

        theta_dist_sorted = dict(sorted(theta_dist.items(), key=lambda item: item[1], reverse=True))
        theta_corr_sorted = dict(sorted(theta_corr.items(), key=lambda item: item[1]))

        start_stat = int(len(theta_dist_sorted)*0.75)
        start_without_stat = int(len(theta_corr_sorted)*0.75)

        eps_dist_start = list(theta_dist_sorted.values())[start_stat]
        eps_corr_start = list(theta_corr_sorted.values())[start_without_stat]

        print(len(list(theta_dist_sorted.values())[start_stat:]))
        print(len(list(theta_corr_sorted.values())[start_without_stat:]))

        eps_dist_list = np.linspace(eps_dist_start,eps_dist_min,100)
        eps_corr_list = np.linspace(eps_corr_start,eps_corr_max,100)

        d_mean_list_stat = mean_distance_closest_theta(theta_dist, theta_ref, eps_dist_list)
        d_mean_list_without_stat = mean_distance_theta(theta_corr, theta_ref, eps_corr_list)

        axes[1,1].scatter(np.linspace(25,1,100), d_mean_list_stat, marker="+", color = color_stat[i], label=f"ABC round {i} summary stat")
        axes[1,1].scatter(np.linspace(25,1,100), d_mean_list_without_stat, marker="+", color = color_without_stat[i], label=f"ABC round {i}")

    axes[1,1].legend()
    axes[1,1].axhline(y=resolution, color="black", linestyle='--', linewidth = 1)
    # axes[1,1].set_ylim(0,1.5*resolution)
    # axes[1,1].set_xlim(0.08,0.13)
    axes[1,1].set_xlabel(r"% of selected best $\theta$" + "\n" + r"(by $||DNN(C_{\theta}) - DNN(C_{ref})||$ or $corr(C_{\theta}, C_{ref}$))")
    axes[1,1].set_ylabel(r"$mean(||\theta_i - \theta_{ref}||, \theta_i$ s.t. $||\theta_i - \theta_{ref}|| \leq \epsilon_{\theta})$ (bp)")
    axes[1,1].set_title(r"mean distance of $\theta$ accepted to $\theta_{ref}$")
    axes[1,1].set_facecolor(background_color)
    axes[1,1].invert_xaxis()  # Reverses the y-axis

    plt.suptitle(rf"$p(\theta_{{centro}}|C_{{ref}})$ for {prop*100} % of $\theta$ with closest $DNN(C_{{\theta}})$ to $DNN(C_{{ref}})$ or highest $corr(C_{{\theta}}, C_{{ref}})$"+f"\n {origin} data - {noisy} - DNN-summary stat"+fr" - $\sigma^2$ {sigma} - res {resolution}")
    plt.tight_layout()
    plt.show()
    #plt.savefig(f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/summary_stat/CNN/5000_linear_default_kernel_5.png")
    # plt.savefig(f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/summary_stat/MLP/5000_linear_2.png")

#best 5% mean distance multi resolutions
if 0:
    nb_seq = 10
    theta_ref = torch.tensor([151584, 238325, 114499])
    origin = "true"
    resolutions = [3200, 10000, 32000]
    colors = {3200:plt.cm.Reds(np.linspace(0, 1, nb_seq+1)), 10000:plt.cm.Blues(np.linspace(0, 1, nb_seq+1)), 32000:plt.cm.Greens(np.linspace(0, 1, nb_seq+1))}
    noisy = "noisy"
    sigma = "variable"
    wavelets = ""
    for resolution in resolutions:
        d_mean_list = []
        for i in range(nb_seq+1):

            with open(f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{wavelets}{i}_thetas_accepted", 'rb') as f:
                        thetas_accepted = pickle.load(f)


            d_mean_list.append(mean_distance_best_thetas(thetas_accepted, theta_ref))
            plt.scatter(i, d_mean_list[-1], color = colors[resolution][i])
        plt.axhline(y = resolution, linestyle="--", color="black")
        plt.plot(range(nb_seq+1), d_mean_list,  color = colors[resolution][nb_seq//2])
    red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=5, label='res 3200')
    blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                            markersize=5, label='res 10000')
    green_dot = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                          markersize=5, label='res 32000')


    plt.xlabel("ABC round")
    plt.ylabel(r"$mean(||\theta_i - \theta_{ref}||)$")
    plt.title(r"$mean(||\theta_i - \theta_{ref}||)$ for 5% 'best' $\theta$ (highest corr)"+f"\n 3 chr - {origin} data - {noisy} - sigma {sigma} - {wavelets}")
    plt.legend(handles=[red_dot, blue_dot, green_dot])
    plt.show()

#best 5% mean distance with/without wavelets
if 0:
    theta_ref = torch.tensor([151584, 238325, 114499])

    nb_seq = 4
    origin = "true"
    resolution = 3200
    colors = {"":plt.cm.Reds(np.linspace(0, 1, nb_seq+1)), "wavelets/3_levels/":plt.cm.Blues(np.linspace(0, 1, nb_seq+1))}
    noisy = "noisy"
    sigma = "variable"
    wave = ["", "wavelets/3_levels/"]

    for w in wave:
        d_mean_list = []

        for i in range(nb_seq+1):
            with open(f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{w}{i}_thetas_accepted", 'rb') as f:
                            thetas_accepted = pickle.load(f)

            d_mean_list.append(mean_distance_best_thetas(thetas_accepted, theta_ref))
            plt.scatter(i, d_mean_list[-1], color = colors[w][i])

        plt.plot(range(nb_seq+1), d_mean_list, color = colors[w][nb_seq//2], alpha = 0.5)
    red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=5, label='no wavelets')
    blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                            markersize=5, label='wavelets 3 levels')



    plt.xlabel("ABC round")
    plt.ylabel(r"$mean(||\theta_i - \theta_{ref}||$")
    plt.title(r"$mean(||\theta_i - \theta_{ref}||$ for 5% best $\theta$"+f"\n 3 chr - {origin} data - resolution {resolution} - {noisy} - sigma {sigma} - ")
    plt.legend(handles=[red_dot, blue_dot])
    plt.show()

#mmd multi resolutions
if 0:
    theta_ref = torch.tensor([151584, 238325, 114499])
    theta_ref = theta_ref.repeat(50,1)
    nb_seq = 10
    origin = "true"
    resolutions = [3200, 10000, 32000]
    colors = {3200:plt.cm.Reds(np.linspace(0, 1, nb_seq+1)), 10000:plt.cm.Blues(np.linspace(0, 1, nb_seq+1)), 32000:plt.cm.Greens(np.linspace(0, 1, nb_seq+1))}
    noisy = "noisy"
    sigma = "variable"
    wavelets = ""
    scale = -1

    for resolution in resolutions:
        mmd_rounds = []

        for i in range(11):
            with open(f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{wavelets}{i}_thetas_accepted", 'rb') as f:
                            thetas_accepted = pickle.load(f)
            if scale==-1 :
                xx = f_mmd(theta_ref, theta_ref)
                xy = f_mmd(theta_ref, thetas_accepted, diag=True)
                yy = f_mmd(thetas_accepted, thetas_accepted)
                scale = torch.median(torch.sqrt(torch.cat((xx, xy, yy))))
            print(resolution, i, "scale", scale)
            mmd_rounds.append(unbiased_mmd_squared(theta_ref, thetas_accepted, scale=scale))

            plt.scatter(i, mmd_rounds[-1], color = colors[resolution][i])

        plt.plot(range(11), mmd_rounds, color = colors[resolution][nb_seq//2], alpha = 0.5)
    red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=5, label='res 3200')
    blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                            markersize=5, label='res 10000')
    green_dot = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                          markersize=5, label='res 32000')


    plt.xlabel("ABC round")
    plt.ylabel("MMD")
    plt.title(r"MMD between $p(\theta|C_{ref})$ and $\delta_{\theta_{ref}}$"+f"\n 3 chr - {origin} data - {noisy} - sigma {sigma} - {wavelets}")
    plt.legend(handles=[red_dot, blue_dot, green_dot])
    plt.show()

#mmd with/without wavelets
if 0:
    theta_ref = torch.tensor([151584, 238325, 114499])
    theta_ref = theta_ref.repeat(50,1)
    nb_seq = 4
    origin = "true"
    resolution = 3200
    colors = {"":plt.cm.Reds(np.linspace(0, 1, nb_seq+1)), "wavelets/3_levels/":plt.cm.Blues(np.linspace(0, 1, nb_seq+1))}
    noisy = "noisy"
    sigma = "variable"
    wave = ["", "wavelets/3_levels/"]
    scale = -1

    for w in wave:
        mmd_rounds = []

        for i in range(nb_seq+1):
            with open(f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{w}{i}_thetas_accepted", 'rb') as f:
                            thetas_accepted = pickle.load(f)
            if scale==-1 :
                xx = f_mmd(theta_ref, theta_ref)
                xy = f_mmd(theta_ref, thetas_accepted, diag=True)
                yy = f_mmd(thetas_accepted, thetas_accepted)
                scale = torch.median(torch.sqrt(torch.cat((xx, xy, yy))))

            mmd_rounds.append(unbiased_mmd_squared(theta_ref, thetas_accepted, scale=scale))
            plt.scatter(i, mmd_rounds[-1], color = colors[w][i])

        plt.plot(range(nb_seq+1), mmd_rounds, color = colors[w][nb_seq//2], alpha = 0.5)
    red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=5, label='no wavelets')
    blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                            markersize=5, label='wavelets 3 levels')



    plt.xlabel("ABC round")
    plt.ylabel("MMD")
    plt.title(r"MMD between $p(\theta|C_{ref})$ and $\delta_{\theta_{ref}}$"+f"\n 3 chr - {origin} data - resolution {resolution} - {noisy} - sigma {sigma} - ")
    plt.legend(handles=[red_dot, blue_dot])
    plt.show()

#wasserstein distance
if 0:
    theta_ref = torch.tensor([151584, 238325, 114499])
    dim = 1
    nb_seq = 10
    origin = "true"
    resolutions = [3200, 10000, 32000]
    colors = {3200:plt.cm.Reds(np.linspace(0, 1, nb_seq+1)), 10000:plt.cm.Blues(np.linspace(0, 1, nb_seq+1)), 32000:plt.cm.Greens(np.linspace(0, 1, nb_seq+1))}
    noisy = "noisy"
    sigma = "variable"
    wavelets = ""
    #fig,ax = plt.subplots(1, dim, figsize=(10, 8))


    for resolution in resolutions:
        wasser_rounds = torch.zeros(nb_seq+1)

        for i in range(nb_seq+1):
            with open(f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{wavelets}{i}_thetas_accepted", 'rb') as f:
                            thetas_accepted = pickle.load(f)

            wasser_rounds[i] = torch.sqrt(wasserstein_distance(theta_ref, thetas_accepted))

        plt.plot(range(1,11), wasser_rounds[1:], '-o', color = colors[resolution][nb_seq//2], alpha = 0.5)
    plt.xlabel("ABC round")
    plt.ylabel("Wasserstein distance")
    red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                            markersize=5, label='res 3200')
    blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                            markersize=5, label='res 10000')
    green_dot = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                            markersize=5, label='res 32000')

    plt.suptitle(r"Wasserstein distance between $p(\theta|C_{ref})$ and $\delta_{\theta_{ref}}$"+f"\n 3 chr - {origin} data - {noisy} - sigma {sigma} - {wavelets}")
    plt.legend(handles=[red_dot, blue_dot, green_dot])
    plt.tight_layout()

    plt.show()

#all metrics : 5%, mmd, wasserstein
if 0:
    theta_ref = torch.tensor([151584, 238325, 114499])
    theta_ref_rep = theta_ref.repeat(50,1)
    scale = -1
    nb_seq = 10
    origin = "true"

    colors = {"3200":"red", "10000":"blue", "32000":"green", "3200_w":"orange"}
    # colors = {"sequential":"red", "summary_stat/CNN/sigmoid/sequential":"blue"}
    # resolution = 32000

    noisy = "noisy"
    sigma = "variable"
    w = ""
    fig,ax = plt.subplots(2, 2, figsize=(10, 8))


    for key in colors.keys():
        if key=="3200_w":
            resolution = 3200
            w = "wavelets/3_levels/"
        else:
            resolution=int(key)

        d_mean_list = []
        mmd_rounds = []
        wasser_rounds = torch.zeros(nb_seq+1)

        for i in range(nb_seq+1):

            with open(f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/sequential/{w}{i}_thetas_accepted", 'rb') as f:
            # with open(f"simulation_little_genome/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/{key}/{i}_thetas_accepted", 'rb') as f:
                            thetas_accepted = pickle.load(f)

            d_mean_list.append(mean_distance_best_thetas(thetas_accepted, theta_ref))

            if scale==-1 :
                xx = f_mmd(theta_ref_rep, theta_ref_rep)
                xy = f_mmd(theta_ref_rep, thetas_accepted, diag=True)
                yy = f_mmd(thetas_accepted, thetas_accepted)
                scale = torch.median(torch.sqrt(torch.cat((xx, xy, yy))))
                print(xx.size())
                print(xy.size())
                print(yy.size())
                print(torch.cat((xx, xy, yy)).size())

            mmd_rounds.append(unbiased_mmd_squared(theta_ref_rep, thetas_accepted, scale=scale))

            wasser_rounds[i] = torch.sqrt(wasserstein_distance(theta_ref, thetas_accepted))


        ax[0,0].plot(range(nb_seq+1), d_mean_list, '-o', color = colors[key], alpha = 0.5)
        #print(resolution)

        ax[0,0].axhline(y = resolution, linestyle="--", color="black")

        ax[0,1].plot(range(nb_seq+1), mmd_rounds, '-o', color = colors[key], alpha = 0.5)
        ax[1,0].plot(range(nb_seq+1), wasser_rounds, '-o', color = colors[key], alpha = 0.5)
        ax[1,1].plot(range(1,nb_seq+1), wasser_rounds[1:], '-o', color = colors[key], alpha = 0.5)


    red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=5, label='res 3200')
    blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                            markersize=5, label='res 10000')
    green_dot = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                            markersize=5, label='res 32000')
    orange_dot = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                            markersize=5, label='res 3200 wavelets')
    # red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
    #                       markersize=5, label='Pearson corr.')
    # blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
    #                         markersize=5, label='Summary stat')


    ax[0,0].set_xlabel("ABC round")
    ax[0,0].set_ylabel(r"$mean(||\theta_i - \theta_{ref}||$")
    ax[0,0].set_title(r"$mean(||\theta_i - \theta_{ref}||$ for 5% best $\theta$")
    ax[0,0].legend(handles=[red_dot, blue_dot, green_dot, orange_dot])
    # ax[0,0].legend(handles=[red_dot, blue_dot])

    ax[0,1].set_xlabel("ABC round")
    ax[0,1].set_ylabel("MMD")
    ax[0,1].set_title(r"MMD between $p(\theta|C_{ref})$ and $\delta_{\theta_{ref}}$")
    ax[0,1].legend(handles=[red_dot, blue_dot,green_dot, orange_dot])
    # ax[0,1].legend(handles=[red_dot, blue_dot])

    ax[1,0].set_xlabel("ABC round")
    ax[1,0].set_ylabel("Wasserstein distance")
    ax[1,0].set_title(r"Wasserstein distance between $p(\theta|C_{ref})$ and $\delta_{\theta_{ref}}$")
    ax[1,0].legend(handles=[red_dot, blue_dot, green_dot, orange_dot])
    # ax[1,0].legend(handles=[red_dot, blue_dot])

    plt.suptitle(f"Comparison of methods for 3 chr - {origin} data - {noisy} - sigma {sigma} - resolution {resolution}")
    plt.tight_layout()
    plt.show()
    
#scatter plot to inspect the prior range
if 0:
    background_color = "white"
    resolution = 32000
    fig, axes = plt.subplots(4,4, figsize=(12, 12))


    nb_seq = 6
    color = plt.cm.RdYlBu_r(np.linspace(0, 1, nb_seq+1))
    sigma = "variable"
    origin = "true"
    noisy = "noisy"
    # nb_levels = 3
    #wavelets = f'wavelets/{nb_levels}_levels/'
    wavelets = ''
    theta_ref = (151584, 238325, 114499, 449819,
                 152103, 148622, 497042, 105698,
                 355742, 436418, 439889, 150946,
                 268149, 628877, 326703, 556070)

    
    for nb_seq in range(nb_seq+1):
        with open(f"simulation_little_genome/16_chr/{origin}/res_{resolution}/{noisy}/sigma_{sigma}/{wavelets}{nb_seq}_theta_P_corr_inter_vector", 'rb') as f:
                        theta_corr = pickle.load(f)
        for i, theta in enumerate(list(theta_corr.keys())[:101]):
              print(i)
              for k in range(len(theta)-1):

                    axes[k%4, k//4].scatter(theta[k], theta[k+1], marker="+", color = color[nb_seq], label=f"ABC round {nb_seq}")

    # ax[1,0].set_xlabel("ABC round")
    # ax[1,0].set_ylabel("Wasserstein distance")
    # ax[1,0].set_title(r"Wasserstein distance between $p(\theta|C_{ref})$ and $\delta_{\theta_{ref}}$")
    # ax[1,0].legend(handles=[red_dot, blue_dot, green_dot, orange_dot])
    for k in range(len(theta_ref)-1):
            axes[k%4, k//4].scatter(theta_ref[k], theta_ref[k+1], marker = "+", s = 50, color = "black", label=f"ref")
            axes[k%4, k//4].set_xlabel(f"dim {k}")
            axes[k%4, k//4].set_ylabel(f"dim {k+1}")

          
    plt.tight_layout()
    plt.show()












