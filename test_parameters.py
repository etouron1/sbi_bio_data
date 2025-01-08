from simulators import chr_seq_yeast_duan, get_num_beads_and_start

def get_compaction_ratio(chr_seq, sep, bead_radius):
    chr_num_bead, chr_start_bead, nbead = get_num_beads_and_start(chr_seq, sep)
    for chr in chr_seq.keys():
        n = chr_num_bead[chr]
        W = bead_radius
        L = chr_seq[chr]
        C = L/(n*W)
        print(chr, C)

get_compaction_ratio(chr_seq_yeast_duan, 3200, 30)

def get_persistence_length(k_bend, bead_radius):
    print(0.9*bead_radius)
    print(1.5*bead_radius)

get_persistence_length(0.2, 15)

