import numpy as np
import itertools
import warnings
from scipy import sparse


def normalize(counts, lengths, cnv):
    if sparse.issparse(counts):
        bias_es = np.ones(counts.data.shape)
    else:
        bias_es = np.ones(counts.shape)

    print("")
    print("Estimating CNV-effects.")
    max_iter = 20
    min_iter = 5
    eps = 1e-1
    old_bias_es = None
    for it in range(max_iter):
        print("Iteration %d:" % it)

        print("1. Estimating mapping...")
        mapping = get_mapping(counts, lengths, bias_es, verbose=True)

        print("2. Estimating expected...")
        c_expected = get_expected(counts, lengths, bias_es, mapping=mapping)
        if np.any(c_expected < 0):
            raise ValueError("Found expected counts below 0.")

        print("3. Estimating CNV biases...")
        bias_es = estimate_bias(counts, cnv, c_expected, lengths,
                                mapping=mapping,
                                normalization_type="independant")

        if np.any(bias_es < 0):
            raise ValueError("Found a bias below 0.")
        
        if old_bias_es is not None:
            error = np.nanmax(np.abs(bias_es - old_bias_es))
        if it >= min_iter and old_bias_es is not None and error < eps:
            print("")
            print("Converged in %d iteration:" % it)
            break
        elif old_bias_es is not None:
            print("Error %0.2f" % error)

        old_bias_es = bias_es.copy()

    return bias_es


def get_genomic_distances(lengths, counts=None):
    """
    Get genomic distances
    """

    return _get_genomic_distances_dense(lengths)




def _get_intra_mask_dense(lengths):
    """
    Returns a mask for intrachromosomal interactions

    Parameters
    ----------
    lengths : ndarray, (n, )
        lengths of the chromosomes

    Returns
    -------
    mask : ndarray (m, m)
        boolean mask
    """
    # Copy the lengths in order not to modify the original matrix
    mask = np.zeros((lengths.sum(), lengths.sum()))
    begin = 0
    for end in lengths.cumsum():
        mask[begin:end, begin:end] = 1
        begin = end
    return mask.astype(bool)

def _get_inter_mask_dense(lengths):
    """
    Returns a mask for interchromosomal interactions

    Parameters
    ----------
    lengths : ndarray, (n, )
        lengths of the chromosomes

    Returns
    -------
    mask : ndarray of dtype boolean
        boolean mask
    """
    
    mask = np.ones((lengths.sum(), lengths.sum()))
    begin = 0
    for end in lengths.cumsum():
        mask[begin:end, begin:end] = 0
        begin = end
    return mask.astype(bool)

def _get_genomic_distances_dense(lengths):
    """
    Returns a matrix of the genomic distances

    Inter chromosomal interactions are set to -1

    Parameters
    ----------
    lengths : ndarray (n, )
        lengths of the chromosomes

    Returns
    -------
    dis: ndarray (n, n), dtype: int
        returns the genomic distance matrix, with -1 for inter chromosomal
        interactions
    """
    inter_mask = _get_inter_mask_dense(lengths)
    
    n = lengths.sum()

    dis = np.concatenate([np.concatenate([np.arange(i, 0, -1),
                                          np.arange(n - i)])
                          for i in range(n)])
    
    dis = dis.reshape((n, n))
    dis[inter_mask] = -1
    
    return dis.astype(int)


def get_expected(counts, lengths, bs=None, mapping=None):

    if mapping is None:
        genomic_distances, expected_counts = get_mapping(counts, lengths, bs)
    else:
        if mapping.shape[0] == 2:
            #ici
            genomic_distances, expected_counts = mapping
        else:
            
            genomic_distances, expected_counts = mapping.T

    gdis = get_genomic_distances(lengths, counts)
    if sparse.issparse(counts):
        c_expected = np.zeros(counts.data.shape)
    else:
        #ici
        c_expected = np.zeros(counts.shape)

    for dis in np.unique(gdis):
        c_expected[gdis == dis] = expected_counts[genomic_distances == dis]
    return c_expected


def create_missing_distances_mask(counts, lengths):
    mask = (np.array(counts.sum(axis=0)).flatten() +
            np.array(counts.sum(axis=1)).flatten()) == 0
    missing_idx = np.where(mask)[0]
    n = counts.shape[0]

    row_idxs = np.concatenate(
        [np.tile(np.arange(n), len(missing_idx)),
         np.repeat(missing_idx, n)])
    col_idxs = np.concatenate(
        [np.repeat(missing_idx, n),
         np.tile(np.arange(n), len(missing_idx))])

    missing_distances_mask = sparse.coo_matrix(
        (np.ones(col_idxs.shape), (row_idxs, col_idxs)), shape=(n, n),
        dtype=np.bool)
    missing_distances_mask.sum_duplicates()
    return sparse.triu(missing_distances_mask, k=1)


def identify_missing_distances(counts, lengths):

    # Find rows that are completely empty
    mask = (np.array(counts.sum(axis=0)).flatten() +
            np.array(counts.sum(axis=1)).flatten()) == 0

    if not np.any(mask):
        gdis = np.arange(-1, lengths.max())
        return gdis, np.zeros(gdis.shape)

    chromosomes = np.array(
        [i for i, l in enumerate(lengths) for j in range(l)])
    l = np.array([i for i in lengths for j in range(i)])
    l_ = np.array(
        [lengths[:i].sum() for i, t in enumerate(lengths) for j in range(t)])
    l_[:lengths[0]] = 0
    missing_loci = np.where(mask)[0] - l_[mask]
    dis = np.concatenate(
        [missing_loci, l[mask] - missing_loci - 1]).flatten()
    dis = dis[dis > 0]
    gdis = np.arange(1, lengths.max() + 1)
    num = (gdis[:, np.newaxis] <= dis).sum(axis=1)

    # Now remove distances counted twice
    c = chromosomes[mask]
    d = np.abs(missing_loci[:, np.newaxis] - missing_loci)
    d[c[:, np.newaxis] != c] = 0
    dis, count = np.unique(d, return_counts=True)
    count /= 2
    num[dis[dis != 0]-1] -= count[dis != 0]

    n = np.unique(chromosomes[mask], return_counts=True)[1]
    num_dup = np.triu(n[:, np.newaxis] * n, 1).sum()
    num = np.concatenate(
        [[(lengths.sum() - l[mask]).sum() - num_dup], [0], num])
    return np.arange(-1, lengths.max() + 1), num


def get_mapping(counts, lengths, bs=None, smoothed=False, verbose=False):
    if verbose:
        print("Computing relationship genomic distance & expected counts")

    gdis, means = _get_mapping_dense(counts, lengths, bs)
    
    if not smoothed:
        return np.array([gdis, means])

    if verbose:
        print("Fitting Isotonic Regression")

    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
    if gdis.min() > 0:
        y = np.array(means).flatten()
        x = np.arange(y.shape[0])
    elif gdis.min() == 0:
        y = np.array(means)[1:].flatten()
        x = np.arange(y.shape[0])
    else:
        y = np.array(means)[2:].flatten()
        x = np.arange(y.shape[0])

    mask = np.invert(np.isnan(y) | np.isinf(y) | (y == 0))
    ir.fit(
        x[mask],
        y[mask])
    means_fitted = ir.transform(x)
    if gdis.min() < 0:
        expected_counts = np.concatenate([[means[0], 0], means_fitted])
    elif gdis.min() == 0:
        expected_counts = np.concatenate([[0], means_fitted])
    else:
        expected_counts = means_fitted

    return np.array([gdis, expected_counts])


def _get_mapping_dense(counts, lengths, bs):
    if bs is None:
        bs = 1
    gdis = _get_genomic_distances_dense(lengths)
    
    means = []
    
    for dis in np.arange(gdis.min(), gdis.max()+1):
        
        c = (counts / bs)[dis == gdis]
       
        means.append(np.nanmean(c))
    
    return np.arange(gdis.min(), gdis.max()+1), means


def _get_mapping_sparse(counts, lengths, bs):
    if bs is None:
        bs = 1
    gdis = get_genomic_distances(lengths, counts)
    means = []
    # Compute the number of missing rows/col for each chromose
    gdis_, num_ = identify_missing_distances(counts, lengths)

    for dis in np.arange(gdis.min(), max(lengths)):
        c = (counts.data / bs)[gdis == dis]
        if dis >= 0:
            t = lengths - dis
            num = (t[t > 0]).sum() - num_[gdis_ == dis]
        else:
            num = ((lengths).sum()**2 -
                   ((lengths)**2).sum()) / 2 - num_[gdis_ == dis]

        if num[0] < 0:
            raise ValueError("There cannot be less than 0 interactions")

        # XXX We have non nan elements that still should not be included in
        # the mean. How can we count those ?
        # means.append(c.sum() / num[0])
        if num[0] - np.isnan(c).sum() == 0:
            means.append(np.nan)
        else:
            means.append(np.nansum(c) / (num[0] - np.isnan(c).sum()))

    return np.arange(gdis.min(), max(lengths)), means


def estimate_bias(counts, cnv, c_expected, lengths, mapping=None,
                  normalization_type="independant"):
    if sparse.issparse(counts):
        return _estimate_bias_sparse(
            counts, cnv, c_expected, lengths,
            mapping, normalization_type=normalization_type)
    else:
        return _estimate_bias_dense(counts, cnv, c_expected, lengths,
                                    normalization_type=normalization_type)


def get_intra_mask(lengths, counts=None):
    import iced
    if counts is not None and sparse.issparse(counts):
        return _get_intra_mask_sparse(lengths, counts)
    else:
        return iced.utils.get_intra_mask(lengths)


def _get_intra_mask_sparse(lengths, counts):
    """
    """
    chr_id = np.array([i for i, l in enumerate(lengths) for _ in range(l)])
    mask = np.ones(counts.col.shape, dtype=bool) * True
    mask[chr_id[counts.col] != chr_id[counts.row]] = False
    return mask


def _estimate_bias_dense_joint(counts, cnv, c_expected, lengths):
    bias = np.ones(counts.shape)

    for cnv_i in np.unique(cnv):
        for cnv_j in np.unique(cnv):
            if cnv_i > cnv_j:
                continue
            mask = ((cnv == cnv_i)[:, np.newaxis] * (cnv == cnv_j) |
                    (cnv == cnv_j)[:, np.newaxis] * (cnv == cnv_i))

            c = counts[mask]
            m = np.invert(np.isnan(c))
            c = c[m]
            c_exp = c_expected[mask][m]
            bias[mask] = (c * c_exp).sum() / (c_exp**2).sum()
    return bias


def _estimate_bias_dense(counts, cnv, c_expected, lengths,
                         normalization_type="independant"):
    if normalization_type not in ["independant", "chrjoint", "joint"]:
        raise ValueError("unknown type of normalization")

    bias = np.ones(counts.shape)
    intra = get_intra_mask(lengths)

    # Start with the inter
    for cnv_i in np.unique(cnv):
        for cnv_j in np.unique(cnv):
            if cnv_i > cnv_j:
                continue
            mask = ((cnv == cnv_i)[:, np.newaxis] * (cnv == cnv_j) |
                    (cnv == cnv_j)[:, np.newaxis] * (cnv == cnv_i))

            c = counts[mask]
            c_intra = intra[mask]
            m = np.invert(np.isnan(c)) & np.invert(c_intra)
            c = c[m]
            c_exp = c_expected[mask][m]
            bias[mask] = (c * c_exp).sum() / (c_exp**2).sum()
    bias[intra] = 1

    # Now do the intra. For the intra, we have to do piece by piece (XXX check
    # this).
    if normalization_type == "chrjoint":
        begin, end = 0, 0
        for i, length in enumerate(lengths):
            end += length
            scnv = cnv[begin:end]
            bias[begin:end, begin:end] = _estimate_bias_dense_joint(
                counts[begin:end, begin:end], scnv,
                c_expected[begin:end, begin:end], np.array([length]))
            begin = end
    elif normalization_type == "joint":
        counts[np.invert(intra)] = np.nan
        bias_ = _estimate_bias_dense_joint(counts, cnv, c_expected, lengths)
        bias[intra] = bias_[intra]
    else:
        breakpoints = np.where((cnv[1:] - cnv[:-1]).astype(bool))[0] + 1
        breakpoints = np.array(list(breakpoints) + list(lengths.cumsum()))
        breakpoints = np.unique(breakpoints)
        breakpoints.sort()

        begin_i, end_i = 0, 0
        for i, length_i in enumerate(breakpoints):
            end_i = length_i
            begin_j, end_j = 0, 0
            for j, length_j in enumerate(breakpoints):
                end_j = length_j
                if np.any(intra[begin_i:end_i, begin_j:end_j]):
                    c = counts[begin_i:end_i, begin_j:end_j]
                    m = (np.invert(np.isnan(c)) &
                        intra[begin_i:end_i, begin_j:end_j])
                    c_exp = c_expected[begin_i:end_i, begin_j:end_j][m]
                    c = c[m]
                    b = (c * c_exp).sum() / (c_exp**2).sum()
                    bias[begin_i:end_i,
                         begin_j:end_j] = b
                begin_j = end_j
            begin_i = end_i
    return bias


def _num_each_gdis(rows, cols, lengths):
    """
    Compute the number of elements that are in each genomic distances for the
    rows and cols provided

    Returns
    -------
    gdis, counts
    """
    chr_id = np.array([i for i, l in enumerate(lengths) for _ in range(l)])

    # All possible gdis.
    gdis = np.arange(-1, lengths.max())
    num = np.zeros(gdis.shape)
    # We are going to loop over the rows
    for row in rows:
        # We need to consider only the upper-triangular matrix, which is where
        # cols are larger than rows (XXX check this)
        dis = row - cols[np.newaxis]
        m = dis <= 0
        dis = np.abs(dis)
        # Find out which ones are inter chromosomal
        dis[chr_id[row] != chr_id[cols][np.newaxis]] = -1
        un, val = np.unique(dis[m], return_counts=True)
        num[un+1] += val
    # Removing diag
    num[1] = 0
    return gdis[num != 0], num[num != 0]


def _estimate_bias_sparse(counts, cnv, c_expected, lengths, mapping,
                          normalization_type="independant"):
    if normalization_type != "independant":
        raise NotImplementedError
    gdis, mapping = mapping

    bias = np.ones(counts.data.shape)
    intra = get_intra_mask(lengths, counts=counts)

    # Identify rows that are completely missing from the data. These should
    # not be taken in account when estimating the bias.
    mask = (np.array(counts.sum(axis=0)).flatten() +
            np.array(counts.sum(axis=1)).flatten()) == 0
    mask_idx = np.where(mask)[0]

    missing_loci = (mask_idx[:, np.newaxis] <= lengths.cumsum()).sum(axis=0)
    missing_loci = np.concatenate(
        [[missing_loci[0]], missing_loci[1:] - missing_loci[:-1]])

    # Start with the inter
    for cnv_i in np.unique(cnv):
        for cnv_j in np.unique(cnv):
            idx = np.where(
                ((cnv[counts.row] == cnv_i) &
                 (cnv[counts.col] == cnv_j)) |
                ((cnv[counts.row] == cnv_j) &
                 (cnv[counts.col] == cnv_i)))[0]
            c_intra = intra[idx]
            if not np.any(np.invert(c_intra)):
                continue

            c = counts.data[idx][np.invert(c_intra)]
            c_exp = c_expected[idx][np.invert(c_intra)]

            # XXX need to take in account 0 in the denominator.
            rows = np.array(
                [i for i in np.where(cnv == cnv_i)[0] if i not in mask_idx])
            cols = np.array(
                [i for i in np.where(cnv == cnv_j)[0] if i not in mask_idx])
            gdis_, num = _num_each_gdis(rows, cols, lengths)

            # We are only interested in the inters here
            denominator = (num[gdis_ == -1] * (mapping[gdis == -1] ** 2)).sum()
            bias[idx] = (c * c_exp).sum() / denominator

    bias[intra] = 1

    # Now do the intra. For the intra, we have to do piece by piece (XXX check
    # this).
    breakpoints = np.where((cnv[1:] - cnv[:-1]).astype(bool))[0] + 1
    breakpoints = np.array(list(breakpoints) + list(lengths.cumsum()))
    breakpoints = np.unique(breakpoints)
    breakpoints.sort()

    begin_i = 0
    for i, end_i in enumerate(breakpoints):
        begin_j, end_j = 0, 0
        rows = np.array(
            [i for i in np.arange(begin_i, end_i) if i not in mask_idx])

        for j, end_j in enumerate(breakpoints):
            idx = np.where((counts.row >= begin_i) & (counts.row < end_i) &
                           (counts.col >= begin_j) & (counts.col < end_j))

            if np.any(intra[idx]):
                cols = np.array(
                    [i for i in np.arange(begin_j, end_j)
                     if i not in mask_idx])

                gdis_, num = _num_each_gdis(rows, cols, lengths)

                c_exp = c_expected[idx][intra[idx]]
                denominator = sum(
                    [(n * (mapping[gdis == g] ** 2)).sum()
                     for g, n in zip(gdis_, num) if (g not in [-1, 0])])
                num = (counts.data[idx][intra[idx]] * c_exp).sum()
                bias[idx] = num / denominator
            begin_j = end_j
        begin_i = end_i
    return bias


def _estimate_bias_sparse_joint(counts, cnv, c_expected, lengths, mapping):
    gdis, mapping = mapping

    bias = np.ones(counts.data.shape)
    intra = get_intra_mask(lengths, counts=counts)

    # Identify rows that are completely missing from the data. These should
    # not be taken in account when estimating the bias.
    mask = (np.array(counts.sum(axis=0)).flatten() +
            np.array(counts.sum(axis=1)).flatten()) == 0
    mask_idx = np.where(mask)[0]

    missing_loci = (mask_idx[:, np.newaxis] <= lengths.cumsum()).sum(axis=0)
    missing_loci = np.concatenate(
        [[missing_loci[0]], missing_loci[1:] - missing_loci[:-1]])

    # Start with the inter
    for cnv_i in np.unique(cnv):
        if len(lengths) == 1:
            break
        for cnv_j in np.unique(cnv):
            if cnv_i > cnv_j:
                continue
            idx = np.where(
                ((cnv[counts.row] == cnv_i) &
                 (cnv[counts.col] == cnv_j)) |
                ((cnv[counts.row] == cnv_j) &
                 (cnv[counts.col] == cnv_i)))[0]
            if not np.any(idx):
                continue

            c = counts.data[idx] #[np.invert(c_intra)]
            c_exp = c_expected[idx] #[np.invert(c_intra)]
            
            # XXX need to take in account 0 in the denominator.
            rows = np.array(
                [i for i in np.unique(counts.row[idx]) if i not in mask_idx])
            cols = np.array(
                [i for i in np.unique(counts.col[idx]) if i not in mask_idx])

            # this is more complicated than this... We need to remove the
            # intersection of those in which cnv_i and cnv_j are not matched.
            gdis_, num = _num_each_gdis(rows, cols, lengths)

            # We are only interested in the inters here
            #denominator = (num[gdis_ == -1] * (mapping[gdis == -1] ** 2)).sum()
            # denominator = (num * mapping**2).sum()
            denominator = sum(
                [(n * (mapping[gdis == g] ** 2)).sum()
                    for g, n in zip(gdis_, num)])
            bias[idx] = (c * c_exp).sum() / denominator
    return bias
    #bias[intra] = 1

    # Now do the intra. For the intra, we have to do piece by piece (XXX check
    # this).
    breakpoints = np.where((cnv[1:] - cnv[:-1]).astype(bool))[0] + 1
    breakpoints = np.array(list(breakpoints) + list(lengths.cumsum()))
    breakpoints = np.unique(breakpoints)
    breakpoints.sort()

    begin_i = 0
    for i, end_i in enumerate(breakpoints):
        begin_j, end_j = 0, 0
        rows = np.array(
            [i for i in np.arange(begin_i, end_i) if i not in mask_idx])

        for j, end_j in enumerate(breakpoints):
            idx = np.where((counts.row >= begin_i) & (counts.row < end_i) &
                           (counts.col >= begin_j) & (counts.col < end_j))

            if np.any(intra[idx]):
                cols = np.array(
                    [i for i in np.arange(begin_j, end_j)
                     if i not in mask_idx])

                gdis_, num = _num_each_gdis(rows, cols, lengths)

                c_exp = c_expected[idx][intra[idx]]
                denominator = sum(
                    [(n * (mapping[gdis == g] ** 2)).sum()
                     for g, n in zip(gdis_, num) if (g not in [-1, 0])])
                num = (counts.data[idx][intra[idx]] * c_exp).sum()
                bias[idx] = num / denominator
            begin_j = end_j
        begin_i = end_i
    return bias


def initialize_biases(counts, cnv, lengths):
    if sparse.issparse(counts):
        return _initialize_biases_sparse(counts, cnv, lengths)
    else:
        return _initialize_biases_dense(counts, cnv, lengths)


def _initialize_biases_dense(counts, cnv, lengths):
    biases = np.ones(counts.shape)

    # For mapping_base, select the region with the most data.
    c = counts.copy()
    b, num = np.unique(cnv, return_counts=True)
    b = b[num.argmax()]
    blocks = (cnv == b)[:, np.newaxis] * (cnv == b)
    blocks = blocks + blocks.T
    c[blocks] = np.nan

    mapping_base = get_mapping(c, lengths, biases, smoothed=False)
    pairs_cnv = [
        (i, j) for i, j in itertools.product(np.unique(cnv), np.unique(cnv))
        if i >= j]
    pairs_cnv = np.array(pairs_cnv)

    # XXX Need to estimate biases for trans as well
    print("Estimate %d elements" % len(pairs_cnv))
    for b in pairs_cnv:
        blocks = (cnv == b[0])[:, np.newaxis] * (cnv == b[1])
        blocks = blocks + blocks.T
        c = counts.copy()
        c[blocks] = np.nan
        mapping = get_mapping(c, lengths, biases, smoothed=False)

        weights = np.arange(lengths.max(), 0, -1)[:len(mapping[1])-1]

        if np.all(np.isnan(mapping_base[1][1:])):
            # Compute this using inter:
            a = mapping_base[1][0]**2 / mapping[1][0]**2
        else:
            a = weights * mapping_base[1][1:] * mapping[1][1:]
            a = np.nansum(a) / np.sum(
                (weights*mapping[1][1:]**2)[np.invert(np.isnan(a))])
        biases[blocks] = 1./a

    if np.any(np.isnan(biases) | np.isinf(biases)):
        warnings.warn("CNV effect may be poorly estimated")
        biases[np.isnan(biases) | np.isinf(biases)] = 1
    return biases


def _initialize_biases_sparse(counts, cnv, lengths):
    raise NotImplementedError


def smooth(counts, lengths,  sigma=1):
    from scipy.ndimage import gaussian_filter
    begin_i, end_i = 0, 0
    for i, l_i in enumerate(lengths):
        end_i = l_i
        begin_j, end_j = 0, 0
        for j, l_j in enumerate(lengths):
            end_j = l_j
            sc = counts[begin_i:end_i, begin_j:end_j]
            counts[begin_i:end_i, begin_j:end_j] = gaussian_filter(sc, sigma)
            begin_j = l_j
        begin_i = end_i
    return counts


def average(counts, lengths):
    begin_i, end_i = 0, 0
    for i, l_i in enumerate(lengths):
        end_i = l_i
        begin_j, end_j = 0, 0
        for j, l_j in enumerate(lengths):
            end_j = l_j
            sc = counts[begin_i:end_i, begin_j:end_j]
            counts[begin_i:end_i, begin_j:end_j] = np.nanmean(sc)
            begin_j = l_j
        begin_i = end_i
    return counts
