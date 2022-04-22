from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.utils import check_random_state
from sklearn.base import clone
import os
import csv
import sys
import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI
rng = np.random.RandomState(1)


def cluster_stability(X, est, n_iter=10, random_state=None, pt_only=False):
    labels = []
    indices = []
    X_copy = {}
    sample_indices = np.arange(0, X["pt_nc_img"].shape[0])
    for i in range(n_iter):
        # draw bootstrap samples, store indices
        # rng.shuffle(sample_indices)
        sample_indices = rng.randint(
            0, X["pt_nc_img"].shape[0], X["pt_nc_img"].shape[0])
        # X_bootstrap = X[sample_indices]
        X_copy["pt_nc_img"] = X["pt_nc_img"][sample_indices]
        X_copy["pt_nc_cov"] = X["pt_nc_cov"][sample_indices]
        X_copy["pt_ID"] = X["pt_ID"][sample_indices]
        X_copy["group"] = X["group"][sample_indices]
        # sample_indices = rng.randint(
        #     0, X["pt_nc_img"].shape[0], X["pt_nc_img"].shape[0])
        # store clustering outcome using original indices
        if(pt_only):
            tmp_indices = sample_indices[X_copy["group"] != 0]-X["len"]
            X_copy["true_label"] = X["true_label"][sample_indices[X_copy["group"] != 0]-X["len"]]
            est.fit(X_copy)
            indices.append(tmp_indices)
            relabel = -np.ones(X["len"], dtype=np.int)
            relabel[tmp_indices] = est.labels_
        else:
            indices.append(sample_indices)
            X_copy["true_label"] = X["true_label"][sample_indices]
            est.fit(X_copy)
            relabel = -np.ones(X["pt_nc_img"].shape[0], dtype=np.int)
            relabel[sample_indices] = est.labels_
        # est = clone(est)
        if hasattr(est, "random_state"):
            # randomize estimator if possible
            est.random_state = rng.randint(1e5)

        labels.append(relabel)
    scores = []
    for l, i in zip(labels, indices):
        for k, j in zip(labels, indices):
            # we also compute the diagonal which is a bit silly
            in_both = np.intersect1d(i, j)
            scores.append(ARI(l[in_both], k[in_both]))
    return np.mean(scores)


def read_data(filename, decimals=16):
    sys.stdout.write('\treading data...\n')
    feat_cov = None
    ID = None
    with open(filename) as f:
        data = list(csv.reader(f, delimiter='\t'))
        header = np.asarray(data[0])
        if 'GROUP' not in header:
            sys.stdout.write(
                'Error: group information not found. Please check csv header line for field "Group".\n')
            sys.exit(1)
        if 'ROI' not in header:
            sys.stdout.write(
                'Error: image features not found. Please check csv header line for field "IMG".\n')
            sys.exit(1)
        data = np.asarray(data[1:])

        group = (data[:, np.nonzero(header == 'GROUP')[0]]).astype(int)

        feat_img = (data[:, np.nonzero(header == 'ROI')[0]]).astype(np.float)
        if 'COVAR' in header:
            feat_cov = (data[:, np.nonzero(header == 'COVAR')[0]]).astype(
                np.int)
        if 'ID' in header:
            ID = (data[:, np.nonzero(header == 'ID')[0]]).astype(np.int)
            # ID = ID[group == 1]
        if 'SET' in header:
            set = (data[:, np.nonzero(header == 'SET')[0]])

    return feat_cov, np.around(feat_img, decimals=decimals), set, ID, group


def write_outputfile(output_file, ID, label, true_label, outcome_file, name):
    with open(output_file, 'w') as f:
        if ID is None:
            f.write('Cluster\n')
            for i in range(len(label)):
                f.write('%d\n' % (label[i]+1))
        else:
            f.write('ID,Cluster\n')
            for i in range(len(label)):
                f.write('%s,%d\n' % (ID[i][0], label[i]+1))

    with open(output_file) as f:
        out_label = np.asarray(list(csv.reader(f)))

    idx = np.nonzero(out_label[0] == "Cluster")[0]
    out_label = out_label[1:, idx].flatten().astype(np.int)

    measure = ARI(true_label, out_label)

    with open(outcome_file, 'a') as f:
        f.write(name+" :\n")
        f.write("ARI: " + str(measure)+'\n')


def get_data(filename, decimals=16):
    pt_nc_cov, pt_nc_img, set, ID, group = read_data(filename, decimals)

    # pt_nc_all = np.hstack((pt_nc_cov, pt_nc_img))

    return pt_nc_img, pt_nc_cov, set, ID, group


def clustering(X, est):
    est.fit(X)
    label = est.labels_
    return label


def time_bar(progress):
    sys.stdout.write('\r')
    prog_int = int(progress*50)
    sys.stdout.write('\t\t[%s%s] %.2f%%' %
                     ('='*prog_int, ' '*(50-prog_int), progress*100))
    sys.stdout.flush()


def plot_pic(title,filename,x_label,y_label,x,y):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y)
    plt.savefig(filename)
    plt.clf()

def eval_K(X, k_min, k_max, filename, est, title, pt_only=False, stride=1, get_k_num=False):

    silhouette_y = []
    ch_y = []
    db_y = []
    k_num_y = []

    for k in np.arange(k_min, k_max, stride):
        time_bar((k-k_min)/(k_max-k_min))
        sk = est(k)
        label = clustering(X, sk)
        if get_k_num:
            if sk.k_num == 1 or sk.k_num == len(X["pt_nc_img"]):
                silhouette_y.append(np.nan)
                ch_y.append(np.nan)
                db_y.append(np.nan)
                k_num_y.append(1)
                continue
            k_num_y.append(sk.k_num)
        # silhouette_score requires more than 1 cluster labels.
        silhouette_y.append(silhouette_score(sk.x_data, label))
        ch_y.append(calinski_harabasz_score(sk.x_data, label))
        db_y.append(davies_bouldin_score(sk.x_data, label))

    x = np.arange(k_min, k_max, stride)
    
   
    if(get_k_num):
        x_label="α"
        plot_pic("Number of Clusters",filename+"_k.png",x_label,"Number of Clusters",x,k_num_y)

    else:
       x_label="number of clusters"
    plot_pic("Silhoutte Score",filename+"_sh.png",x_label,"Silhoutte Score",x,silhouette_y)

    plot_pic("Calinski Harabasz Score",filename+"_ch.png",x_label,"Calinski Harabasz Score",x,ch_y)

    plot_pic("Davies Bouldin Score",filename+"_db.png",x_label,"Davies Bouldin Score",x,db_y)


