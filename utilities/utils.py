import itertools
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics.cluster import contingency_matrix
import csv
import sys
import numpy as np

from collections import Counter

rng = np.random.RandomState(1)

# 计算稳定性(deprecated)
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
            scores.append(adjusted_rand_score(l[in_both], k[in_both]))
    return np.mean(scores)

# 读取数据
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

# 写结果(deprecated)
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

    measure = adjusted_rand_score(true_label, out_label)

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
    sys.stdout.write('\r')
    sys.stdout.flush()


def normalization(data):
    data = np.array(data)
    _range = np.max(abs(data))
    return data / _range


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

# 画折线图
def plot_pic(title, filename, x_label, y_label, x, y):
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.title(title)
    y = normalization(np.array(y))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y)
    plt.savefig(filename)
    plt.clf()


def helper(x, y):
    x_ = []
    for i in range(len(x)):
        if x[i] not in x_:
            x_.append(i)
    y = y[np.array(x_)]
    return y

# 裁判法得到最终的最优聚类数目
def get_K_from_three_score(sh_y, ch_y, db_y):
    sh_y = np.array(sh_y)
    ch_y = np.array(ch_y)
    db_y = np.array(db_y)
    sh_k = np.argmax(sh_y)
    ch_k = np.argmax(ch_y)
    db_k = np.argmin(db_y)
    # 三者都一样
    if sh_k == ch_k and sh_k == db_k:
        return sh_k+2
    # 两个一样
    elif db_k == ch_k:
        return db_k+2
    # 两个一样或都不一样
    else:
        return sh_k+2

from matplotlib.pyplot import MultipleLocator

# 绘制最优聚类数目图。横轴为K，纵轴为评估指标数值
def plot_K(X, k_min, k_max, filename, est, title, stride=1):
    sh_y = []
    ch_y = []
    db_y = []

    for k in np.arange(k_min, k_max, stride):
        time_bar((k-k_min)/(k_max-k_min))
        sk = est(k)
        label = clustering(X, sk)
        try:
            sh_y.append(silhouette_score(sk.x_data, label))
        except:
            sh_y.append(0.0)
        try:
            ch_y.append(calinski_harabasz_score(sk.x_data, label))
        except:
            ch_y.append(0.0)
        try:
            db_y.append(davies_bouldin_score(sk.x_data, label))
        except:
            db_y.append(0.0)

    x = np.arange(k_min, k_max, stride)
    sh_y = normalization(sh_y)
    ch_y = normalization(ch_y)
    db_y = normalization(db_y)

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    
    x_major_locator=MultipleLocator(1)
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    k = get_K_from_three_score(sh_y, ch_y, db_y)
    # plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Score")
    # plt.xlim(1.5,9)
    si, = plt.plot(x, sh_y, label="Silhoutte score", linestyle="solid")
    ar, = plt.plot(x, ch_y, label="CH score", linestyle="dashdot")
    st, = plt.plot(x, db_y, label="DB score", linestyle="dashed")
    plt.legend([st, si, ar], ["DB score", "Silhoutte score", "CH score"])

    plt.savefig(filename+"_"+title+".png")
    plt.clf()


REPEAT_NUM = 5

# 得到算法的最优聚类数目
def get_final_K(X, k_min, k_max, est):
    K_num = []
    for i in range(REPEAT_NUM):
        sh_y = []
        ch_y = []
        db_y = []
        for k in np.arange(k_min, k_max):
            time_bar((k-k_min)/(k_max-k_min))
            sk = est(k)
            label = clustering(X, sk)
            try:
                sh_y.append(silhouette_score(sk.x_data, label))
            except:
                sh_y.append(0.0)
            try:
                ch_y.append(calinski_harabasz_score(sk.x_data, label))
            except:
                ch_y.append(0.0)
            try:
                db_y.append(davies_bouldin_score(sk.x_data, label))
            except:
                db_y.append(0.0)

        sh_y = normalization(sh_y)
        ch_y = normalization(ch_y)
        db_y = normalization(db_y)

        k = get_K_from_three_score(sh_y, ch_y, db_y)
        K_num.append(k)
    return K_num

# 得到ARI
def get_ari(X, k_min, k_max, est1, est2, filename, title):
    ARI_score = []
    for k in np.arange(k_min, k_max):
        time_bar((k-k_min)/(k_max-k_min))
        sk1 = est1(k)
        sk2 = est2(k)
        label1 = clustering(X, sk1)
        label2 = clustering(X, sk2)
        ARI = adjusted_rand_score(label1, label2)
        ARI_score.append(ARI)
    x = np.arange(k_min, k_max)

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

    plt.xlabel("K")
    plt.ylabel("Score")

    plt.plot(x, ARI_score, label="ARI score", linestyle="solid")


    plt.savefig(filename+"_"+title+".png")
    plt.clf()

# 计算混淆矩阵
def helper_matrix(label1,label2,K):
    c1 = Counter(label1)
    c2 = Counter(label2)

    tuple_c1 = c1.most_common(K)
    tuple_c2 = c2.most_common(K)
   

    l1 = np.array([tuple_c1[i][0] for i in range(len(tuple_c1))])
    l2 = np.array([tuple_c2[i][0] for i in range(len(tuple_c2))])
    l1_idx=[0 for i in range(K)]
    l2_idx=[0 for i in range(K)]
    for i in range(K):
        idx_c1 = np.where(label1 == l1[i])
        idx_c2 = np.where(label2 == l2[i])
        l1_idx[l1[i]]=idx_c1
        l2_idx[l2[i]]=idx_c2

    rank = np.arange(K)
    for i in range(K):
        max_common = -1
        max_common_idx = i
        for j in range(K):
            set1=l1_idx[i][0]
            set2=l2_idx[j][0]
            inter = np.intersect1d(set1, set2)
            # print("i:"+str(i)+" num "+str(set1))
            # print("j:"+str(j)+" num "+str(set2))
            # print(str(i)+" with "+str(j)+" :"+str(len(inter)))
            if len(inter) > max_common:
                max_common = len(inter)
                max_common_idx = j
            # max_common_idx=
        rank[i], rank[max_common_idx] = rank[max_common_idx], rank[i]
        l2_idx[i],l2_idx[max_common_idx]=l2_idx[max_common_idx],l2_idx[i]

    # l2=l2[rank]
    # l1_idx=np.array(l1_idx)
    # l2_idx=np.array(l2_idx)
    # l2_idx=l2_idx[rank]

    matrix = np.zeros((K, K),dtype=int)
    for i in range(K):
        for j in range(K):
            inter = np.intersect1d(l1_idx[i][0], l2_idx[j][0])
            matrix[i][j]=int(len(inter))


    return matrix

# 得到两个算法est1和est2的混淆矩阵
def get_matrix(X, k_min, k_max, est1, est2, filename, title):
    for k in np.arange(k_min, k_max):
        labels=list(range(k))
        time_bar((k-k_min)/(k_max-k_min))
        sk1 = est1(k)
        sk2 = est2(k)
        label1 = clustering(X, sk1)
        label2 = clustering(X, sk2)
        # ct=contingency_matrix(label1,label2)
        # cm = confusion_matrix(label1, label2, labels)
        cm = helper_matrix(label1,label2,k)

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
        plt.ylim(len(labels) - 0.5, -0.5)
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            # plt.text(j, i, cm[i, j])
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        plt.xlabel(sk1.name)
        plt.ylabel(sk2.name)
        plt.tight_layout()
        plt.savefig(filename+title+str(k)+".png")
        plt.clf()

# 绘制矩阵
def plot_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    