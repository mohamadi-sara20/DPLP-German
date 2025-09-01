from dplp_parser.data import Data
import gzip
from pickle import load, dump
from dplp_parser.util import *
from numpy import linalg as LA
import numpy as np
from multiclass_svm import MulticlassSVM
from scipy.sparse import lil_matrix, coo_matrix
import time
import random
import sklearn
import cProfile
from dplp_parser.evalparser import evalparser
from sklearn.svm import LinearSVC
import re
import os
import argparse
import shutil
import random

config = {
    'fvocab': "vocab.pickle.gz",
    'fdpvocab': "word-dict.pickle.gz",
    'fbcvocab': "bcvocab.pickle.gz"
}

class ProjmatLearn(object):
    def __init__(self, basepath, clf=None, topn=10000):
        """ Initialization

        :type vocab:
        :param vocab:

        :type idxrelamap:
        :param idxrelamap:

        :type clf:
        :param clf:
        """
        self.basepath = basepath
        self.topn = topn

        if clf is None:
            self.clf = MulticlassSVM(C=1, tol=0.01, max_iter=1000, random_state=0, verbose=0)
        else:
            self.clf = clf

        with gzip.open('{}/{}'.format(basepath, config['fbcvocab']), 'rb') as fin:
            bcvocab = load(fin)
        print("bcvocab length: " + str(len(bcvocab)))

        self.data = Data(bcvocab=bcvocab,
                        withdp=False  # it's being created before any projection matrix
                        )
        self.data.builddata(basepath + '/training')
        # tmp_data.buildvocab(topn=topn)
        self.data.loadvocab('{}/{}'.format(basepath, config['fvocab']))
        print("vocab length: " + str(len(self.data.vocab)))

        with gzip.open('{}/{}'.format(basepath, config['fdpvocab']), 'rb') as fin:
            self.data.dpvocab = load(fin)
        print("dpvocab length: " + str(len(self.data.dpvocab)))

        print('Finish initializing ParsingModel')

    def createdata_raw(self):
        data = self.data
        # custom build matrix
        nR, nC = len(data.samplelist), len(data.dpvocab) * 3
        nC += len(data.vocab)
        rows, cols, vals, L = [], [], [], []
        for (sidx, sample) in enumerate(data.samplelist):
            label = action2label(data.actionlist[sidx])
            lidx = data.labelmap[label]
            vec = self.vectorize_raw(sample, data)
            L.append(lidx)
            vec = lil_matrix(vec)
            r, c = vec.nonzero()
            r, c = list(r), list(c)
            for (row, col) in zip(r, c):
                val = vec[row, col]
                rows.append(sidx)
                cols.append(col)
                vals.append(val)
                #M.append((sidx, col, val))
        return coo_matrix((vals, (rows, cols)), shape=(nR, nC)), L

    def vectorize_raw(self, features, data):
        vocab, dpvocab = data.vocab, data.dpvocab

        vec = lil_matrix((1, len(vocab)))
        ndpvocab = len(dpvocab)
        dpvec1 = lil_matrix((1, ndpvocab))
        dpvec2 = lil_matrix((1, ndpvocab))
        dpvec3 = lil_matrix((1, ndpvocab))

        for feat in features:
            try:
                fidx = vocab[feat]
                vec[0, fidx] += 1.0
            except KeyError:
                pass
            if (feat[0] == 'DisRep'):
                tag, word = feat[1], feat[2]
                try:
                    widx = dpvocab[word]
                    if tag == 'Top1Span':
                        dpvec1[0, widx] += 1.0
                    elif tag == 'Top2Span':
                        dpvec2[0, widx] += 1.0
                    elif tag == 'FirstSpan':
                        dpvec3[0, widx] += 1.0
                    else:
                        raise ValueError("Error")
                except KeyError:
                    widx = None

        vec_dense = lil_matrix((1, 3 * ndpvocab))
        vec_dense[0, 0:ndpvocab] = dpvec1
        vec_dense[0, ndpvocab:(2 * ndpvocab)] = dpvec2
        vec_dense[0, (2 * ndpvocab):(3 * ndpvocab)] = dpvec3

        vec_dense = hstack([vec, vec_dense])
        return vec_dense

    def project_samples(self, raw_samples, data):
        nv = len(data.vocab)
        v = raw_samples[:,:nv]
        ndp = len(data.dpvocab)
        v1 = raw_samples[:,nv:nv+ndp].dot(data.projmat)
        v2 = raw_samples[:,nv+ndp:nv+ndp*2].dot(data.projmat)
        v3 = raw_samples[:,nv+ndp*2:].dot(data.projmat)
        v1 = sklearn.preprocessing.normalize(v1, norm="l1")
        v2 = sklearn.preprocessing.normalize(v2, norm="l1")
        v3 = sklearn.preprocessing.normalize(v3, norm="l1")
        v  = sklearn.preprocessing.normalize(v, norm="l1")
        return np.concatenate((v,v1,v2,v3), axis=1)

    # def createdata_with_projmat(self, data):
    #     nR = len(data.samplelist)
    #     nC = data.projmat.shape[1]
    #     if not self.free_form:
    #         # each sample is a concatenation of three vectors multiplied by projmat
    #         nC = 3 * nC
    #
    #     rows, cols, vals, L = [], [], [], []
    #     for (sidx, sample) in enumerate(data.samplelist):
    #         label = action2label(data.actionlist[sidx])
    #         lidx = data.labelmap[label]
    #         vec = self.vectorize_with_projmat(sample, data)
    #         L.append(lidx)
    #         vec = lil_matrix(vec)
    #         r, c = vec.nonzero()
    #         r, c = list(r), list(c)
    #         for (row, col) in zip(r, c):
    #             val = vec[row, col]
    #             rows.append(sidx)
    #             cols.append(col)
    #             vals.append(val)
    #
    #     M = coo_matrix((vals, (rows, cols)), shape=(nR, nC))
    #     return M, L
    #
    # def vectorize_with_projmat(self, features, data):
    #     vocab, dpvocab , projmat = data.vocab, data.dpvocab, data.projmat
    #
    #     vec = lil_matrix((1, len(vocab)))
    #     ndpvocab = len(dpvocab)
    #     dpvec1 = lil_matrix((1, ndpvocab))
    #     dpvec2 = lil_matrix((1, ndpvocab))
    #     dpvec3 = lil_matrix((1, ndpvocab))
    #
    #     for feat in features:
    #         try:
    #             if self.free_form:
    #                 fidx = vocab[feat]
    #                 vec[0, fidx] += 1.0
    #         except KeyError:
    #             pass
    #         if (feat[0] == 'DisRep'):
    #             tag, word = feat[1], feat[2]
    #             try:
    #                 widx = dpvocab[word]
    #                 if tag == 'Top1Span':
    #                     dpvec1[0, widx] += 1.0
    #                 elif tag == 'Top2Span':
    #                     dpvec2[0, widx] += 1.0
    #                 elif tag == 'FirstSpan':
    #                     dpvec3[0, widx] += 1.0
    #                 else:
    #                     raise ValueError("Error")
    #             except KeyError:
    #                 widx = None
    #
    #     # Projection
    #     if self.free_form:
    #         vec_dense = hstack([vec, dpvec1, dpvec2, dpvec3])
    #         vec = normalize(vec_dense.dot(projmat))
    #     else:
    #         nlat = projmat.shape[1]
    #         vec_dense = lil_matrix((1, 3*nlat))
    #         v1 = dpvec1.dot(projmat)
    #         v2 = dpvec2.dot(projmat)
    #         v3 = dpvec3.dot(projmat)
    #         vec_dense[0, 0:nlat] = normalize(v1)
    #         vec_dense[0, nlat:(2*nlat)] = normalize(v2)
    #         vec_dense[0, (2*nlat):(3*nlat)] = normalize(v3)
    #         vec = vec_dense
    #
    #     return vec

    def create_identity_matrix(self, outpath):
        nfeatures = len(self.data.dpvocab)
        print('Features (dpvocab size): ' + str(nfeatures))

        vals = []
        for i in range(0,nfeatures):
            vals.append(1)

        A = coo_matrix((vals, (range(0,nfeatures), range(0,nfeatures))), shape=(nfeatures, nfeatures))
        print("Identity matrix created with shape: " + str(A.shape))

        with gzip.open(outpath, 'wb') as fout:
            dump(A, fout)

    def create_zero_matrix(self, outpath):
        nfeatures = len(self.data.dpvocab)

        A = np.zeros((1, nfeatures))

        with gzip.open(outpath, 'wb') as fout:
            dump(A, fout)

    def train(self, modelpath, outpath, K, tau, C, n_samples=-1, iterations=200, eps=0.0001):
        """ Perform batch-learning on parsing model
        """
        print('Training ...')
        nfeatures = len(self.data.dpvocab)
        A_t_1 = np.random.uniform(0, 1, (nfeatures, K))

        self.data.withdp = True
        self.data.projmat = A_t_1

        print('{} - re-generating raw training data'.format(time.strftime("%H:%M:%S", time.localtime())))
        trn_raw_data, trnL = self.createdata_raw()
        trn_raw_data = trn_raw_data.toarray()
        n_total_sample = trn_raw_data.shape[0]
        print('Total number of samples = {}'.format(n_total_sample))
        if n_samples >= n_total_sample:
            n_samples = -1
        if n_samples > 0:
            sample_indices = random.sample(range(0, n_total_sample), n_samples)
            t = [trn_raw_data[i] for i in sample_indices]
            l = [trnL[i] for i in sample_indices]
            trn_raw_data_filtered = np.array(t)
            trnL_filtered = np.array(l)
        else:
            trn_raw_data_filtered = trn_raw_data
            trnL_filtered = trnL

        lclf = MulticlassSVM(C=C, tol=0.01, max_iter=1000, random_state=0, verbose=0)
        #svm = LinearSVC(C=1.0, penalty='l1',loss='squared_hinge', dual=False, tol=1e-7)
        for t in range(1, iterations + 1):
            print('{} - re-generating projected training data'.format(time.strftime("%H:%M:%S", time.localtime())))
            self.data.projmat = A_t_1 # equal to A_t at this point
            trnMArr = self.project_samples(trn_raw_data_filtered, self.data)

            print('{} - learnig svm'.format(time.strftime("%H:%M:%S", time.localtime())))
            lclf.fit(trnMArr, trnL_filtered)
            # svm.fit(trnMArr, trnL_filtered)

            print('{} - solving projmat - {}'.format(time.strftime("%H:%M:%S", time.localtime()), outpath))
            A_t = self.solve_projmat_tiles1(lclf, A_t_1, t, tau, trn_raw_data_filtered, trnL_filtered)

            save_model_projmat(self.data, None, modelpath, A_t, outpath, t)

            print("******** iteration:{} **********".format(t))
            if t == 2:
                A2_diff = LA.norm(A_t - A_t_1)
                print("******** A2 diff: {} ********".format(A2_diff))
            if t >= 2:
                A_diff = LA.norm(A_t - A_t_1)
                diff_ratio = A_diff / A2_diff
                print("******** A diff: {} diff_ratio: {} ********".format(A_diff, diff_ratio))

                #if t % 10 == 0:
                print("{0} - Ratio {1:.4f} - {2}".format(time.strftime("%H:%M:%S", time.localtime()), diff_ratio, t))
                if diff_ratio < eps or A2_diff == 0:
                    break
            A_t_1 = A_t

        if (A2_diff == 0):
            diff_ratio = "Error: A2_diff was zero!!!"
            return
        elif (diff_ratio < eps ):
            print("{} - Prosjmat converged - diff ratio: {}".format(time.strftime("%H:%M:%S", time.localtime()), diff_ratio))
        else:
            print("{} - Training finished without convergence - last ratio: {}".format(time.strftime("%H:%M:%S", time.localtime()), diff_ratio))

        save_model_projmat(self.data, None, modelpath, A_t, outpath)

        #lclf = self.learn_with_all_samples(A_t, trn_raw_data, trnL, modelpath, outpath, C)

        return A_t, lclf

    def learn_with_all_samples(self, projmat, samples, labels, model_path, projmat_path, c_):
        self.data.projmat = projmat
        lclf = MulticlassSVM(C=c_, tol=0.01, max_iter=1000, random_state=0, verbose=0)
        #lclf = LinearSVC(C=1.0, penalty='l1',loss='squared_hinge', dual=False, tol=1e-7)
        trnMArr = self.project_samples(samples, self.data)
        print('Training SVM with full training set for projmat {}'.format(projmat_path))
        lclf.fit(trnMArr, labels)
        save_model_projmat(self.data, lclf, model_path, projmat, projmat_path)
        print('Saved SVM in {}'.format(model_path))
        return lclf


    def solve_projmat_tile3(self, clf, B_prev, t, tau, x_vecs, y_labels, samples):
        """
            Solve projection matrix A iteratively
        """
        b_rows, b_cols = B_prev.shape[0], B_prev.shape[1]
        x_vecs = x_vecs[:,len(self.data.vocab):]

        # recreate matrix A by combining three B matrices orthogonally
        A_prev = np.zeros((b_rows*3, b_cols*3))
        for i in range(0,b_rows):
            for j in range(0,b_cols):
                for num in range(0, 3):
                    r_start = num * b_rows
                    c_start = num * b_cols
                    A_prev[r_start + i][c_start + j] = B_prev[i][j]
        A_prev = A_prev.T

        le = clf._label_encoder
        n_classes = len (le.classes_)
        n_vec = len(y_labels)

        alpha = clf.dual_coef_
        w = clf.coef_[:,len(self.data.vocab):]
        A = np.zeros(A_prev.shape)

        for i in range(n_vec):
            [y_i] = le.transform([y_labels[i]])
            expected_weight = sum([(self.delta_f(m, y_i) - alpha[m, i]) * w[m, :] for m in range(n_classes)])
            A += np.outer(w[y_i, :] - expected_weight, x_vecs[i].T)

        A = A.T

        # Estimate B as the sum of three corresponding blocks in matrix A
        B = np.zeros(B_prev.shape)
        for i in range(0,b_rows):
            for j in range(0,b_cols):
                B[i][j] = (A[i][j] + A[i+b_rows][j+b_cols] + A[i+b_rows*2][j+b_cols*2]) / 3

        B *= (1 / float(t))
        B += (1 - tau / float(t)) * B_prev
        return B

    def delta_f(self, m, y_i):
        if y_i == m:
            return 1
        return 0

    def solve_projmat_tiles1(self, clf, B_prev, t, tau, x_vecs, y_labels):
        """
            Solve projection matrix A iteratively
        """
        b_rows, b_cols = B_prev.shape[0], B_prev.shape[1]
        x_vecs = x_vecs[:,len(self.data.vocab):]

        A_prev = B_prev.T

        le = clf._label_encoder
        n_classes = len (le.classes_)
        n_vec = len(y_labels)

        alpha = clf.dual_coef_
        w = clf.coef_[:,len(self.data.vocab):]
        A = np.zeros(A_prev.shape)

        vec_arr = [x_vecs[:,:b_rows], x_vecs[:,b_rows:b_rows*2], x_vecs[:,b_rows*2:]]
        w_arr = [w[:,:b_cols], w[:,b_cols:b_cols*2], w[:,b_cols*2:]]

        for i in range(n_vec):
            [y_i] = le.transform([y_labels[i]])
            for j in range(3):
                expected_weight = sum([(self.delta_f(m, y_i) - alpha[m, i]) * w_arr[j][m, :] for m in range(n_classes)])
                A += np.outer(w_arr[j][y_i, :] - expected_weight, vec_arr[j][i].T)

        B = A.T

        B *= (1 / float(t))
        B += (1 - tau / float(t)) * B_prev
        return B

    def delta_f(self, m, y_i):
        if y_i == m:
            return 1
        return 0

    def test_eval(self, basepath, projmat_path, model_path, learn_model):
        if learn_model:
            print('{} - Loading test projmat'.format(time.strftime("%H:%M:%S", time.localtime())))
            with gzip.open(projmat_path, 'rb') as fin:
                projmat = load(fin)
                try:
                    projmat = projmat.toarray()
                except:
                    pass

            print('{} - Creating train data from corpus'.format(time.strftime("%H:%M:%S", time.localtime())))
            trn_raw_data, trnL = self.createdata_raw()
            trn_raw_data = trn_raw_data.toarray()

            found = re.search('C_(\d+)\.', projmat_path)
            if not found:
                c = 10
            c = int(found.group(1))

            print("{} - Training the SVM".format(time.strftime("%H:%M:%S", time.localtime())))
            self.learn_with_all_samples(projmat, trn_raw_data, trnL, model_path, projmat_path, c)

        print('{} - Evaluating the SVM'.format(time.strftime("%H:%M:%S", time.localtime())))
        return evalparser(path= '{}/dev/'.format(basepath), report=True,
                   bcvocab=self.data.bcvocab, draw=False,
                   withdp=True,
                   fdpvocab='{}/{}'.format(basepath, config["fdpvocab"]),
                   fprojmat= projmat_path,
                   modelpath= model_path)

def save_model_projmat(data, svm, fsvm, projmat, fprojmat, iteration = None):
    if svm is not None:
        if not fsvm.endswith('.gz'):
            fsvm += '.gz'
        D = {'clf':svm, 'vocab':data.vocab,
             'idxlabelmap':reversedict(data.labelmap)}
        with gzip.open(fsvm, 'wb') as fout:
            dump(D, fout)

    if iteration is not None:
        if iteration == 1 or iteration % 10 == 0:
            fprojmat = fprojmat.replace(".projmat", "_iter_{}.projmat".format(iteration))
        else:
            return # don't save projmat on all iterations
    if projmat is not None:
        if not fprojmat.endswith('.gz'):
            fprojmat += '.gz'
        with gzip.open(fprojmat, 'wb') as fout:
            dump(projmat, fout)
    

if __name__ == '__main__':
    print('====================== parser training =================')

    parser = argparse.ArgumentParser(
        prog= os.path.basename(__file__),
        description='Creates projection matrix.',
    )
    parser.add_argument('basepath')           # positional argument
    parser.add_argument("-e", "--proceed", type=bool, default=False, help="Continue training after a pause or stop (default = False)")
    parser.add_argument("-n", "--num_matrix", type=int, default=0, help="Number of matrices to check (0: unlimited)")
    parser.add_argument("-z", "--randomize", type=bool, default=False)
    parser.add_argument("-i", "--iterations", type=int, default=100, help="Train iterations (default = 100)")
    parser.add_argument("-r", "--report_file", type=str, default='projmat_report.txt', help="Report file name")
    parser.add_argument("-o", "--out_file", type=str, default='projmat.pickle.gz', help="The output file; the projection matrix with best score.")
    parser.add_argument("-c", "--c_param", type=int, default=0)
    parser.add_argument("-t", "--tau_param", type=float, default=0)
    parser.add_argument("-k", "--k_param", type=int, default=0)
    parser.add_argument("-m", "--mat_type", type=str, default="identity")

    args = parser.parse_args()

    basepath = args.basepath
    modelpath = basepath + '/model'
    outpath = modelpath + '/' + args.out_file
    reportpath = modelpath+ '/' + args.report_file

    pm = ProjmatLearn(basepath, topn=10000)
    pm.create_identity_matrix(outpath= modelpath + '/identity.projmat.gz')
    pm.create_zero_matrix(outpath= modelpath + '/zero.projmat.gz')

    if args.mat_type == "identity" or args.mat_type == "zero":
        modelpath = "{}/{}.projmat.gz".format(modelpath, args.mat_type)
        shutil.copy(modelpath, outpath)
        print("{} matrix is created and copied to {}".format(modelpath, outpath))
        exit(0)

    C = [1, 10, 50, 100]
    if (args.c_param > 0):
        C = [args.c_param]
    
    tau = [0.5, 0.999, 0.1, 0.01, 0.001]
    if (args.tau_param > 0):
        tau = [args.tau_param]
    
    K = [150, 90, 60, 30]
    if (args.k_param > 0):
        K = [args.k_param]
    total = len(K) * len(tau) * len(C)
    
    n_sample = 500
    
    if os.path.exists(reportpath) and not args.proceed:
        os.remove(reportpath)
    
    best = {'score': 0, 'matrix': None}
    if args.proceed and os.path.exists(reportpath):
        with open(reportpath) as f:
            best = f.readlines()[-1]
        
        tokens = best.split()
        assert(tokens[0] == 'Best:')
        best = {'matrix': tokens[2], 'score': tokens[4]}
        print('The best score {} found for matrix {}'.format(best['score'], best['matrix']))

    count = 0
    for k in K:
        for t in tau:
            for ci in C:
                count += 1

                if args.num_matrix > 0 and count > args.num_matrix:
                    "The matrix count limit {} is reached".format(count)
                    break
                
                if args.randomize and args.num_matrix > 0:
                    if (count - 1) % (total / args.num_matrix) != 0:
                        continue

                modelname = '{}/K_{}_Tau_{}_C_{}.model.gz'.format(modelpath, k, t, ci)
                projmatname = '{}/K_{}_Tau_{}_C_{}.projmat.gz'.format(modelpath, k, t, ci)
                if os.path.exists(modelname) and args.proceed:
                    continue

                pm.train(modelpath=modelname, outpath=projmatname, n_samples=n_sample, K=k, tau=t, C=ci, iterations=args.iterations, eps=0.0001)

                # if args.single_matrix:
                #     shutil.copy(projmatname, outpath)
                #     print('The matrix {} is created and moved to {}.'.format(projmatname, outpath))
                    
                result = pm.test_eval(basepath, projmatname, modelname, True)
                if result[2] > best['score']:
                    best = {'matrix': projmatname, 'score': result[2]}
                
                with open(reportpath, 'a') as rep:
                    rep.write('{} ------------ training results with projmat: {}\n'.format(projmatname, time.strftime("%H:%M:%S", time.localtime())))
                    rep.write('Span: {} Nuclearity: {} Relation: {}\n'.format(result[0], result[1], result[2]))
                    rep.write('Best: matrix {} score {}\n'.format(best["matrix"], best["score"]))


    if best['score'] > 0:
        print('The best matrix {} is copied to output: {}'.format(best['matrix'], outpath))
        shutil.copy(best['matrix'], outpath)

# Cross validation should also be included. 
# ACL Parsers emailed
# Compare different parsers. 
# DPLP, StageDP, ACL parsers. 
# Argumentative Microtext > second corpus just below PCC
