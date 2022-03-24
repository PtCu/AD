from mlni.base import WorkFlow, RegressionAlgorithm, RegressionValidation
import numpy as np
import pandas as pd
import os, json
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, ShuffleSplit
from multiprocessing.pool import ThreadPool
from mlni.utils import time_bar
from joblib import dump

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen, Jorge Samper-González"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

class RB_RepeatedHoldOut_DualSVM_Regression(WorkFlow):
    """
    The main class to run MLNI with repeated holdout CV for regression.
    """

    def __init__(self, input, split_index, output_dir, n_threads=8, n_iterations=100, test_size=0.2,
                 grid_search_folds=5,
                 c_range=[j * 10 ** k for k in range(1, 4) for j in [1, 2, 4, 7]],
                 gamma_range=['auto', 'scale'],
                 kernel=None, verbose=False):
        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._grid_search_folds = grid_search_folds
        self._gamma_range = gamma_range
        self._c_range = c_range
        self._verbose = verbose
        self._test_size = test_size
        self._validation = None
        self._algorithm = None
        self._kernel = kernel

    def run(self):
        x = self._input.get_x()
        y = self._input.get_y_raw()

        self._algorithm = SVRAlgorithmWithRBFKernel(x, y, grid_search_folds=self._grid_search_folds, c_range=self._c_range,
                                                     gamma_range=self._gamma_range, n_threads=self._n_threads, verbose=self._verbose)

        self._validation = RepeatedHoldOut(self._algorithm, n_iterations=self._n_iterations, test_size=self._test_size)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._split_index, verbose=self._verbose)
        classifier_dir = os.path.join(self._output_dir, 'regressor')
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        self._algorithm.save_weights(classifier, x, classifier_dir)
        self._validation.save_results(self._output_dir)

class VB_RepeatedHoldOut_DualSVM_Regression(WorkFlow):
    """
    The main class to run MLNI with repeated holdout CV for regression.
    """

    def __init__(self, input, split_index, output_dir, n_threads=8, n_iterations=100, test_size=0.2,
                 grid_search_folds=5, c_range=[j * 10 ** k for k in range(1, 4) for j in [1, 2, 4, 7]],
                 gamma_range=['auto', 'scale'],
                 kernel=None, verbose=False):
        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._grid_search_folds = grid_search_folds
        self._c_range = c_range
        self._gamma_range = gamma_range
        self._verbose = verbose
        self._test_size = test_size
        self._validation = None
        self._algorithm = None
        self._kernel = kernel

    def run(self):
        x = self._input.get_x()
        y = self._input.get_y_raw()

        self._algorithm = SVRAlgorithmWithRBFKernel(x, y, grid_search_folds=self._grid_search_folds,
                                                     c_range=self._c_range, gamma_range=self._gamma_range,
                                                     n_threads=self._n_threads, verbose=self._verbose)

        self._validation = RepeatedHoldOut(self._algorithm, n_iterations=self._n_iterations, test_size=self._test_size)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._split_index, verbose=self._verbose)
        classifier_dir = os.path.join(self._output_dir, 'regressor')
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        weights = self._algorithm.save_weights(classifier, x, classifier_dir)
        self._input.save_weights_as_nifti(weights, classifier_dir)
        self._validation.save_results(self._output_dir)

class SVRAlgorithmWithRBFKernel(RegressionAlgorithm):
    '''
    SVR with precomputed linear kernel.
    '''
    def __init__(self, x, y, grid_search_folds=10, c_range=[j * 10 ** k for k in range(1, 4) for j in [1, 2, 4, 7]],
                 gamma_range=['auto', 'scale'], n_threads=15, verbose=False):
        self._x = x
        self._y = y
        self._grid_search_folds = grid_search_folds
        self._c_range = c_range
        self._gamma_range = gamma_range
        self._n_threads = n_threads
        self._verbose = verbose

    def _launch_svr(self, x_train, x_test, y_train, y_test, c, gamma):
        svr = SVR(C=c, kernel='rbf', gamma=gamma)
        svr.fit(x_train, y_train)
        y_hat_train = svr.predict(x_train)
        y_hat = svr.predict(x_test)
        ## compute the mae
        mae = mean_absolute_error(y_test, y_hat)

        return svr, y_hat, y_hat_train, mae

    def _grid_search(self, x_train, x_test, y_train, y_test, c, gamma):

        _, _, _, mae = self._launch_svr(x_train, x_test, y_train, y_test, c, gamma)

        return mae

    def _select_best_parameter(self, async_result):
        results_array = np.empty((self._grid_search_folds, len(self._gamma_range), len(self._c_range)))

        for key, values in async_result.items():
            gamma = key[0]
            index_gamma = self._gamma_range.index(gamma)
            c = key[1]
            index_c = self._c_range.index(c)
            fold = key[2]
            results_array[fold, index_gamma, index_c] = values.get()

        index_fold, index_gamma_opt, index_c_opt = np.where(results_array == results_array.min())
        best_mae = results_array[index_fold[0], index_gamma_opt[0], index_c_opt[0]]
        best_gamma = self._gamma_range[index_gamma_opt[0]]
        best_c = self._c_range[index_c_opt[0]]
        
        return {'gamma': best_gamma, 'c': best_c, 'mae': best_mae}

    def evaluate(self, train_index, test_index):

        inner_pool = ThreadPool(processes=self._n_threads)
        y_train = self._y[train_index]

        async_result = {}

        for j in range(len(self._gamma_range)):
            x_train = self._x[train_index]
            skf = KFold(n_splits=self._grid_search_folds, shuffle=True)
            inner_cv = list(skf.split(np.zeros(len(y_train)), y_train))
            for k in range(len(inner_cv)):
                inner_train_index, inner_test_index = inner_cv[k]
                x_train_inner = x_train[inner_train_index]
                x_test_inner = x_train[inner_test_index]
                y_train_inner, y_test_inner = y_train[inner_train_index], y_train[inner_test_index]
                for l in range(len(self._c_range)):
                    if self._verbose:
                        print("Inner CV for c=%f and gamma=%s..." % (self._c_range[l], self._gamma_range[j]))
                    async_result[self._gamma_range[j], self._c_range[l], k] = inner_pool.apply_async(self._grid_search,
                                    args=(x_train_inner, x_test_inner, y_train_inner, y_test_inner, self._c_range[l], self._gamma_range[j]))
        inner_pool.close()
        inner_pool.join()

        best_parameter = self._select_best_parameter(async_result)
        x_train = self._x[train_index]
        x_test = self._x[test_index]
        y_train, y_test = self._y[train_index], self._y[test_index]

        _, y_hat, y_hat_train, mae = self._launch_svr(x_train, x_test, y_train, y_test, best_parameter['c'], best_parameter['gamma'])

        result = dict()
        result['best_parameter'] = best_parameter
        result['y_hat'] = y_hat
        result['y_hat_train'] = y_hat_train
        result['y'] = y_test
        result['y_train'] = y_train
        result['y_index'] = test_index
        result['x_index'] = train_index
        result['mae'] = mae

        return result

    def apply_best_parameters(self, results_list):

        best_c_list = []
        best_gamma_list = []
        bal_mae_list = []

        for result in results_list:
            best_c_list.append(result['best_parameter']['c'])
            best_gamma_list.append(result['best_parameter']['gamma'])
            bal_mae_list.append(result['best_parameter']['mae'])
        best_c = np.power(10, np.mean(np.log10(best_c_list)))
        best_gamma = max(best_gamma_list, key = best_gamma_list.count)
        # MAE
        mean_mae = np.mean(bal_mae_list)
        svr = SVR(C=best_c, kernel='rbf', gamma=best_gamma)
        svr.fit(self._x, self._y)

        return svr, {'gamma': best_gamma, 'c': best_c, 'mae': mean_mae}

    def save_classifier(self, classifier, output_dir):

        np.savetxt(os.path.join(output_dir, 'dual_coefficients.txt'), classifier.dual_coef_)
        np.savetxt(os.path.join(output_dir, 'support_vectors_indices.txt'), classifier.support_)
        np.savetxt(os.path.join(output_dir, 'intersect.txt'), classifier.intercept_)

        ## save the svr instance to apply external test data
        dump(classifier, os.path.join(output_dir, 'svr.joblib'))

    def save_weights(self, classifier, x, output_dir):

        dual_coefficients = classifier.dual_coef_
        sv_indices = classifier.support_

        weighted_sv = dual_coefficients.transpose() * x[sv_indices]
        weights = np.sum(weighted_sv, 0)

        np.savetxt(os.path.join(output_dir, 'weights.txt'), weights)

        return weights

    def save_parameters(self, parameters_dict, output_dir):
        with open(os.path.join(output_dir, 'best_parameters.json'), 'w') as f:
            json.dump(parameters_dict, f)

class RepeatedHoldOut(RegressionValidation):
    """
    Repeated holdout splits CV.
    """

    def __init__(self, ml_algorithm, n_iterations=100, test_size=0.3):
        self._ml_algorithm = ml_algorithm
        self._split_results = []
        self._classifier = None
        self._best_params = None
        self._cv = None
        self._n_iterations = n_iterations
        self._test_size = test_size

    def validate(self, y, n_threads=15, splits_indices=None, inner_cv=True, verbose=False):

        if splits_indices is None:
            splits = ShuffleSplit(n_splits=self._n_iterations, test_size=self._test_size)
            self._cv = list(splits.split(np.zeros(len(y)), y))
        else:
            self._cv = splits_indices
        async_pool = ThreadPool(processes=n_threads)
        async_result = {}

        for i in range(self._n_iterations):
            time_bar(i, self._n_iterations)
            print()
            if verbose:
                print("Repetition %d of CV..." % i)
            train_index, test_index = self._cv[i]
            if inner_cv:
                async_result[i] = async_pool.apply_async(self._ml_algorithm.evaluate, args=(train_index, test_index))
            else:
                raise Exception("We always do nested CV")

        async_pool.close()
        async_pool.join()

        for i in range(self._n_iterations):
            self._split_results.append(async_result[i].get())

        self._classifier, self._best_params = self._ml_algorithm.apply_best_parameters(self._split_results)
        return self._classifier, self._best_params, self._split_results

    def save_results(self, output_dir):
        if self._split_results is None:
            raise Exception("No results to save. Method validate() must be run before save_results().")

        all_results_list = []
        all_train_subjects_list = []
        all_test_subjects_list = []

        for iteration in range(len(self._split_results)):

            iteration_dir = os.path.join(output_dir, 'iteration-' + str(iteration))
            if not os.path.exists(iteration_dir):
                os.makedirs(iteration_dir)
            iteration_train_subjects_df = pd.DataFrame({'iteration': iteration,
                                                        'y': self._split_results[iteration]['y_train'],
                                                        'y_hat': self._split_results[iteration]['y_hat_train'],
                                                        'subject_index': self._split_results[iteration]['x_index']})
            iteration_train_subjects_df.to_csv(os.path.join(iteration_dir, 'train_subjects.tsv'),
                                               index=False, sep='\t', encoding='utf-8')
            all_train_subjects_list.append(iteration_train_subjects_df)

            iteration_test_subjects_df = pd.DataFrame({'iteration': iteration,
                                                       'y': self._split_results[iteration]['y'],
                                                       'y_hat': self._split_results[iteration]['y_hat'],
                                                       'subject_index': self._split_results[iteration]['y_index']})
            iteration_test_subjects_df.to_csv(os.path.join(iteration_dir, 'test_subjects.tsv'),
                                              index=False, sep='\t', encoding='utf-8')
            all_test_subjects_list.append(iteration_test_subjects_df)

            iteration_results_df = pd.DataFrame(
                    {'mae': self._split_results[iteration]['mae'],
                     }, index=['i', ])
            iteration_results_df.to_csv(os.path.join(iteration_dir, 'results.tsv'),
                                        index=False, sep='\t', encoding='utf-8')

            all_results_list.append(iteration_results_df)

        all_train_subjects_df = pd.concat(all_train_subjects_list)
        all_train_subjects_df.to_csv(os.path.join(output_dir, 'train_subjects.tsv'),
                                     index=False, sep='\t', encoding='utf-8')

        all_test_subjects_df = pd.concat(all_test_subjects_list)
        all_test_subjects_df.to_csv(os.path.join(output_dir, 'test_subjects.tsv'),
                                    index=False, sep='\t', encoding='utf-8')

        all_results_df = pd.concat(all_results_list)
        all_results_df.to_csv(os.path.join(output_dir, 'results.tsv'),
                              index=False, sep='\t', encoding='utf-8')

        mean_results_df = pd.DataFrame(all_results_df.apply(np.nanmean).to_dict(),
                                       columns=all_results_df.columns, index=[0, ])
        mean_results_df.to_csv(os.path.join(output_dir, 'mean_results.tsv'),
                               index=False, sep='\t', encoding='utf-8')

        print("Mean results of the regression:")
        print("Mean absolute error: %s" % (mean_results_df['mae'].to_string(index=False)))

class RB_KFold_DualSVM_Regression(WorkFlow):
    """
    The main class to run MLNI with stritified KFold CV for regression with ROI features.
    """

    def __init__(self, input, split_index, output_dir, n_folds, n_threads=8, grid_search_folds=10,
                 c_range=[j * 10 ** k for k in range(1, 4) for j in [1, 2, 4, 7]],
                 gamma_range=['auto', 'scale'],
                 kernel=None, verbose=False):
        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._grid_search_folds = grid_search_folds
        self._c_range = c_range
        self._gamma_range = gamma_range
        self._verbose = verbose
        self._n_folds = n_folds
        self._validation = None
        self._algorithm = None
        self._kernel = kernel

    def run(self):
        x = self._input.get_x()
        y = self._input.get_y_raw()

        self._algorithm = SVRAlgorithmWithRBFKernel(x, y,
                                                     grid_search_folds=self._grid_search_folds, c_range=self._c_range,
                                                    gamma_range=self._gamma_range,
                                                     n_threads=self._n_threads, verbose=self._verbose)

        self._validation = KFoldCV(self._algorithm)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._split_index,
                                                                     n_folds=self._n_folds, verbose=self._verbose)
        classifier_dir = os.path.join(self._output_dir, 'regressor')
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        self._algorithm.save_weights(classifier, x, classifier_dir)
        self._validation.save_results(self._output_dir)

class VB_KFold_DualSVM_Regression(WorkFlow):
    """
    The main class to run MLNI with stritified KFold CV for regression with voxel-wise images.
    """

    def __init__(self, input, split_index, output_dir, n_folds, n_threads=8, grid_search_folds=10,
                 c_range=[j * 10 ** k for k in range(1, 4) for j in [1, 2, 4, 7]],
                 gamma_range=['auto', 'scale'], kernel=None, verbose=False):
        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._grid_search_folds = grid_search_folds
        self._c_range = c_range
        self._gamma_range = gamma_range
        self._verbose = verbose
        self._n_folds = n_folds
        self._validation = None
        self._algorithm = None
        self._kernel = kernel

    def run(self):
        x = self._input.get_x()
        y = self._input.get_y_raw()

        self._algorithm = SVRAlgorithmWithRBFKernel(x, y, grid_search_folds=self._grid_search_folds, c_range=self._c_range,
                                                    gamma_range=self._gamma_range, n_threads=self._n_threads, verbose=self._verbose)

        self._validation = KFoldCV(self._algorithm)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._split_index,
                                                                     n_folds=self._n_folds, verbose=self._verbose)
        classifier_dir = os.path.join(self._output_dir, 'regressor')
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        weights = self._algorithm.save_weights(classifier, x, classifier_dir)
        self._input.save_weights_as_nifti(weights, classifier_dir)
        self._validation.save_results(self._output_dir)

class KFoldCV(RegressionValidation):
    """
    KFold CV.
    """
    def __init__(self, ml_algorithm):
        self._ml_algorithm = ml_algorithm
        self._fold_results = []
        self._classifier = None
        self._best_params = None
        self._cv = None

    def validate(self, y, n_folds=10, n_threads=15, splits_indices=None, verbose=False):
        if splits_indices is None:
            skf = KFold(n_splits=n_folds, shuffle=True)
            self._cv = list(skf.split(np.zeros(len(y)), y))
        else:
            self._cv = splits_indices

        async_pool = ThreadPool(processes=n_threads)
        async_result = {}

        for i in range(n_folds):
            time_bar(i, n_folds)
            print()
            if verbose:
                print("Repetition %d of CV..." % i)
            train_index, test_index = self._cv[i]
            async_result[i] = async_pool.apply_async(self._ml_algorithm.evaluate, args=(train_index, test_index))
        async_pool.close()
        async_pool.join()

        for i in range(n_folds):
            self._fold_results.append(async_result[i].get())

        ## save the mean of the best models
        self._classifier, self._best_params = self._ml_algorithm.apply_best_parameters(self._fold_results)

        return self._classifier, self._best_params, self._fold_results

    def save_results(self, output_dir):
        if self._fold_results is None:
            raise Exception("No results to save. Method validate() must be run before save_results().")

        subjects_folds = []
        results_folds = []
        container_dir = os.path.join(output_dir, 'folds')

        if not os.path.exists(container_dir):
            os.makedirs(container_dir)

        for i in range(len(self._fold_results)):
            subjects_df = pd.DataFrame({'y': self._fold_results[i]['y'],
                                        'y_hat': self._fold_results[i]['y_hat'],
                                        'y_index': self._fold_results[i]['y_index']})
            subjects_df.to_csv(os.path.join(container_dir, 'subjects_fold-' + str(i) + '.tsv'),
                               index=False, sep='\t', encoding='utf-8')
            subjects_folds.append(subjects_df)

            results_df = pd.DataFrame({'mae': self._fold_results[i]['mae']}, index=['i', ])
            results_df.to_csv(os.path.join(container_dir, 'results_fold-' + str(i) + '.tsv'),
                              index=False, sep='\t', encoding='utf-8')
            results_folds.append(results_df)

        all_subjects = pd.concat(subjects_folds)
        all_subjects.to_csv(os.path.join(output_dir, 'subjects.tsv'),
                            index=False, sep='\t', encoding='utf-8')

        all_results = pd.concat(results_folds)
        all_results.to_csv(os.path.join(output_dir, 'results.tsv'),
                           index=False, sep='\t', encoding='utf-8')

        mean_results = pd.DataFrame(all_results.apply(np.nanmean).to_dict(), columns=all_results.columns, index=[0, ])
        mean_results.to_csv(os.path.join(output_dir, 'mean_results.tsv'),
                            index=False, sep='\t', encoding='utf-8')
        print("Mean results of the regression:")
        print("Mean absolute error: %s" % (mean_results['mae'].to_string(index = False)))
