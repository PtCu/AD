# import the python packages needed to generate simulated data for the tutorial
from sim.simfuncs import generate_random_Zscore_sustain_model, generate_data_Zscore_sustain
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sklearn.model_selection
import pandas as pd
import pylab
import sys
import pySuStaIn

################################################# # The linear score mode #################################################
# # this needs to point to wherever the sim folder inside pySuStaIn is on your computer
# sys.path.insert(
#     0, '/Users/alexandrayoung/Documents/Code/pySuStaIn-test/pySuStaIn/sim/')
# # if you're running the notebook from within the existing structure you can use
# # sys.path.insert(0,'../sim/')

cwd_path = os.getcwd()+"/Sustain"

# N = 5         # number of biomarkers

# SuStaInLabels = []
# for i in range(N):
#     # labels of biomarkers for plotting
#     SuStaInLabels.append('Biomarker '+str(i))

# Z_vals = np.array([[1, 2, 3]]*N)     # Z-scores for each biomarker
# Z_max = np.array([5]*N)           # maximum z-score

# # To demonstrate how to set different biomarkers to have different z-scores,
# # set biomarker 0 to have z-scores of 1 and 2 only and a maximum of 3
# # to do this change the corresponding row of Z_vals to read 1 2 0
# # and change the corresponding row of Z_max to 3
# Z_vals[np.array(0), np.array(2)] = 0
# Z_max[np.array(0)] = 3

# # and set biomarker 2 to have a z-score of 1 only and a maximum of 2
# # to do this change the corresponding row of Z_vals to read 1 0 0
# # and change the corresponding row of Z_max to 2
# Z_vals[np.array(2), np.array([1, 2])] = 0
# Z_max[np.array(2)] = 2


# # generate a random sequence for the linear z-score model
# gt_sequence = generate_random_Zscore_sustain_model(Z_vals,
#                                                    1)

# # ignore this part, it's only necessary so that the generate_data_sustain function
# # can be used in this demo setting
# gt_stages = np.array([0])
# gt_subtypes = np.array([0])

# # this code generates data from z-score sustain
# # - here i've just output the z-score model itself rather than any datapoints
# _, _, gt_stage_value = generate_data_Zscore_sustain(gt_subtypes,
#                                                     gt_stages,
#                                                     gt_sequence,
#                                                     Z_vals,
#                                                     Z_max)

# # ignore this part, just calculates some parameters of sustain to output below
# stage_zscore = np.array([y for x in Z_vals.T for y in x])
# stage_zscore = stage_zscore.reshape(1, len(stage_zscore))
# IX_select = stage_zscore > 0
# stage_zscore = stage_zscore[IX_select]
# stage_zscore = stage_zscore.reshape(1, len(stage_zscore))
# num_zscores = Z_vals.shape[1]
# IX_vals = np.array([[x for x in range(N)]] * num_zscores).T
# stage_biomarker_index = np.array([y for x in IX_vals.T for y in x])
# stage_biomarker_index = stage_biomarker_index.reshape(
#     1, len(stage_biomarker_index))
# stage_biomarker_index = stage_biomarker_index[IX_select]
# stage_biomarker_index = stage_biomarker_index.reshape(
#     1, len(stage_biomarker_index))

# # print out some of the values and plot a picture of the model
# print('Simulated sequence:', (gt_sequence.astype(int).flatten()))
# print('At the beginning of the progression (stage 0) the biomarkers have scores of 0')
# print('At the stages:', 1+np.arange(np.array(stage_zscore).shape[1]))
# print('the biomarkers:',
#       stage_biomarker_index[:, gt_sequence.astype(int).flatten()].flatten())
# print('reach z-scores of:',
#       stage_zscore[:, gt_sequence.astype(int).flatten()].flatten())
# print('At the end of the progression (stage', np.array(
#       stage_zscore).shape[1]+2, ') the biomarkers reach scores of:', Z_max)
# print('The z-score model assumes individuals belong to some unknown stage of this progression,')
# print('with gaussian noise with a standard deviation of 1 for each biomarker')

# temp_stages = np.array(range(np.array(stage_zscore).shape[1]+2))
# for b in range(N):
#     ax = plt.plot(temp_stages, gt_stage_value[b, :, :])

# _ = plt.xlabel('SuStaIn stage')
# _ = plt.ylabel('Z-score')
# _ = plt.legend(SuStaInLabels)
# _ = plt.title('Figure 1')


################################################## Generate simulated data #################################################
N = 5         # number of biomarkers
M = 500       # number of observations ( e.g. subjects )
M_control = 100       # number of these that are control subjects
N_S_gt = 2         # number of ground truth subtypes

SuStaInLabels = []
for i in range(N):
    # labels of biomarkers for plotting
    SuStaInLabels.append('Biomarker '+str(i))

Z_vals = np.array([[1, 2, 3]]*N)     # Z-scores for each biomarker
Z_max = np.array([5]*N)           # maximum z-score

# ground truth proportion of individuals belonging to each subtype
gt_f = [1+0.5*x for x in range(N_S_gt)]
gt_f = [x/sum(gt_f) for x in gt_f][::-1]

# ground truth sequence for each subtype
gt_sequence = generate_random_Zscore_sustain_model(Z_vals,
                                                   N_S_gt)

# simulate subtypes and stages for individuals, including a control population at stage 0
N_k = np.sum(Z_vals > 0)+1
gt_subtypes = np.random.choice(range(N_S_gt), M, replace=True, p=gt_f)
gt_stages_control = np.zeros((M_control, 1))
gt_stages = np.concatenate((gt_stages_control,
                            np.ceil(np.random.rand(M-M_control, 1)*N_k)),
                           axis=0)

# generate simulated data
data, gt_data_denoised, gt_stage_value = generate_data_Zscore_sustain(gt_subtypes,
                                                                      gt_stages,
                                                                      gt_sequence,
                                                                      Z_vals,
                                                                      Z_max)

# ignore this part, just calculates some parameters of sustain to output below
stage_zscore = np.array([y for x in Z_vals.T for y in x])
stage_zscore = stage_zscore.reshape(1, len(stage_zscore))
IX_select = stage_zscore > 0
stage_zscore = stage_zscore[IX_select]
stage_zscore = stage_zscore.reshape(1, len(stage_zscore))
num_zscores = Z_vals.shape[1]
IX_vals = np.array([[x for x in range(N)]] * num_zscores).T
stage_biomarker_index = np.array([y for x in IX_vals.T for y in x])
stage_biomarker_index = stage_biomarker_index.reshape(
    1, len(stage_biomarker_index))
stage_biomarker_index = stage_biomarker_index[IX_select]
stage_biomarker_index = stage_biomarker_index.reshape(
    1, len(stage_biomarker_index))

for s in range(N_S_gt):
    # print out the parameters
    print('For subtype', s, '(', gt_f[s]*100, '% of individuals)')
    print('Simulated sequence:', (gt_sequence[s, :].astype(int).flatten()))
    print('At the beginning of the progression (stage 0) the biomarkers have scores of 0')
    print('At the stages:', 1+np.arange(np.array(stage_zscore).shape[1]))
    print('the biomarkers:', stage_biomarker_index[:, gt_sequence[s, :].astype(
        int).flatten()].flatten())
    print('reach z-scores of:',
          stage_zscore[:, gt_sequence[s, :].astype(int).flatten()].flatten())
    print('At the end of the progression (stage', np.array(
        stage_zscore).shape[1]+2, ') the biomarkers reach scores of:', Z_max)
    print('')


################################################## SustaIn #################################################
# extract data for control subjects
data_control = data[np.tile(gt_stages, (1, N)) == 0].reshape(M_control, N)

# compute the mean and standard deviation of the control population
mean_control = np.mean(data_control, axis=0)
std_control = np.std(data_control, axis=0)

# z-score the data
data = (data-mean_control)/std_control
data_control = (data_control-mean_control)/std_control

# multiply data for decreasing biomarkers by -1
IS_decreasing = np.mean(data, axis=0) < np.mean(data_control, axis=0)
data[np.tile(IS_decreasing, (M, 1))] = -1*data[np.tile(IS_decreasing, (M, 1))]
data_control[np.tile(IS_decreasing, (M_control, 1))] = - \
    1*data_control[np.tile(IS_decreasing, (M_control, 1))]

# Check that the mean of the control population is 0
print('Mean of controls is ', np.mean(data_control, axis=0))
# Check that the standard deviation of the control population is 1
print('Standard deviation of controls is ', np.std(data_control, axis=0))
# Check that the mean of the whole dataset is positive
print('Mean of whole dataset is ', np.mean(data, axis=0))
# Check that the standard deviation of the whole dataset is greater than 1
print('Standard deviation of whole dataset is ', np.std(data, axis=0))


# Input the settings for z-score SuStaIn
# To make the tutorial run faster I've set
# N_startpoints = 10 and N_iterations_MCMC = int(1e4)
# I recommend using N_startpoints = 25 and
# N_iterations_MCMC = int(1e5) or int(1e6) in general though
N_startpoints = 10
N_S_max = N_S_gt+1
N_iterations_MCMC = int(1e4)
output_folder =cwd_path+"/sim"
dataset_name = 'sim'
sustain_input = pySuStaIn.ZscoreSustain(data,
                                        Z_vals,
                                        Z_max,
                                        SuStaInLabels,
                                        N_startpoints,
                                        N_S_max,
                                        N_iterations_MCMC,
                                        output_folder,
                                        dataset_name,
                                        False)

# runs the sustain algorithm with the inputs set in sustain_input above
samples_sequence,   \
    samples_f,          \
    ml_subtype,         \
    prob_ml_subtype,    \
    ml_stage,           \
    prob_ml_stage,      \
    prob_subtype_stage = sustain_input.run_sustain_algorithm()


# Output a figure showing the ground truth
temp_gt_sequence = np.tile(np.reshape(
    gt_sequence, (gt_sequence.shape[0], gt_sequence.shape[1], 1)), 100)
temp_gt_f = np.asarray(gt_f).reshape(len(gt_f), 1)
pySuStaIn.ZscoreSustain._plot_sustain_model(
    sustain_input, temp_gt_sequence, temp_gt_f, M, subtype_order=(0, 1))
_ = plt.suptitle('Figure 3: Ground truth progression pattern')

# The code below opens the results for the ground truth number of subtypes
# and plots the output
s = N_S_gt-1
pickle_filename_s = output_folder + '/pickle_files/' + \
    dataset_name + '_subtype' + str(s) + '.pickle'
pickle_filepath = Path(pickle_filename_s)
pickle_file = open(pickle_filename_s, 'rb')
loaded_variables = pickle.load(pickle_file)
samples_sequence = loaded_variables["samples_sequence"]
samples_f = loaded_variables["samples_f"]
pickle_file.close()

pySuStaIn.ZscoreSustain._plot_sustain_model(
    sustain_input, samples_sequence, samples_f, M, subtype_order=(0, 1))
_ = plt.suptitle('Figure 4: SuStaIn output')


# go through each subtypes model and plot MCMC samples of the likelihood
for s in range(N_S_max):
    pickle_filename_s = output_folder + '/pickle_files/' + \
        dataset_name + '_subtype' + str(s) + '.pickle'
    pickle_filepath = Path(pickle_filename_s)
    pickle_file = open(pickle_filename_s, 'rb')
    loaded_variables = pickle.load(pickle_file)
    samples_likelihood = loaded_variables["samples_likelihood"]
    pickle_file.close()

    _ = plt.figure(0)
    _ = plt.plot(range(N_iterations_MCMC),
                 samples_likelihood, label="subtype" + str(s))
    _ = plt.figure(1)
    _ = plt.hist(samples_likelihood, label="subtype" + str(s))

_ = plt.figure(0)
_ = plt.legend(loc='upper right')
_ = plt.xlabel('MCMC samples')
_ = plt.ylabel('Log likelihood')
_ = plt.title('Figure 5: MCMC trace')

_ = plt.figure(1)
_ = plt.legend(loc='upper right')
_ = plt.xlabel('Log likelihood')
_ = plt.ylabel('Number of samples')
_ = plt.title('Figure 6: Histograms of model likelihood')


# identify a control population
index_control = np.reshape(gt_stages, (M)) == 0

# label cases and controls to perform stratified cross-validation
labels = 1 * np.ones(data.shape[0], dtype=int)
labels[index_control] = 0

# choose the number of folds - here i've used three for speed but i recommend 10 typically
N_folds = 3

# generate stratified cross-validation training and test set splits
cv = sklearn.model_selection.StratifiedKFold(n_splits=N_folds,
                                             shuffle=True)
cv_it = cv.split(data, labels)

test_idxs = []
for train, test in cv_it:
    test_idxs.append(test)
test_idxs = np.array(test_idxs)

s = N_S_gt-1
Nfolds = len(test_idxs)
for fold in range(Nfolds):
    pickle_filename_fold_s = sustain_input.output_folder + '/pickle_files/' + \
        sustain_input.dataset_name + '_fold' + \
        str(fold) + '_subtype' + str(s) + '.pickle'
    pickle_filepath = Path(pickle_filename_fold_s)

    pickle_file = open(pickle_filename_fold_s, 'rb')

    loaded_variables = pickle.load(pickle_file)

    samples_sequence = loaded_variables["samples_sequence"]
    samples_f = loaded_variables["samples_f"]

    pickle_file.close()

    if fold == 0:
        samples_sequence_cval = samples_sequence
        samples_f_cval = samples_f
    else:
        samples_sequence_cval = np.concatenate(
            (samples_sequence_cval, samples_sequence), axis=2)
        samples_f_cval = np.concatenate((samples_f_cval, samples_f), axis=1)

N_samples = 1000
ml_subtype_cval,             \
    prob_ml_subtype_cval,        \
    ml_stage_cval,               \
    prob_ml_stage_cval,          \
    prob_subtype_cval,           \
    prob_stage_cval,             \
    prob_subtype_stage_cval = sustain_input.subtype_and_stage_individuals_newData(data,
                                                                                  samples_sequence_cval,
                                                                                  samples_f_cval,
                                                                                  N_samples)
