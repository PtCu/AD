import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pySuStaIn
import statsmodels.formula.api as smf
from scipy import stats
import sklearn.model_selection
# Load and view simulated tutorial data
# The data needs to be in the same directory as your notebook
cwd_path = os.getcwd()

output_file_with_cov = cwd_path+"/Sustain/output/output_with_cov.tsv"

output_file_without_cov = cwd_path+"/Sustain/output/output_without_cov.tsv"

simulated_data = cwd_path+"/Sustain/data/simulated_data.tsv"

outcome_file = cwd_path+"/Sustain/output/oucome.txt"

output_dir=cwd_path+"/Sustain/output/"

data = pandas.read_csv(simulated_data, delimiter='\t')

data.head()
data.GROUP.value_counts()
# store our biomarker labels as a variable
biomarkers = data.columns[5:]
print(biomarkers)

# first a quick look at the patient and control distribution for one of our biomarkers

biomarker = biomarkers[0]
sns.displot(data=data,  # our dataframe
            x=biomarker,  # name of the the distribution we want to plot
            hue='GROUP',  # the "grouping" variable
            kind='kde')  # kind can also be 'hist' or 'ecdf'
plt.title(biomarker)
plt.show()

# now we perform the normalization

# make a copy of our dataframe (we don't want to overwrite our original data)
zdata = pandas.DataFrame(data, copy=True)

# for each biomarker
for biomarker in biomarkers:
    mod = smf.ols('%s ~ AGE + SEX' % biomarker,  # fit a model finding the effect of age and headsize on biomarker
                  # fit this model *only* to individuals in the control group
                  data=data[data.GROUP == 0]
                  ).fit()  # fit model
    print(mod.summary())
    # get the "predicted" values for all subjects based on the control model parameters
    predicted = mod.predict(data[['AGE', 'SEX', biomarker]])

    # calculate our zscore: observed - predicted / SD of the control group residuals
    w_score = (data.loc[:, biomarker] - predicted) / mod.resid.std()

    # print(np.mean(w_score[data.Diagnosis==0]))
    # print(np.std(w_score[data.Diagnosis==0]))

    # save zscore back into our new (copied) dataframe
    zdata.loc[:, biomarker] = w_score


plt.figure(0)
sns.scatterplot(x=data.AGE, y=data.ROI1, hue=data.GROUP)
plt.figure(1)
sns.scatterplot(x=zdata.AGE, y=zdata.ROI1, hue=zdata.GROUP)

biomarker = biomarkers[0]
sns.displot(data=zdata, x=biomarker, hue='GROUP', kind='kde')
plt.title(biomarker)
# the 0 line *should* be the mean of the control distribution
plt.axvline(0, ls='--', c='black')
plt.show()
N = len(biomarkers)         # number of biomarkers

SuStaInLabels = biomarkers
Z_vals = np.array([[1, 2, 3]]*N)     # Z-scores for each biomarker
Z_max = np.array([5]*N)           # maximum z-score
print(Z_vals)
# Input the settings for z-score SuStaIn
# To make the tutorial run faster I've set
# N_startpoints = 10 and N_iterations_MCMC = int(1e4)
# I recommend using N_startpoints = 25 and
# N_iterations_MCMC = int(1e5) or int(1e6) in general though

N_startpoints = 10
N_S_max = 3
N_iterations_MCMC = int(1e4)
output_folder = os.path.join(output_dir, 'WorkshopOutput')
dataset_name = 'WorkshopOutput'

# Initiate the SuStaIn object
sustain_input = pySuStaIn.ZscoreSustain(
    zdata[biomarkers].values,
    Z_vals,
    Z_max,
    SuStaInLabels,
    N_startpoints,
    N_S_max,
    N_iterations_MCMC,
    output_folder,
    dataset_name,
    False)

# make the output directory if it's not already created
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
samples_sequence,   \
    samples_f,          \
    ml_subtype,         \
    prob_ml_subtype,    \
    ml_stage,           \
    prob_ml_stage,      \
    prob_subtype_stage = sustain_input.run_sustain_algorithm()


# for each subtype model
for s in range(N_S_max):
    # load pickle file (SuStaIn output) and get the sample log likelihood values
    pickle_filename_s = output_folder + '/pickle_files/' + \
        dataset_name + '_subtype' + str(s) + '.pickle'
    pk = pandas.read_pickle(pickle_filename_s)
    samples_likelihood = pk["samples_likelihood"]

    # plot the values as a line plot
    plt.figure(0)
    plt.plot(range(N_iterations_MCMC),
             samples_likelihood, label="subtype" + str(s))
    plt.legend(loc='upper right')
    plt.xlabel('MCMC samples')
    plt.ylabel('Log likelihood')
    plt.title('MCMC trace')

    # plot the values as a histogramp plot
    plt.figure(1)
    plt.hist(samples_likelihood, label="subtype" + str(s))
    plt.legend(loc='upper right')
    plt.xlabel('Log likelihood')
    plt.ylabel('Number of samples')
    plt.title('Histograms of model likelihood')

# Let's plot positional variance diagrams to interpret the subtype progressions

s = 1  # 1 split = 2 subtypes
M = len(zdata)

# get the sample sequences and f
pickle_filename_s = output_folder + '/pickle_files/' + \
    dataset_name + '_subtype' + str(s) + '.pickle'
pk = pandas.read_pickle(pickle_filename_s)
samples_sequence = pk["samples_sequence"]
samples_f = pk["samples_f"]

# use this information to plot the positional variance diagrams
tmp = pySuStaIn.ZscoreSustain._plot_sustain_model(
    sustain_input, samples_sequence, samples_f, M, subtype_order=(0, 1))

# let's take a look at all of the things that exist in SuStaIn's output (pickle) file
pk.keys()
# The SuStaIn output has everything we need. We'll use it to populate our dataframe.

s = 1
pickle_filename_s = output_folder + '/pickle_files/' + \
    dataset_name + '_subtype' + str(s) + '.pickle'
pk = pandas.read_pickle(pickle_filename_s)

for variable in ['ml_subtype',  # the assigned subtype
                 'prob_ml_subtype',  # the probability of the assigned subtype
                 'ml_stage',  # the assigned stage
                 'prob_ml_stage', ]:  # the probability of the assigned stage

    # add SuStaIn output to dataframe
    zdata.loc[:, variable] = pk[variable]

# let's also add the probability for each subject of being each subtype
for i in range(s):
    zdata.loc[:, 'prob_S%s' % i] = pk['prob_subtype'][:, i]
zdata.head()
# IMPORTANT!!! The last thing we need to do is to set all "Stage 0" subtypes to their own subtype
# We'll set current subtype (0 and 1) to 1 and 0, and we'll call "Stage 0" individuals subtype 0.

# make current subtypes (0 and 1) 1 and 2 instead
zdata.loc[:, 'ml_subtype'] = zdata.ml_subtype.values + 1

# convert "Stage 0" subjects to subtype 0
zdata.loc[zdata.ml_stage == 0, 'ml_subtype'] = 0

zdata.ml_subtype.value_counts()

sns.displot(x='ml_stage', hue='Diagnosis', data=zdata, col='ml_subtype')

sns.pointplot(x='ml_stage', y='prob_ml_subtype',  # input variables
              hue='ml_subtype',                 # "grouping" variable
              data=zdata[zdata.ml_subtype > 0])  # only plot for Subtypes 1 and 2 (not 0)
plt.ylim(0, 1)
# plot a line representing change (0.5 in the case of 2 subtypes)
plt.axhline(0.5, ls='--', color='k')

# Plotting relationship between a biomarker and SuStaIn stage across subtypes

var = 'Biomarker3'

# plot relationship
sns.lmplot(x='ml_stage', y=var, hue='ml_subtype',
           data=zdata[zdata.ml_subtype > 0],
           # lowess=True # uncomment if you would prefer a lowess curve to a linear curve
           )

# get stats
for subtype in [1, 2]:
    # get r and p value
    r, p = stats.pearsonr(x=zdata.loc[zdata.ml_subtype == subtype, var].values,
                          y=zdata.loc[zdata.ml_subtype == subtype, 'ml_stage'].values)
    # add them to plot
    plt.text(16, 0-subtype, 'S%s: r = %s, p = %s' %
             (subtype, round(r, 3), round(p, 2)))


results = pandas.DataFrame(index=biomarkers)
for biomarker in biomarkers:
    t, p = stats.ttest_ind(zdata.loc[zdata.ml_subtype == 0, biomarker],
                           zdata.loc[zdata.ml_subtype == 1, biomarker],)
    results.loc[biomarker, 't'] = t
    results.loc[biomarker, 'p'] = p

print(results)
sns.heatmap(pandas.DataFrame(results['t']), square=True, annot=True,
            cmap='RdBu_r')
# plot an example variable:

var = 'Biomarker3'
sns.boxplot(x='ml_subtype', y=var, data=zdata)

# choose the number of folds - here i've used three for speed but i recommend 10 typically
N_folds = 3

# generate stratified cross-validation training and test set splits
labels = zdata.Diagnosis.values
cv = sklearn.model_selection.StratifiedKFold(n_splits=N_folds, shuffle=True)
cv_it = cv.split(zdata, labels)

# SuStaIn currently accepts ragged arrays, which will raise problems in the future.
# We'll have to update this in the future, but this will have to do for now
test_idxs = []
for train, test in cv_it:
    test_idxs.append(test)
test_idxs = np.array(test_idxs, dtype='object')
# perform cross-validation and output the cross-validation information criterion and
# log-likelihood on the test set for each subtypes model and fold combination
CVIC, loglike_matrix = sustain_input.cross_validate_sustain_model(test_idxs)
# go through each subtypes model and plot the log-likelihood on the test set and the CVIC
print("CVIC for each subtype model: " + str(CVIC))
print("Average test set log-likelihood for each subtype model: " +
      str(np.mean(loglike_matrix, 0)))

plt.figure(0)
plt.plot(np.arange(N_S_max, dtype=int), CVIC)
plt.xticks(np.arange(N_S_max, dtype=int))
plt.ylabel('CVIC')
plt.xlabel('Subtypes model')
plt.title('CVIC')

plt.figure(1)
df_loglike = pandas.DataFrame(data=loglike_matrix, columns=[
                              "s_" + str(i) for i in range(sustain_input.N_S_max)])
df_loglike.boxplot(grid=False)
plt.ylabel('Log likelihood')
plt.xlabel('Subtypes model')
plt.title('Test set log-likelihood across folds')

# this part estimates cross-validated positional variance diagrams
for i in range(N_S_max):
    sustain_input.combine_cross_validated_sequences(i+1, N_folds)

N_S_selected = 2

pySuStaIn.ZscoreSustain._plot_sustain_model(
    sustain_input, samples_sequence, samples_f, M, subtype_order=(0, 1))
_ = plt.suptitle('SuStaIn output')

sustain_input.combine_cross_validated_sequences(N_S_selected, N_folds)
_ = plt.suptitle('Cross-validated SuStaIn output')
