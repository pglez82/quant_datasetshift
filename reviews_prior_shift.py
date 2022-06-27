from quapy.data.reader import from_text
from quapy.data.base import LabelledCollection
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from quapy.protocol import APP
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import quapy as qp
import numpy as np
from datetime import datetime
import os

def load_data():
    domainA = LabelledCollection.load(os.path.join(base_path, "Electronics.txt"), from_text)
    domainB = LabelledCollection.load(os.path.join(base_path, "Books.txt"), from_text)
    return domainA, domainB

def binarize_dataset(dataset):
    instances = dataset.X
    labels = dataset.y
    labels, instances = labels[labels!=3], instances[labels!=3]
    labels[labels<3] = 0
    labels[labels>3] = 1
    return LabelledCollection(instances, labels)

#Configuration
base_path = "/media/nas/pgonzalez/quant_datasetshift/datasets/reviews"
training_sample_size = 5000
test_sample_size = 500
n_test_samples = 50
n_reps_train = 10
n_prevalences = 11
error_function = qp.error.mae
seed = 2032

#set numpy seed
np.random.seed(seed)

qp.environ['SAMPLE_SIZE'] = test_sample_size
qp.environ['N_JOBS'] = 12

domainA, domainB = load_data()

fulldataset = domainA+domainB
fulldataset = binarize_dataset(fulldataset)

train, test = fulldataset.split_stratified(train_prop=0.6, random_state=seed)
test = test.uniform_sampling(50000, random_state=seed)

print("dataset",fulldataset.stats(show=False))
print("train",train.stats(show=False))
print("test",test.stats(show=False))


quant_methods = {
    "CC":qp.method.aggregative.CC(LogisticRegression(max_iter=1000)),
    "PCC":qp.method.aggregative.PCC(LogisticRegression(max_iter=1000)),
    "ACC":qp.method.aggregative.ACC(LogisticRegression(max_iter=1000), val_split=5, n_jobs=-1),
    "PACC":qp.method.aggregative.PACC(LogisticRegression(max_iter=1000), val_split=5, n_jobs=-1),
    "HDy":qp.method.aggregative.HDy(LogisticRegression(max_iter=1000)),
    "EMQ":qp.method.aggregative.EMQ(CalibratedClassifierCV(LogisticRegression(max_iter=1000),n_jobs=-1)),
    "MLPE":qp.method.non_aggregative.MaximumLikelihoodPrevalenceEstimation()
}

param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000],'class_weight': ['balanced', None]}

#We want to generate bags with covariate shift for training
trainSampleGenerator = APP(train, sample_size = training_sample_size,n_prevalences=11,return_type="labelled_collection",repeats=n_reps_train,random_state=seed,smooth_limits_epsilon=0.01)

experiment_results = {}
for method_name in quant_methods.keys():
    experiment_results[method_name] = pd.DataFrame(columns=["p_train","p_test","train_rep","test_sample","error"])

for n_training_sample, training_sample in enumerate(trainSampleGenerator()):
    rep = n_training_sample % n_reps_train

    print("%d/%d Prior shift: p_train=%f. Rep: %d" % (n_training_sample+1,n_reps_train*n_prevalences,training_sample.prevalence()[1],rep))
    print("Tkfid for training...")
    vectorizer = TfidfVectorizer(min_df=3, sublinear_tf=True)
    vec_documents = vectorizer.fit_transform(training_sample.X)
    print("Transforming test set with same tkfid...")
    trainset = LabelledCollection(vec_documents, training_sample.y)
    trainsplit, valsplit = trainset.split_stratified(train_prop=0.6, random_state=seed)
    testtranformed = LabelledCollection(vectorizer.transform(test.X),test.y)
    print("Done. Fitting quantification methods...")
    grids = {}
    for quant_name, quantifier in quant_methods.items():
        grids[quant_name] = qp.model_selection.GridSearchQ(quantifier,param_grid=param_grid,protocol=APP(valsplit,sample_size=test_sample_size,n_prevalences=n_prevalences, smooth_limits_epsilon=0.01, random_state=seed),refit=True,verbose=False).fit(trainsplit)
    print("Done. Evaluating...")   
    for quant_name, quantifier in quant_methods.items():
        print("Evaluating quantifier %s" % quant_name)
        testSampleGenerator = APP(testtranformed, sample_size = test_sample_size,n_prevalences=n_prevalences, repeats=n_test_samples, smooth_limits_epsilon=0.01, random_state=seed)
        for n_test_sample, test_sample in enumerate(testSampleGenerator()):
            n_test_sample = n_test_sample % n_test_samples
            preds = grids[quant_name].quantify(test_sample[0])
            true = test_sample[1]
            error = error_function(true,preds)
            experiment_results[quant_name] = experiment_results[quant_name].append([{
                                                    'p_train':trainset.p[1],
                                                    'p_test':test_sample[1][1],
                                                    'train_rep':rep,
                                                    'test_sample':n_test_sample,
                                                    'error':error}],ignore_index=True)


for quant_name, quantifier in quant_methods.items():
    #add date to file name
    date_string = f'{datetime.now():%Y_%m_%d_%H_%M}'
    #save pandas dataframe
    experiment_results[quant_name].to_csv("results/prior/results_%s_%s.csv" % (date_string,quant_name))


