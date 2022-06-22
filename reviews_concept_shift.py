from quapy.data.reader import from_text
from quapy.data.base import LabelledCollection
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from quapy.protocol import NPP
from concept_shift_protocol import ConceptShiftProtocol
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

def bias(true,prev):
    return true[1]-prev[1]


#Configuration
base_path = "/media/nas/pgonzalez/quant_datasetshift/datasets/reviews"
training_sample_size = 5000
test_sample_size = 500
n_test_samples = 50
n_reps_train = 10
error_function = bias
seed = 2032

#set numpy seed
np.random.seed(seed)

qp.environ['SAMPLE_SIZE'] = test_sample_size
qp.environ['N_JOBS'] = 12

domainA, domainB = load_data()

fulldataset = domainA+domainB

train, test = fulldataset.split_stratified(train_prop=0.6, random_state=seed)
test = test.uniform_sampling(50000, random_state=seed)

print("dataset",fulldataset.stats(show=False))
print("train",train.stats(show=False))
print("test",test.stats(show=False))

quant_methods = {
    "CC":qp.method.aggregative.CC(LogisticRegression(max_iter=1000)),
    "PCC":qp.method.aggregative.PCC(LogisticRegression(max_iter=1000)),
    "ACC":qp.method.aggregative.ACC(LogisticRegression(max_iter=1000)),
    "PACC":qp.method.aggregative.ACC(LogisticRegression(max_iter=1000)),
    "HDy":qp.method.aggregative.HDy(LogisticRegression(max_iter=1000)),
    "EMQ":qp.method.aggregative.EMQ(CalibratedClassifierCV(LogisticRegression(max_iter=1000),n_jobs=-1)),
    "MLPE":qp.method.non_aggregative.MaximumLikelihoodPrevalenceEstimation()
}

param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000],'class_weight': ['balanced', None]}
cut_points = [2,3,4]

#We want to generate bags with concept shift for training
trainSampleGenerator = ConceptShiftProtocol(train, sample_size = training_sample_size, cut_points=cut_points, return_type="labelled_collection",repeats=n_reps_train,random_state=seed)

experiment_results = {}
for method_name in quant_methods.keys():
    experiment_results[method_name] = pd.DataFrame(columns=["cut_point_train","cut_point_test","train_rep","test_sample","error"])

for n_training_sample, training_sample in enumerate(trainSampleGenerator()):
    i_cut_point_train = n_training_sample // n_reps_train
    rep = n_training_sample % n_reps_train

    print("%d/%d Concept shift: cut_point_train=%d. Rep: %d" % (n_training_sample+1,n_reps_train*len(cut_points),cut_points[i_cut_point_train],rep))
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
        grids[quant_name] = qp.model_selection.GridSearchQ(quantifier,param_grid=param_grid,protocol=NPP(valsplit,sample_size=n_test_samples, random_state=seed),refit=True,verbose=False).fit(trainsplit)
    print("Done. Evaluating...")   
    for quant_name, quantifier in quant_methods.items():
        print("Evaluating quantifier %s" % quant_name)
        testSampleGenerator = ConceptShiftProtocol(testtranformed, sample_size = test_sample_size,cut_points=cut_points,prevalence=trainset.prevalence(),repeats=n_test_samples, random_state=seed)
        for n_test_sample, test_sample in enumerate(testSampleGenerator()):
            i_cut_point_test = n_test_sample // n_test_samples
            n_test_sample = n_test_sample % n_test_samples
            preds = grids[quant_name].quantify(test_sample[0])
            true = test_sample[1]
            error = error_function(true,preds)
            experiment_results[quant_name] = experiment_results[quant_name].append([{
                                                    'cut_point_train':cut_points[i_cut_point_train],
                                                    'cut_point_test':cut_points[i_cut_point_test],
                                                    'train_rep':rep,
                                                    'test_sample':n_test_sample,
                                                    'error':error}],ignore_index=True)


for quant_name, quantifier in quant_methods.items():
    #add date to file name
    date_string = f'{datetime.now():%Y_%m_%d_%H_%M}'
    #save pandas dataframe
    experiment_results[quant_name].to_csv("results/concept/results_%s_%s.csv" % (date_string,quant_name))


