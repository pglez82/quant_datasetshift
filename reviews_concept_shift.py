from quapy.data.reader import from_text
from quapy.data.base import LabelledCollection
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from quapy.protocol import NPP
from utils.concept_shift_protocol import ConceptShiftProtocolV2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import quapy as qp
import numpy as np
from datetime import datetime
import os

def load_data(base_path):
    domainA = LabelledCollection.load(os.path.join(base_path, "Electronics.txt"), from_text)
    domainB = LabelledCollection.load(os.path.join(base_path, "Books.txt"), from_text)
    #Clean the dataset as there are a couple of opinions with 0 stars
    domainB = LabelledCollection(domainB.X[domainB.y!=0], domainB.y[domainB.y!=0])
    return domainA, domainB

def bias(true,pred):
    return pred[1]-true[1]

def create_quant_methods(max_iter):
    return {
        "CC":qp.method.aggregative.CC(LogisticRegression(max_iter=max_iter)),
        "PCC":qp.method.aggregative.PCC(LogisticRegression(max_iter=max_iter)),
        "ACC":qp.method.aggregative.ACC(LogisticRegression(max_iter=max_iter), val_split=5, n_jobs=-1),
        "PACC":qp.method.aggregative.PACC(LogisticRegression(max_iter=max_iter), val_split=5, n_jobs=-1),
        "DyS":qp.method.aggregative.DyS(LogisticRegression(max_iter=max_iter), distance='topsoe',n_bins=10),
        "SLD":qp.method.aggregative.EMQ(CalibratedClassifierCV(LogisticRegression(max_iter=max_iter),n_jobs=-1)),
        "MLPE":qp.method.non_aggregative.MaximumLikelihoodPrevalenceEstimation()
    }


#Configuration
base_path = "/media/nas/pgonzalez/quant_datasetshift/datasets/reviews"
training_sample_size = 5000
test_sample_size = 500
n_test_samples = 50
n_reps_train = 10
error_function = bias
max_iter = 1000
seed = 2032

#set numpy seed
np.random.seed(seed)

qp.environ['SAMPLE_SIZE'] = test_sample_size
qp.environ['N_JOBS'] = 14

with qp.util.temp_seed(seed):

    domainA, domainB = load_data(base_path)

    fulldataset = domainA+domainB

    #sampling in this case is a little bit different as we want to get same number of opinions for each number of stars
    fulldataset = fulldataset.sampling(1000000,0.2,0.2,0.2,0.2,0.2)

    train, test = fulldataset.split_stratified(train_prop=0.6, random_state=seed)
    test = test.uniform_sampling(100000, random_state=seed)

    print("dataset",fulldataset.stats(show=False))
    print("train",train.stats(show=False))
    print("test",test.stats(show=False))

    param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000],'class_weight': ['balanced', None]}
    cut_points = [2, 3, 4, 5] #the examples which class matches each cut point are considered positives

    experiment_results = {}
    quant_methods = create_quant_methods(max_iter)
    for method_name in quant_methods.keys():
        experiment_results[method_name] = pd.DataFrame(columns=["cut_point_train","cut_point_test","train_rep","test_sample","p_train","p_test","p_hat","error"])

    trainSampleGenerator = ConceptShiftProtocolV2(train, sample_size = training_sample_size, cut_points=cut_points, return_type="labelled_collection",repeats=n_reps_train,random_state=seed)
    for n_training_sample, training_sample in enumerate(trainSampleGenerator()):
        i_cut_point_train = n_training_sample // n_reps_train
        rep = n_training_sample % n_reps_train
        p_train = training_sample.p[1]
        print("%d/%d Concept shift: cut_point_train=%d. p_train=%.2f. Rep: %d" % (i_cut_point_train,len(cut_points),cut_points[i_cut_point_train],p_train,rep))
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
            testSampleGenerator = ConceptShiftProtocolV2(testtranformed, sample_size = test_sample_size,cut_points=cut_points,repeats=n_test_samples, random_state=seed)
            for n_test_sample, test_sample in enumerate(testSampleGenerator()):
                i_cut_point_test = n_test_sample // n_test_samples
                p_test = test_sample[1][1]
                #print("Evaluating quantifier %s with cut point %d and test prevalence %.2f" % (quant_name,cut_points[i_cut_point_test],p_test))
                n_test_sample = n_test_sample % n_test_samples
                preds = grids[quant_name].quantify(test_sample[0])
                true = test_sample[1]
                error = error_function(true,preds)
                experiment_results[quant_name] = experiment_results[quant_name].append([{
                                                        'cut_point_train':cut_points[i_cut_point_train],
                                                        'cut_point_test':cut_points[i_cut_point_test],
                                                        'train_rep':rep,
                                                        'test_sample':n_test_sample,
                                                        'p_test':p_test,
                                                        'p_train':trainset.p[1],
                                                        'p_hat':preds,
                                                        'error':error}],ignore_index=True)

        quant_methods = create_quant_methods(max_iter)

    #add date to file name
    date_string = f'{datetime.now():%Y_%m_%d_%H_%M}'
    for quant_name, quantifier in quant_methods.items():
        #save pandas dataframe
        experiment_results[quant_name].to_csv("results/concept/results_v2_%s_%s.csv" % (date_string,quant_name))


