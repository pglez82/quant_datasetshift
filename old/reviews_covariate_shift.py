from quapy.data.reader import from_text
from quapy.data.base import LabelledCollection
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import quapy as qp
import numpy as np
from quapy.protocol import CovariateShiftPP,NPP
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from datetime import datetime
from utils.pcc_weighted import PCCWeighted
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

def create_quant_methods(max_iter):
    return {
        "CC":qp.method.aggregative.CC(LogisticRegression(max_iter=max_iter)),
        "PCC":qp.method.aggregative.PCC(LogisticRegression(max_iter=max_iter)),
        "ACC":qp.method.aggregative.ACC(LogisticRegression(max_iter=max_iter), val_split=5, n_jobs=-1),
        "PACC":qp.method.aggregative.PACC(LogisticRegression(max_iter=max_iter), val_split=5, n_jobs=-1),
        "HDy":qp.method.aggregative.HDy(LogisticRegression(max_iter=max_iter)),
        "DyS":qp.method.aggregative.DyS(LogisticRegression(max_iter=max_iter), distance='topsoe',n_bins=10),
        "SMM":qp.method.aggregative.SMM(LogisticRegression(max_iter=max_iter)),
        "SLD":qp.method.aggregative.EMQ(CalibratedClassifierCV(LogisticRegression(max_iter=max_iter),n_jobs=-1)),
        "PCCW":PCCWeighted(
                        LogisticRegression(max_iter=max_iter),
                        qp.method.aggregative.EMQ(CalibratedClassifierCV(LogisticRegression(max_iter=max_iter),n_jobs=-1)),
                        param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000],'class_weight': ['balanced', None]}
                        ),
        "MLPE":qp.method.non_aggregative.MaximumLikelihoodPrevalenceEstimation()
    }


#Configuration
base_path = "/media/nas/pgonzalez/quant_datasetshift/datasets/reviews"
training_sample_size = 5000
test_sample_size = 500
n_test_samples = 50
n_reps_train = 10
mix_points = 11
ps_train = [0.25,0.5,0.75]
ps_test = [0.25, 0.5, 0.75]
error_function = qp.error.mae
max_iter = 1000
seed = 2032


#set numpy seed
np.random.seed(seed)

qp.environ['SAMPLE_SIZE'] = test_sample_size
qp.environ['N_JOBS'] = 14

with qp.util.temp_seed(seed):

    domainA, domainB = load_data()

    domainA = binarize_dataset(domainA)
    domainB = binarize_dataset(domainB)

    domainA_train, domainA_test = domainA.split_stratified(train_prop=0.6, random_state=seed)
    domainB_train, domainB_test = domainB.split_stratified(train_prop=0.6, random_state=seed)

    print("domainA_train",domainA_train.stats(show=False))
    print("domainB_train",domainB_train.stats(show=False))
    print("domainA_test",domainA_test.stats(show=False))
    print("domainB_test",domainB_test.stats(show=False))

    #provisional, subset the test dataset to make things faster
    domainA_test = domainA_test.uniform_sampling(50000, random_state=seed)
    domainB_test = domainB_test.uniform_sampling(50000, random_state=seed)

    print("domainA_test_reduced",domainA_test.stats(show=False))
    print("domainB_test_reduced",domainB_test.stats(show=False))

    mixture_points = np.linspace(0, 1, mix_points)[::-1]
    param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000],'class_weight': ['balanced', None]}

    experiment_results = {}
    quant_methods = create_quant_methods(max_iter)
    for method_name in quant_methods.keys():
        experiment_results[method_name] = pd.DataFrame(columns=["domainA_prop_train","domainA_prop_test","train_rep","test_sample","p_train","p_test","p_hat","error"])

    for n_p_train, p_train in enumerate(ps_train):
        #We want to generate bags with covariate shift for training
        trainSampleGenerator = CovariateShiftPP(domainA_train, domainB_train, prevalence=(1-p_train,p_train), sample_size = training_sample_size, return_type="labelled_collection",mixture_points=mixture_points,repeats=n_reps_train,random_state=seed)
        print("Prevalence of training samples: ")
        print(trainSampleGenerator.prevalence)
        
        for n_training_sample, training_sample in enumerate(trainSampleGenerator()):
            rep = n_training_sample % n_reps_train
            i_covariate_shift_train = n_training_sample // n_reps_train
            mixture_point_train = trainSampleGenerator.mixture_points[i_covariate_shift_train]

            print("%d/%d Covariate shift: DomainA: %f DomainB: %f. Rep: %d" % ((n_p_train*n_reps_train)+n_training_sample,n_reps_train*len(ps_train)*len(mixture_points),mixture_point_train, 1-mixture_point_train,rep))
            print("Tkfid for training...")
            vectorizer = TfidfVectorizer(min_df=3, sublinear_tf=True)
            vec_documents = vectorizer.fit_transform(training_sample.X)
            print("Transforming test set with same tkfid...")
            trainset = LabelledCollection(vec_documents, training_sample.y)
            trainsplit, valsplit = trainset.split_stratified(train_prop=0.6, random_state=seed)
            testA = LabelledCollection(vectorizer.transform(domainA_test.X),domainA_test.y)
            testB = LabelledCollection(vectorizer.transform(domainB_test.X),domainB_test.y)
            print("Done. Fitting quantification methods...")
            grids = {}
            for quant_name, quantifier in quant_methods.items():
                if quant_name != "PCCW":
                    grids[quant_name] = qp.model_selection.GridSearchQ(quantifier,param_grid=param_grid,protocol=NPP(valsplit,sample_size=n_test_samples, random_state=seed),refit=True,verbose=False).fit(trainsplit)
                else:
                    grids[quant_name] = quantifier.fit(trainset)
            print("Done. Evaluating...")
            for p_test in ps_test:
                for quant_name, quantifier in quant_methods.items():
                    print("Evaluating quantifier %s with test prevalence %f" % (quant_name,p_test))
                    testSampleGenerator = CovariateShiftPP(testA, testB, sample_size = test_sample_size, mixture_points=mixture_points, prevalence=(1-p_test,p_test), repeats=n_test_samples, random_state=seed)
                    for n_test_sample, test_sample in enumerate(testSampleGenerator()):
                        i_covariate_shift_test = n_test_sample // n_test_samples
                        n_test_sample = n_test_sample % n_test_samples
                        mixture_point_test = testSampleGenerator.mixture_points[i_covariate_shift_test]
                        preds = grids[quant_name].quantify(test_sample[0])
                        true = test_sample[1] 
                        error = error_function(true,preds)
                        experiment_results[quant_name] = experiment_results[quant_name].append([{'domainA_prop_train':mixture_point_train,
                                                                'domainA_prop_test':mixture_point_test,
                                                                'train_rep':rep,
                                                                'test_sample':n_test_sample,
                                                                'p_train':p_train,
                                                                'p_test':p_test,
                                                                'p_hat':preds,
                                                                'error':error}],ignore_index=True)
            quant_methods = create_quant_methods(max_iter)

    date_string = f'{datetime.now():%Y_%m_%d_%H_%M}'
    for quant_name, quantifier in quant_methods.items():
        #save pandas dataframe
        experiment_results[quant_name].to_csv("results/covariate/results_%s_%s.csv" % (date_string,quant_name))


