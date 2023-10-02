from quapy.data.reader import from_text
from quapy.data.base import LabelledCollection
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from quapy.protocol import APP
from utils.changeprior_protocol import ChangePriorProtocolV2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import quapy as qp
import numpy as np
from datetime import datetime
import os

def load_data():
    domainA_1 = LabelledCollection.load(os.path.join(base_path, "Electronics.txt"), from_text)
    domainA_2 = LabelledCollection.load(os.path.join(base_path, "Home_and_Kitchen.txt"), from_text)
    domainB_1 = LabelledCollection.load(os.path.join(base_path, "Books.txt"), from_text)
    domainB_2 = LabelledCollection.load(os.path.join(base_path, "Movies_and_TV.txt"), from_text)
    #First two domains are the class 0, and last two domains are class 1
    domainA_1 = LabelledCollection(domainA_1.X, np.zeros((len(domainA_1))))
    domainA_2 = LabelledCollection(domainA_2.X, np.zeros((len(domainA_2))))
    domainB_1 = LabelledCollection(domainB_1.X, np.ones((len(domainB_1))))
    domainB_2 = LabelledCollection(domainB_2.X, np.ones((len(domainB_2))))
    return domainA_1, domainA_2, domainB_1, domainB_2

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
        "MLPE":qp.method.non_aggregative.MaximumLikelihoodPrevalenceEstimation()
    }

#Configuration
base_path = "/media/nas/pgonzalez/quant_datasetshift/datasets/reviews"
training_sample_size = 5000
test_sample_size = 500
n_test_samples = 50
n_reps_train = 10
max_iter = 1000
error_function = qp.error.mae
seed = 2032

#set numpy seed
np.random.seed(seed)

qp.environ['SAMPLE_SIZE'] = test_sample_size
qp.environ['N_JOBS'] = 14

with qp.util.temp_seed(seed):
    #Now we are going to have two classes, 'house and electronics' ->domainA and 'movies and books' -> domainB
    domainA_1, domainA_2, domainB_1, domainB_2 = load_data()

    domainA_1_train, domainA_1_test = domainA_1.split_stratified(train_prop=0.6, random_state=seed)
    domainA_2_train, domainA_2_test = domainA_2.split_stratified(train_prop=0.6, random_state=seed)
    domainB_1_train, domainB_1_test = domainB_1.split_stratified(train_prop=0.6, random_state=seed)
    domainB_2_train, domainB_2_test = domainB_2.split_stratified(train_prop=0.6, random_state=seed)

    domainA_1_test = domainA_1_test.uniform_sampling(50000, random_state=seed)
    domainA_2_test = domainA_2_test.uniform_sampling(50000, random_state=seed)
    domainB_1_test = domainB_1_test.uniform_sampling(50000, random_state=seed)
    domainB_2_test = domainB_2_test.uniform_sampling(50000, random_state=seed)

    param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000],'class_weight': ['balanced', None]}

    #We want to generate bags with covariate shift for training
    trainSampleGenerator = ChangePriorProtocolV2(domainA_1=domainA_1_train, domainA_2=domainA_2_train, domainB_1=domainB_1_train,domainB_2=domainB_2_train,change_conditionals=False, sample_size = training_sample_size,prevalences=[0.5], return_type="labelled_collection",repeats=n_reps_train,random_state=seed)

    experiment_results = {}
    quant_methods = create_quant_methods(max_iter)
    for method_name in quant_methods.keys():
        experiment_results[method_name] = pd.DataFrame(columns=["p_train","p_test","n_dA_2","change_conditionals","train_rep","test_sample","t_sample_size","p_hat","error"])

    for n_training_sample, training_sample in enumerate(trainSampleGenerator()):
        rep = n_training_sample % n_reps_train

        print("%d/%d Prior shift: p_train=%f. Rep: %d" % (n_training_sample+1,n_reps_train,training_sample.prevalence()[1],rep))
        print("Tkfid for training...")
        vectorizer = TfidfVectorizer(min_df=3, sublinear_tf=True)
        vec_documents = vectorizer.fit_transform(training_sample.X)
        print("Transforming test set with same tkfid...")
        trainset = LabelledCollection(vec_documents, training_sample.y)
        trainsplit, valsplit = trainset.split_stratified(train_prop=0.6, random_state=seed)
        testA_1 = LabelledCollection(vectorizer.transform(domainA_1_test.X),domainA_1_test.y)
        testA_2 = LabelledCollection(vectorizer.transform(domainA_2_test.X),domainA_2_test.y)
        testB_1 = LabelledCollection(vectorizer.transform(domainB_1_test.X),domainB_1_test.y)
        testB_2 = LabelledCollection(vectorizer.transform(domainB_2_test.X),domainB_2_test.y)
        print("Done. Fitting quantification methods...")
        grids = {}
        for quant_name, quantifier in quant_methods.items():
            grids[quant_name] = qp.model_selection.GridSearchQ(quantifier,param_grid=param_grid,protocol=APP(valsplit,sample_size=test_sample_size,n_prevalences=11, random_state=seed),refit=True,verbose=False).fit(trainsplit)
        print("Done. Evaluating...")   
        for quant_name, quantifier in quant_methods.items():
            print("Evaluating quantifier %s" % quant_name)
            testSampleGenerator = ChangePriorProtocolV2(domainA_1=testA_1,domainA_2=testA_2,domainB_1=testB_1,domainB_2=testB_2,prevalences=np.linspace(0.25,0.75,11), change_conditionals = True, sample_size = test_sample_size,repeats=n_test_samples, random_state=seed)
            for n_test_sample, test_sample in enumerate(testSampleGenerator()):
                n_p_value = n_test_sample//n_test_samples
                n_test_sample = n_test_sample % n_test_samples
                #print("p_test = %f, n_dA_pos = %d" % (test_sample[1][1],testSampleGenerator.n_dA_pos[n_p_value]))
                preds = grids[quant_name].quantify(test_sample[0])
                true = test_sample[1]
                error = error_function(true,preds)
                experiment_results[quant_name] = experiment_results[quant_name].append([{
                                                        'p_train':trainset.p[1],
                                                        'p_test':test_sample[1][1],
                                                        'train_rep':rep,
                                                        'change_conditionals':True,
                                                        'n_dA_2':testSampleGenerator.n_dA_2[n_p_value],
                                                        'test_sample':n_test_sample,
                                                        't_sample_size':test_sample[0].shape[0],
                                                        'p_hat':preds,
                                                        'error':error}],ignore_index=True)
            testSampleGenerator = ChangePriorProtocolV2(domainA_1=testA_1,domainA_2=testA_2,domainB_1=testB_1,domainB_2=testB_2,prevalences=np.linspace(0.25,0.75,11), change_conditionals = False, sample_size = test_sample_size,repeats=n_test_samples, random_state=seed)
            for n_test_sample, test_sample in enumerate(testSampleGenerator()):
                n_p_value = n_test_sample//n_test_samples
                n_test_sample = n_test_sample % n_test_samples
                #print("p_test = %f, n_dA_pos = %d" % (test_sample[1][1],testSampleGenerator.n_dA_pos[n_p_value]))
                preds = grids[quant_name].quantify(test_sample[0])
                true = test_sample[1]
                error = error_function(true,preds)
                experiment_results[quant_name] = experiment_results[quant_name].append([{
                                                        'p_train':trainset.p[1],
                                                        'p_test':test_sample[1][1],
                                                        'train_rep':rep,
                                                        'change_conditionals':False,
                                                        'n_dA_2':testSampleGenerator.n_dA_2[n_p_value],
                                                        'test_sample':n_test_sample,
                                                        't_sample_size':test_sample[0].shape[0],
                                                        'p_hat':preds,
                                                        'error':error}],ignore_index=True)
            
                                                        
        quant_methods = create_quant_methods(max_iter)

    #add date to file name
    date_string = f'{datetime.now():%Y_%m_%d_%H_%M}'
    for quant_name, quantifier in quant_methods.items():
        #save pandas dataframe
        experiment_results[quant_name].to_csv("results/changeprior/results_v2_%s_%s.csv" % (date_string,quant_name))


