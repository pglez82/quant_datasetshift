import quapy as qp
from quapy.method.aggregative import PCC
from sklearn.base import BaseEstimator
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
from quapy.protocol import APP

class PCCWeighted(PCC):
    """
    `Probabilistic Classify & Count <https://ieeexplore.ieee.org/abstract/document/5694031>`_,
    the probabilistic variant of CC that relies on the posterior probabilities returned by a probabilistic classifier.

    In this case this a modification of the algorithm that retrains the classifier based on the prevalences of the test set.
    Estimated prevalences for the test set are computed using another quantifier.

    :param learner: a sklearn's Estimator that generates a classifier
    :estim_quantifier: quantifier to use for estimating the prevalences of the test set
    :param_grid: hyperparameters to use in the estim_quantifier. Can be none.
    """

    def __init__(self, learner: BaseEstimator, estim_quantifier: BaseQuantifier, param_grid = None):
        super(PCCWeighted, self).__init__(learner=learner)
        self.estim_quantifier = estim_quantifier
        self.param_grid = param_grid

    def fit(self, data: LabelledCollection):
        #Base class fit is not called as it is delayed to test
        self.training_data = data
        if self.param_grid is not None:
            trainsplit, valsplit = data.split_stratified(train_prop=0.6, random_state=2032)
            self.estim_quantifier=qp.model_selection.GridSearchQ(self.estim_quantifier,param_grid=self.param_grid,protocol=APP(valsplit,sample_size=500, random_state=2032),refit=True,verbose=False).fit(trainsplit)
        else:
            self.estim_quantifier.fit(data)
        self.prevalences = data.prevalence()
        return self

    def quantify(self, instances):
        #Before quantifying we need to retrain the learner with the weights computed by the base quantifier
        p_hat = self.estim_quantifier.quantify(instances)
        self.learner.class_weight = {0: p_hat[0]/self.prevalences[0],1: p_hat[1]/self.prevalences[1]}
        #print("Class weights: ", self.learner.class_weight)
        super().fit(self.training_data,fit_learner=True)
        return super().quantify(instances)



