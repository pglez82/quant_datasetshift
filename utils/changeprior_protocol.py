from quapy.protocol import AbstractStochasticSeededProtocol, OnLabelledCollectionProtocol
from quapy.data import LabelledCollection
import quapy.functional as F
import numpy as np

class ChangePriorProtocol(AbstractStochasticSeededProtocol):
    """
    Generate samples with a change just in the prior, not mantaining P(X|Y). 
    For that we need two domains of the data. We start from an unbalanced situation
    where postive examples are 1/3 domainB, 2/3 domainA, and negatives, 2/3 domainB
    and 1/3 domainA. This way, we can achieve a bigger change in P(Y) changing only
    the domainA positives.

    :param data: LabelledCollection with the data
    :param sample_size: desired size for the samples. This will only match with p=0.5
    :param cut_points: array with the cut points to use to generate concept shift
    :param repeats: number of repetitions (number of samples for each cut point)
    :param prevalences: list of prevalences to generate.
    :param change_conditionals: boolean indicating if we want a change in P(X|Y) or not
        If false, both domains in the positives keep training proportions. Sample
        size is the same than with change_conditionals=False
    :param random_state: random state to make results reproducible
    """

    def __init__(
            self,
            domainA: LabelledCollection,
            domainB: LabelledCollection,
            sample_size,
            prevalences,
            change_conditionals = True,
            repeats=1,
            random_state=None,
            return_type='sample_prev'):
        super(ChangePriorProtocol, self).__init__(random_state)
        self.domainA = domainA
        self.domainB = domainB
        self.sample_size = sample_size
        self.repeats = repeats
        self.prevalences = prevalences
        self.change_conditionals = change_conditionals
        self.random_state = random_state
        self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

    def samples_parameters(self):
        indexesA = []
        indexesB = []

        #store the number of positives computed for each prevalence in case is needed
        self.n_dA_pos=np.zeros((len(self.prevalences)),dtype=int)
        self.n_dA_neg=np.zeros((len(self.prevalences)),dtype=int)
        self.n_dB_pos=np.zeros((len(self.prevalences)),dtype=int)
        self.n_dB_neg=np.zeros((len(self.prevalences)),dtype=int)

        n_dA_neg = n_dB_pos = self.sample_size//6
        n_dB_neg = self.sample_size//3

        dA_pos_ex = np.where(self.domainA.y==1)[0]
        dA_neg_ex = np.where(self.domainA.y==0)[0]
        dB_pos_ex = np.where(self.domainB.y==1)[0]
        dB_neg_ex = np.where(self.domainB.y==0)[0]

        for i, p in enumerate(self.prevalences):
            n_dA_pos = round((n_dB_pos-p*(n_dB_pos+n_dA_neg+n_dB_neg))/(p-1))

            if not self.change_conditionals:
                #In this case the idea is to generate a normal sample, but keeping the size of the sample equal to the one if we have a class conditional change
                new_sample_size = round((n_dB_pos-p*(n_dB_pos+n_dA_neg+n_dB_neg))/(p-1))+n_dA_neg+n_dB_pos+n_dB_neg
                n_pos = round(new_sample_size * p)
                n_neg = new_sample_size-n_pos
                n_dA_pos = n_pos*2//3
                n_dB_pos = n_pos//3
                n_dA_neg = n_neg//3
                n_dB_neg = n_neg*2//3
            #store it
            self.n_dA_pos[i] = n_dA_pos
            self.n_dA_neg[i] = n_dA_neg
            self.n_dB_pos[i] = n_dB_pos
            self.n_dB_neg[i] = n_dB_neg


            #print("[p=%.2f,classcond_change=%s] n_dA_neg=%d,n_dB_neg=%d,n_dA_pos=%d,n_dB_pos=%d" % (p,self.change_conditionals,n_dA_neg,n_dB_neg,n_dA_pos,n_dB_pos))

            for _ in range(self.repeats):
                dA_pos = np.random.choice(dA_pos_ex, n_dA_pos, replace=False)
                dA_neg = np.random.choice(dA_neg_ex, n_dA_neg, replace=False)
                dB_pos = np.random.choice(dB_pos_ex, n_dB_pos, replace=False)
                dB_neg = np.random.choice(dB_neg_ex, n_dB_neg, replace=False)

                idxsA = np.concatenate((dA_pos,dA_neg))
                idxsB = np.concatenate((dB_pos,dB_neg))
                indexesA.append(idxsA)
                indexesB.append(idxsB)
            
        return list(zip(indexesA, indexesB))

    def sample(self, indexes):
        indexesA, indexesB = indexes
        sampleA = self.domainA.sampling_from_index(indexesA)
        sampleB = self.domainB.sampling_from_index(indexesB)
        return sampleA+sampleB

    def total(self):
        return self.repeats * len(self.prevalences)

# class ChangePriorProtocol2(AbstractStochasticSeededProtocol):
#     """
#     Generate samples with a change just in the prior, not mantaining P(X|Y). For that we need two domains of the data.
#     The number of examples in one domain is going to be altered just for one class

#     :param data: LabelledCollection with the data
#     :param sample_size: desired size for the samples. This will only match with p=0.5
#     :param cut_points: array with the cut points to use to generate concept shift
#     :param repeats: number of repetitions (number of samples for each cut point)
#     :param prevalences: list of prevalences to generate.
#     :param random_state: random state to make results reproducible
#     """

#     def __init__(
#             self,
#             domainA: LabelledCollection,
#             domainB: LabelledCollection,
#             sample_size,
#             prevalences,
#             change_conditionals = True,
#             repeats=1,
#             random_state=None,
#             return_type='sample_prev'):
#         super(ChangePriorProtocol2, self).__init__(random_state)
#         self.domainA = domainA
#         self.domainB = domainB
#         self.sample_size = sample_size
#         self.repeats = repeats
#         self.prevalences = prevalences
#         self.change_conditionals = change_conditionals
#         self.random_state = random_state
#         self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

#     def samples_parameters(self):
#         indexesA = []
#         indexesB = []

#         #store the number of positives computed for each prevalence in case is needed
#         self.n_dA_pos=np.zeros((len(self.prevalences)),dtype=int)

#         n_dA_neg = n_dB_pos = self.sample_size//4
#         n_dB_neg = self.sample_size//4

#         dA_pos_ex = np.where(self.domainA.y==1)[0]
#         dA_neg_ex = np.where(self.domainA.y==0)[0]
#         dB_pos_ex = np.where(self.domainB.y==1)[0]
#         dB_neg_ex = np.where(self.domainB.y==0)[0]

#         for i, p in enumerate(self.prevalences):
#             n_dA_pos = round((n_dB_pos-p*(n_dB_pos+n_dA_neg+n_dB_neg))/(p-1))

#             if not self.change_conditionals:
#                 #In this case the idea is to generate a normal sample, but keeping the size of the sample equal to the one if we have a class conditional change
#                 new_sample_size = round((n_dB_pos-p*(n_dB_pos+n_dA_neg+n_dB_neg))/(p-1))+n_dA_neg+n_dB_pos+n_dB_neg
#                 n_pos = round(new_sample_size * p)
#                 n_neg = new_sample_size-n_pos
#                 n_dA_pos = n_pos//2
#                 n_dB_pos = n_pos//2
#                 n_dA_neg = n_neg//2
#                 n_dB_neg = n_neg//2
#             #store it
#             self.n_dA_pos[i] = n_dA_pos

#             print("[p=%.2f,classcond_change=%s] n_dA_neg=%d,n_dB_neg=%d,n_dA_pos=%d,n_dB_pos=%d" % (p,self.change_conditionals,n_dA_neg,n_dB_neg,n_dA_pos,n_dB_pos))

#             for _ in range(self.repeats):
#                 dA_pos = np.random.choice(dA_pos_ex, n_dA_pos, replace=False)
#                 dA_neg = np.random.choice(dA_neg_ex, n_dA_neg, replace=False)
#                 dB_pos = np.random.choice(dB_pos_ex, n_dB_pos, replace=False)
#                 dB_neg = np.random.choice(dB_neg_ex, n_dB_neg, replace=False)

#                 idxsA = np.concatenate((dA_pos,dA_neg))
#                 idxsB = np.concatenate((dB_pos,dB_neg))
#                 indexesA.append(idxsA)
#                 indexesB.append(idxsB)
            
#         return list(zip(indexesA, indexesB))

#     def sample(self, indexes):
#         indexesA, indexesB = indexes
#         sampleA = self.domainA.sampling_from_index(indexesA)
#         sampleB = self.domainB.sampling_from_index(indexesB)
#         return sampleA+sampleB

#     def total(self):
#         return self.repeats * len(self.prevalences)