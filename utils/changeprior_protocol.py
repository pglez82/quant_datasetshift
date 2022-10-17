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


class ChangePriorProtocolV2(AbstractStochasticSeededProtocol):
    """
    Generate samples with a change just in the prior, not mantaining P(X|Y). 
    For this version we need for domains of the data. 
    domainA_1 and domainA_2 will be the negative class, and domainB_1 and
    domainB_2 is the postive class. We start from an unbalanced situation (referring to
    the subclasses) in which 1/3 domainA_1, 2/3 domainA_2, 2/3 domainB_1
    and 1/3 domainB_2. This way, we can achieve a bigger change in P(Y) changing only
    one subdomain (domainA_2).

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
            domainA_1: LabelledCollection,
            domainA_2: LabelledCollection,
            domainB_1: LabelledCollection,
            domainB_2: LabelledCollection,
            sample_size,
            prevalences,
            change_conditionals = True,
            repeats=1,
            random_state=None,
            return_type='sample_prev'):
        super(ChangePriorProtocolV2, self).__init__(random_state)
        self.domainA_1 = domainA_1
        self.domainA_2 = domainA_2
        self.domainB_1 = domainB_1
        self.domainB_2 = domainB_2
        self.sample_size = sample_size
        self.repeats = repeats
        self.prevalences = prevalences
        self.change_conditionals = change_conditionals
        self.random_state = random_state
        self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

    def samples_parameters(self):
        indexesA_1 = []
        indexesA_2 = []
        indexesB_1 = []
        indexesB_2 = []

        #store the number of examples of each domain computed for each prevalence in case is needed
        self.n_dA_1=np.zeros((len(self.prevalences)),dtype=int)
        self.n_dA_2=np.zeros((len(self.prevalences)),dtype=int)
        self.n_dB_1=np.zeros((len(self.prevalences)),dtype=int)
        self.n_dB_2=np.zeros((len(self.prevalences)),dtype=int)

        for i, p in enumerate(self.prevalences):
            n_dA_1 = n_dB_2 = round(self.sample_size/6)
            n_dB_1 = round(self.sample_size/3)
            n_dA_2 = round((n_dB_2-p*(n_dB_2+n_dA_1+n_dB_1))/(p-1))

            if not self.change_conditionals:
                #In this case the idea is to generate a normal sample, but keeping the size of the sample equal to the one if we have a class conditional change
                new_sample_size = n_dA_1+n_dA_2+n_dB_1+n_dB_2
                n_B = round(new_sample_size * p)
                n_A = new_sample_size-n_B
                n_dA_1 = round(n_A/3)
                n_dA_2 = round(n_A*2/3)                
                n_dB_1 = round(n_B*2/3)
                n_dB_2 = round(n_B/3)
            
            #store it
            self.n_dA_1[i] = n_dA_1
            self.n_dA_2[i] = n_dA_2
            self.n_dB_1[i] = n_dB_1
            self.n_dB_2[i] = n_dB_2

            for _ in range(self.repeats):
                idx_dA_1 = self.domainA_1.uniform_sampling_index(n_dA_1)
                idx_dA_2 = self.domainA_2.uniform_sampling_index(n_dA_2)
                idx_dB_1 = self.domainB_1.uniform_sampling_index(n_dB_1)
                idx_dB_2 = self.domainB_2.uniform_sampling_index(n_dB_2)

                indexesA_1.append(idx_dA_1)
                indexesA_2.append(idx_dA_2)
                indexesB_1.append(idx_dB_1)
                indexesB_2.append(idx_dB_2)
            
        return list(zip(indexesA_1,indexesA_2, indexesB_1, indexesB_2))

    def sample(self, indexes):
        indexesA_1,indexesA_2,indexesB_1,indexesB_2 = indexes
        sampleA_1 = self.domainA_1.sampling_from_index(indexesA_1)
        sampleA_2 = self.domainA_2.sampling_from_index(indexesA_2)
        sampleB_1 = self.domainB_1.sampling_from_index(indexesB_1)
        sampleB_2 = self.domainB_2.sampling_from_index(indexesB_2)
        sample = sampleA_1+sampleA_2+sampleB_1+sampleB_2
        suffle = np.random.permutation(len(sample))
        sample = LabelledCollection(sample.X[suffle],sample.y[suffle])
        return sample

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