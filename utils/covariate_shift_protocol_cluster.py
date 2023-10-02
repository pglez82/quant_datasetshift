from quapy.protocol import AbstractStochasticSeededProtocol, OnLabelledCollectionProtocol
from quapy.data import LabelledCollection
import numpy as np
import quapy.functional as F

class CovariateShiftPPCluster(AbstractStochasticSeededProtocol):
    """
    Generates mixtures of two domains (A and B) at controlled rates, but preserving the original class prevalence.

    :param domainA:
    :param domainB:
    :param sample_size:
    :param repeats:
    :param prevalence: the prevalence to preserv along the mixtures. If specified, should be an array containing
        one prevalence value (positive float) for each class and summing up to one. If not specified, the prevalence
        will be taken from the domain A (default).
    :param mixture_points: an integer indicating the number of points to take from a linear scale (e.g., 21 will
        generate the mixture points [1, 0.95, 0.9, ..., 0]), or the array of mixture values itself.
        the specific points
    :param random_state:
    """

    def __init__(
            self,
            domainA: LabelledCollection,
            domainB: LabelledCollection,
            sample_size,
            repeats=1,
            prevalence=None,
            mixture_points=11,
            random_state=None,
            return_type='sample_prev'):
        super(CovariateShiftPPCluster, self).__init__(random_state)
        self.A = domainA
        self.B = domainB
        self.sample_size = sample_size
        self.repeats = repeats
        if prevalence is None:
            self.prevalence = domainA.prevalence()
        else:
            self.prevalence = np.asarray(prevalence)
            assert len(self.prevalence) == domainA.n_classes, \
                f'wrong shape for the vector prevalence (expected {domainA.n_classes})'
            assert F.check_prevalence_vector(self.prevalence), \
                f'the prevalence vector is not valid (either it contains values outside [0,1] or does not sum up to 1)'
        if isinstance(mixture_points, int):
            self.mixture_points = np.linspace(0, 1, mixture_points)[::-1]
        else:
            self.mixture_points = np.asarray(mixture_points)
            assert all(np.logical_and(self.mixture_points >= 0, self.mixture_points<=1)), \
                'mixture_model datatype not understood (expected int or a sequence of real values in [0,1])'
        self.random_state = random_state
        self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

    def samples_parameters(self):
        indexesA, indexesB = [], []
        for propA in self.mixture_points:
            for _ in range(self.repeats):
                nA = int(np.round(self.sample_size * propA))
                nB = self.sample_size-nA
                sampleAidx = self.A.sampling_index(nA, *self.prevalence)
                sampleBidx = self.B.sampling_index(nB, *self.prevalence)
                indexesA.append(sampleAidx)
                indexesB.append(sampleBidx)
        return list(zip(indexesA, indexesB))

    def sample(self, indexes):
        indexesA, indexesB = indexes
        sampleA = self.A.sampling_from_index(indexesA)
        sampleB = self.B.sampling_from_index(indexesB)
        y = sampleB.y
        y[y==0] = 2
        y[y==1] = 3
        sampleB = LabelledCollection(sampleB.X,y)
        return sampleA+sampleB

    def total(self):
        return self.repeats * len(self.mixture_points)
