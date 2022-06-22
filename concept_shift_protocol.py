from quapy.protocol import AbstractStochasticSeededProtocol, OnLabelledCollectionProtocol
from quapy.data import LabelledCollection
import quapy.functional as F
import numpy as np

class ConceptShiftProtocol(AbstractStochasticSeededProtocol):
    """
    Generates mixtures of two domains (A and B) at controlled rates, but preserving the original class prevalence.

    :param domainA:
    :param domainB:
    :param sample_size:
    :param repeats:
    :param prevalence: the prevalence to preserv along the mixtures. If specified, should be an array containing
        one prevalence value (positive float) for each class and summing up to one. If not specified, the prevalence
        will be taken from data (default).
    :param mixture_points: an integer indicating the number of points to take from a linear scale (e.g., 21 will
        generate the mixture points [1, 0.95, 0.9, ..., 0]), or the array of mixture values itself.
        the specific points
    :param random_state:
    """

    def __init__(
            self,
            data: LabelledCollection,
            sample_size,
            cut_points,
            repeats=1,
            prevalence=None,
            random_state=None,
            return_type='sample_prev'):
        super(ConceptShiftProtocol, self).__init__(random_state)
        self.data = data
        self.sample_size = sample_size
        self.repeats = repeats
        self.cut_points = np.asarray(cut_points)
        self.prevalence = prevalence
        self.random_state = random_state
        self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

    def binarize_dataset(self, dataset: LabelledCollection, cut_point: int):
        instances = dataset.X
        labels = dataset.y
        labels, instances = labels[labels!=cut_point], instances[labels!=cut_point]
        labels[labels<cut_point] = 0
        labels[labels>cut_point] = 1
        return LabelledCollection(instances, labels)

    def samples_parameters(self):
        indexes = []
        cut_point_sample = []
        self.binarized_datasets = {}
        for cut_point in self.cut_points:
            #we need to binarize the dataset according to the cutpoint
            self.binarized_datasets[cut_point] = self.binarize_dataset(self.data, cut_point = cut_point)
            for _ in range(self.repeats):
                if self.prevalence is not None:
                    idxs = self.binarized_datasets[cut_point].sampling_index(self.sample_size,self.prevalence[0])
                else:
                    idxs = self.binarized_datasets[cut_point].uniform_sampling_index(self.sample_size)
                indexes.append(idxs)
                cut_point_sample.append(cut_point)
        return list(zip(cut_point_sample,indexes))

    def sample(self, indexes):
        cut_point, indexes = indexes
        return self.binarized_datasets[cut_point].sampling_from_index(indexes)

    def total(self):
        return self.repeats * len(self.cut_points)