from quapy.protocol import AbstractStochasticSeededProtocol, OnLabelledCollectionProtocol
from quapy.data import LabelledCollection
import quapy.functional as F
import numpy as np

class ConceptShiftProtocol(AbstractStochasticSeededProtocol):
    """
    Generates samples with concept shift. The idea is to use a multiclass dataset where there is a cut point used
    to binarize it. Concept shift is simulated changing the cut point to transform examples in positive or negative.

    :param data: LabelledCollection with the data
    :param sample_size: desired size for the samples
    :param cut_points: array with the cut points to use to generate concept shift
    :param repeats: number of repetitions (number of samples for each cut point)
    :param prevalence: the prevalence to preserv along the mixtures. Array with two numbers indicating the prevalence
        of the negative and positive class.
    :param random_state: random state to make results reproducible
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