import copy
import torch
import numpy as np
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import StandardScaler
from mytorch.utils.goodies import *

# custom imports
# import config
from utils.simple_classification_dataset import *
from utils.iterator import *
from utils.iterator import CombinedIterator, TextClassificationDataset


class SimpleAdvDatasetReader():
    def __init__(self, dataset_name:str,**params):
        self.dataset_name = dataset_name.lower()
        self.batch_size = params['batch_size']
        self.fairness_iterator = params['fairness_iterator']
        self.train_split = .80
        self.is_fair_grad = params['fair_grad_newer']

        if 'celeb' in self.dataset_name:
            self.X, self.y, self.s = get_celeb_data()
        elif 'adult' in self.dataset_name and 'multigroup' not in self.dataset_name:
            self.X, self.y, self.s = get_adult_data()
        elif 'crime' in self.dataset_name:
            self.X, self.y, self.s = get_crimeCommunities_data()
        elif 'dutch' in self.dataset_name:
            self.X, self.y, self.s = get_dutch_data()
        elif 'compas' in self.dataset_name:
            self.X, self.y, self.s = get_compas_data()
        elif 'german' in self.dataset_name:
            self.X, self.y, self.s = get_german_data()
        elif 'adult' in self.dataset_name and 'multigroup' in self.dataset_name:
            self.X, self.y, self.s = get_celeb_multigroups_data()
        elif 'gaussian' in self.dataset_name:
            raise NotImplementedError
            # self.X, self.y, self.s = drh.get_gaussian_data(50000)
        else:
            raise NotImplementedError

        # converting all -1,1 -> 0,1
        self.y = (self.y+1)/2

        if len(np.unique(self.s)) == 2 and -1 in np.unique(self.s):
            self.s = (self.s + 1) / 2


    def process_data(self, X,y,s, vocab):
        """raw data is assumed to be tokenized"""

        final_data = [(a,b,c) for a,b,c in zip(y,X,s)]


        label_transform = sequential_transforms()
        input_transform = sequential_transforms()
        aux_transform = sequential_transforms()

        transforms = (label_transform, input_transform, aux_transform)

        return TextClassificationDataset(final_data, vocab, transforms)


    def collate(self, batch):
        labels, input, aux = zip(*batch)

        labels = torch.LongTensor(labels)
        aux = torch.LongTensor(aux)
        lengths = torch.LongTensor([len(x) for x in input])
        input = torch.FloatTensor(input)

        input_data = {
            'labels': labels,
            'input': input,
            'lengths': lengths,
            'aux': aux
        }

        return input_data


    def run(self):

        dataset_size = self.X.shape[0] # examples*feature_size
        # the dataset is shuffled so as to get a unique test set for each seed.
        index = np.random.permutation(dataset_size)
        self.X, self.y, self.s = self.X[index], self.y[index], self.s[index]
        test_index = int(self.train_split*dataset_size)

        if self.is_fair_grad:
            dev_index = int(self.train_split*dataset_size) - int(self.train_split*dataset_size*.25)
        else:
            dev_index = int(self.train_split * dataset_size) - int(self.train_split * dataset_size * .10)

        number_of_labels = len(np.unique(self.y))

        train_X, train_y, train_s = self.X[:dev_index,:], self.y[:dev_index], self.s[:dev_index]
        dev_X, dev_y, dev_s = self.X[dev_index:test_index, :], self.y[dev_index:test_index], self.s[dev_index:test_index]
        test_X, test_y, test_s = self.X[test_index:, :], self.y[test_index:], self.s[test_index:]

        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        dev_X = scaler.transform(dev_X)
        test_X = scaler.transform(test_X)





        vocab = {'<pad>':1} # no need of vocab in these dataset. It is there for code compatibility purposes.

        train_data = self.process_data(train_X,train_y,train_s, vocab=vocab)
        dev_data = self.process_data(dev_X,dev_y,dev_s, vocab=vocab)
        test_data = self.process_data(test_X,test_y,test_s, vocab=vocab)

        #
        # fairness_data = \
        #     create_fairness_data(
        #         train_X, train_y, train_s, dev_X, dev_y,
        #         dev_s, self.process_data, vocab, self.fairness_iterator)

        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        # fairness_iterator = torch.utils.data.DataLoader(fairness_data,
        #                                             512,
        #                                             shuffle=False,
        #                                             collate_fn=self.collate
        #                                             )

        iterators = []  # If it was k-fold. One could append k iterators here.
        iterator_set = {
            'train_iterator': train_iterator,
            'valid_iterator': dev_iterator,
            'test_iterator': test_iterator,
            # 'fairness_iterator': fairness_iterator # now this can be independent of the dev iterator.
        }
        iterators.append(iterator_set)

        other_meta_data = {}
        other_meta_data['task'] = 'simple_classification'
        other_meta_data['dataset_name'] = self.dataset_name


        return vocab, number_of_labels, number_of_labels, iterators, other_meta_data

if __name__ == '__main__':
    dataset_name = 'adult'
    params = {
        'batch_size': 64,
        'fairness_iterator': 'custom_3',
        'fair_grad_newer': False,
        'fairness_iterator': 'some dummy sruff'
    }

    dataset = SimpleAdvDatasetReader(dataset_name, **params)
    _ = dataset.run()