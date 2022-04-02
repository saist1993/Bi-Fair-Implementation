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
            self.X, self.y, self.s = get_gaussian_data(50000)
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


class EncodedEmoji:
    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name
        self.n = 100000 # https://github.com/HanXudong/Diverse_Adversaries_for_Mitigating_Bias_in_Training/blob/b5b4c99ada17b3c19ab2ae8789bb56058cb72643/scripts_deepmoji.py#L270
        self.folder_location = '../DPNLP/datasets/deepmoji'
        try:
            self.ratio = params['ratio_of_pos_neg']
        except:
            self.ratio = 0.8 # this the default in https://arxiv.org/pdf/2101.10001.pdf
        self.batch_size = params['batch_size']
        self.fairness_iterator = params['fairness_iterator']

    def read_data_file(self, input_file: str):
        vecs = np.load(input_file)

        np.random.shuffle(vecs)

        return vecs[:40000], vecs[40000:42000], vecs[42000:44000]

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

        try:

            train_pos_pos, dev_pos_pos, test_pos_pos = self.read_data_file(f"{self.folder_location}/pos_pos.npy")
            train_pos_neg, dev_pos_neg, test_pos_neg = self.read_data_file(f"{self.folder_location}/pos_neg.npy")
            train_neg_pos, dev_neg_pos, test_neg_pos = self.read_data_file(f"{self.folder_location}/neg_pos.npy")
            train_neg_neg, dev_neg_neg, test_neg_neg = self.read_data_file(f"{self.folder_location}/neg_neg.npy")

        except:

            self.folder_location = '/home/gmaheshwari/storage/fair_nlp_dataset/data/deepmoji'
            train_pos_pos, dev_pos_pos, test_pos_pos = self.read_data_file(f"{self.folder_location}/pos_pos.npy")
            train_pos_neg, dev_pos_neg, test_pos_neg = self.read_data_file(f"{self.folder_location}/pos_neg.npy")
            train_neg_pos, dev_neg_pos, test_neg_pos = self.read_data_file(f"{self.folder_location}/neg_pos.npy")
            train_neg_neg, dev_neg_neg, test_neg_neg = self.read_data_file(f"{self.folder_location}/neg_neg.npy")


        n_1 = int(self.n * self.ratio / 2)
        n_2 = int(self.n * (1 - self.ratio) / 2)

        fnames = ['pos_pos.npy', 'pos_neg.npy', 'neg_pos.npy', 'neg_neg.npy']
        main_labels = [1, 1, 0, 0]
        protected_labels = [1, 0, 1, 0]
        ratios = [n_1, n_2, n_2, n_1]
        data = [train_pos_pos, train_pos_neg, train_neg_pos, train_neg_neg]

        X_train, y_train, s_train = [], [], []

        # loading data for train

        for data_file, main_label, protected_label, ratio in zip(data, main_labels, protected_labels, ratios):
            X_train = X_train + list(data_file[:ratio])
            y_train = y_train + [main_label] * len(data_file[:ratio])
            s_train = s_train + [protected_label] * len(data_file[:ratio])


        X_dev, y_dev, s_dev = [], [], []
        for data_file, main_label, protected_label in zip([dev_pos_pos, dev_pos_neg, dev_neg_pos, dev_neg_neg]
                , main_labels, protected_labels):
            X_dev = X_dev + list(data_file)
            y_dev = y_dev + [main_label] * len(data_file)
            s_dev = s_dev + [protected_label] * len(data_file)


        X_test, y_test, s_test = [], [], []
        for data_file, main_label, protected_label in zip([test_pos_pos, test_pos_neg, test_neg_pos, test_neg_neg]
                , main_labels, protected_labels):
            X_test = X_test + list(data_file)
            y_test = y_test + [main_label] * len(data_file)
            s_test = s_test + [protected_label] * len(data_file)


        X_train, y_train, s_train = np.asarray(X_train), np.asarray(y_train), np.asarray(s_train)
        X_dev, y_dev, s_dev = np.asarray(X_dev), np.asarray(y_dev), np.asarray(s_dev)
        X_test, y_test, s_test = np.asarray(X_test), np.asarray(y_test), np.asarray(s_test)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_dev = scaler.transform(X_dev)
        X_test = scaler.transform(X_test)




        all_x = [[a, b] for a, b in zip(y_train, s_train)]
        new_stuff = {}
        for i in all_x:
            try:
                new_stuff[str(i)] = new_stuff[str(i)] + 1
            except:
                new_stuff[str(i)] = 1

        print(new_stuff)

        vocab = {'<pad>':1} # no need of vocab in these dataset. It is there for code compatibility purposes.
        number_of_labels = 2

        # shuffling data
        shuffle_train_index = np.random.permutation(len(X_train))
        X_train, y_train, s_train = X_train[shuffle_train_index], y_train[shuffle_train_index], s_train[shuffle_train_index]

        shuffle_dev_index = np.random.permutation(len(X_dev))
        X_dev, y_dev, s_dev = X_dev[shuffle_dev_index], y_dev[shuffle_dev_index], s_dev[
            shuffle_dev_index]

        shuffle_test_index = np.random.permutation(len(X_test))
        X_test, y_test, s_test = X_test[shuffle_test_index], y_test[shuffle_test_index], s_test[
            shuffle_test_index]

        train_data = self.process_data(X_train,y_train,s_train, vocab=vocab)
        dev_data = self.process_data(X_dev,y_dev,s_dev, vocab=vocab)
        test_data = self.process_data(X_test,y_test,s_test, vocab=vocab)

        # fairness_data = \
        #     create_fairness_data(
        #         X_train, y_train, s_train, X_dev, y_dev,
        #         s_dev, self.process_data, vocab, self.fairness_iterator)


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
        #                                                 512,
        #                                                 shuffle=False,
        #                                                 collate_fn=self.collate
        #                                                 )

        iterators = []  # If it was k-fold. One could append k iterators here.
        iterator_set = {
            'train_iterator': train_iterator,
            'valid_iterator': dev_iterator,
            'test_iterator': test_iterator
            # 'fairness_iterator': fairness_iterator  # now this can be independent of the dev iterator.
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