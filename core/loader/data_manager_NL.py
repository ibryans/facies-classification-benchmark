import os
import numpy
import torch

from sklearn.model_selection import train_test_split

__all__ = ['split_train_val', 'CustomSampler']


class CustomSampler(torch.utils.data.Sampler):
    def __init__(self, sample_list):
        self.sample_list = sample_list
        
    def __iter__(self):
        char = ['i' if numpy.random.randint(2) == 1 else 'x']
        self.indices = [idx for (idx, name) in enumerate(self.sample_list) if char[0] in name]
        return (self.indices[i] for i in torch.randperm(len(self.indices)))


def split_train_val(args, per_val=0.1, data_folder='data_NL'):
    # create inline and crossline sections for training and validation:
    loader_type = 'section'
    labels = numpy.load(os.path.join(data_folder, 'train', 'train_labels.npy'))
    i_list = list(range(labels.shape[0]))
    i_list = ['i_'+str(inline) for inline in i_list]

    x_list = list(range(labels.shape[1]))
    x_list = ['x_'+str(crossline) for crossline in x_list]

    list_train_val = i_list + x_list

    # create train and test splits:
    list_train, list_val = train_test_split(list_train_val, test_size=per_val, shuffle=True)

    # write to files to disK:
    file_object = open(os.path.join(data_folder, 'splits', loader_type + '_train_val.txt'), 'w')
    file_object.write('\n'.join(list_train_val))
    file_object.close()
    file_object = open(os.path.join(data_folder, 'splits', loader_type + '_train.txt'), 'w')
    file_object.write('\n'.join(list_train))
    file_object.close()
    file_object = open(os.path.join(data_folder, 'splits', loader_type + '_val.txt'), 'w')
    file_object.write('\n'.join(list_val))
    file_object.close()
