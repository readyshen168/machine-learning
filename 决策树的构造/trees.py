'''
香农决策树
'''
from math import log


def calc_shannon_ent(data_set):
    '''
    calcShannonEnt 的 Docstring

    :param dataSet: 说明
    '''
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for label, count in label_counts.items():
        prob = float(count)/num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def create_data_set():
    '''
    create_dataset 的 Docstring
    '''
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']

    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels
