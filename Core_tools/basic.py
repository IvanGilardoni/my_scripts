""" User-defined Python functions employed in basic staffs """

import numpy as np
import pandas

#%% 0. USEFUL READINGS
'''

- how to avoid overwriting and make copy correctly: https://realpython.com/copying-python-objects/
 BE CAREFUL: if you deepcopy a class, then this does not mean you are doing a deepcopy of its attributes;
 for example, if class has a dictionary has an attribute, then you are not doing a deepcopy of dictionary!!!

'''
#%% 1. BASIC OPERATIONS WITH LIST, DICTIONARIES AND SO ON

#%% from list of dicts to dict of lists
def swap_list_dict(my_var):
    """ either:
    from dict of lists to list of dicts (provided all the lists have the same length) or
    from list of dicts to dict_of_lists (provided that all the dicts have the same keys)
    """
    if type(my_var) is dict:
        my_var = [dict(zip(my_var.keys(), values)) for values in zip(*my_var.values())]
    elif type(my_var) is list:
        my_var = {key: [d[key] for d in my_var] for key in my_var[0]}
    else:
        print('error')
        my_var = None
    return my_var

#%% 1a. unwrap_dict: unwrap dictionaries (of dictionaries of...) made of lists
# see also: my_dict.values()

def unwrap_dict(d):

    res = []  # Result list
    
    if isinstance(d, dict):
        for val in d.values():
            res.extend(unwrap_dict(val))
    #elif not isinstance(d, list):
    #    res = d        
    #else:
    #    raise TypeError("Undefined type for flatten: %s"%type(d))
    else:
        if isinstance(d, list):
            res = d
        else:
            res = [d]

        # the result is a list of arrays, then do np.hstack
    return np.hstack(res)
    
# unwrap dict with titles: done for 2-layer dictionary, you could do it recursively as for unwrap_dict

def unwrap_dict2(d):
    
    res = []
    keys = []
    
    for key1, value1 in d.items():
        for key2, value2 in value1.items():

            key = key1 + ' ' + key2

            if isinstance(value2, list):
                length = len(value2)
                res.extend(value2)
            else:
                length = np.array(value2).shape[0]
                res.append(value2)

            if length > 1:
                names = [key + ' ' + str(i) for i in range(length)]
            else:
                names = [key]

            keys.extend(names)

    return keys, res

#%% 1b. distinguish (identify) unique and duplicate elements in list

def id_unique_dupl(lista):
    uniq = np.unique(np.array(lista)).tolist()
    
    seen = set()
    dupl = [x for x in lista if x in seen or seen.add(x)]

    return uniq,dupl

#%% 1c. get user-defined attributes of a class Result
def get_attributes(Result):
    return [x for x in dir(Result) if not x.startswith('__')]
    # equivalently:
    # return [x for x in vars(Result).keys() if not x.startswith('__')]

#%% 1d. define new class Result_new with same attributes of class Result:
# you can also transform a class Result into a dictionary with Result.__dict__ (does it work?)
# WATCH OUT: if you make a new class and the old class contains dictionaries, then the two (old and new) 
# dictionaries are the same dictionary (modifying one, also the other is modified); so, do copy.deepcopy()

import copy

def make_new_class(Result,my_keys=None):
    
    class Result_new: pass
    if my_keys is None: my_keys = get_attributes(Result)
    for k in my_keys: setattr(Result_new,k,copy.deepcopy(getattr(Result,k)))

    return Result_new

#%% 1e. Make title to save test_obs in a text file as a single list
# it works for a dictionary with a single layer of subdictionaries;
# you could write a recursive algorithm to make it for an arbitrary n. of subdictionaries

# vars(my_instance_class) ### from class to dictionary
# my_dict.items() my_dict.values() my_dict.keys() ### elements of dictionary (items contains keys and values)

def make_title_from_dict(my_dict):
    
    title = []

    for n1 in my_dict.keys():
        for n2 in my_dict[n1].keys():
            # title.extend(len(out[1][n1][n2])*[str(n1)+' '+str(n2)])

            my_list1 = len(my_dict[n1][n2])*[str(n1)+' '+str(n2)]
            my_list2 = list(np.arange(len(my_list1)))

            title.extend([i+' '+str(j) for i,j in zip(my_list1,my_list2)])

    return title


def swap_dict_to_txt(my_dict, txt_path, sep : str=' '):
    """
    Save a dictionary as a txt file with column names given by indicization of dict keys.
    Each item value should be 0- or 1-dimensional (either int, float, np.ndarray or list),
    not 2-dimensional or more.

    If `my_dict` is None, do the opposite: from txt to dict.
    """

    if my_dict is not None:
        header = []
        values = []

        for key, arr in my_dict.items():
            if (type(arr) is int) or (type(arr) is float):
                header.append(key)
                values.append(arr)
            else:
                # assert ((type(arr) is np.ndarray) and (len(arr.shape) == 1)) or (type(arr) is list), 'error on element with key %s' % key
                # you could also have jax arrays, so manage as follows:

                try:
                    l = len(arr.shape)
                except:
                    l = 0
                assert (l == 1) or (type(arr) is list), 'error on element with key %s' % key
                
                # you should also check that each element in the list is 1-dimensional
                for i, val in enumerate(arr, 1):
                    header.append(f"{key}_{i}")
                    values.append(val)

        with open(txt_path, 'w') as f:
            f.write(sep.join(header) + '\n')
            f.write(sep.join(str(v) for v in values) + '\n')

        return
    
    else:
        df = pandas.read_csv(txt_path, sep=sep)
        output_dict = {}

        # Extract all unique keys (prefix before last "_")
        key_to_cols = {}
        for col in df.columns:
            if '_' in col:
                key, idx = col.rsplit('_', 1)
                key_to_cols.setdefault(key, []).append((int(idx), col))

        # For each key, sort columns and flatten the values
        for key, cols in key_to_cols.items():
            sorted_cols = [col for _, col in sorted(cols)]
            output_dict[key] = df[sorted_cols].values.flatten()

        return output_dict


# recursive: not working
def make_title_of_dict_rec(my_var, name_list = []):
    
    title = []

    if isinstance(my_var, dict):
        for n in my_var.keys():
            print('name: ',name_list)

            name_list.append(n)
            # if name_ == '': name_old = str(n)
            # else: name_old = str(name)+' '+str(n)

            print(name_list)
            out = make_title_of_dict_rec(my_var[n],name_list)
            title.extend(out[0])
    else:
        print(name_list,my_var)
        
        my_string = ''
        for i in name_list: my_string = my_string+' '+i
        my_string = my_string[1:]
        
        my_list1 = len(my_var)*[str(my_string)]
        my_list2 = list(np.arange(len(my_var)))

        title = [i+' '+str(j) for i,j in zip(my_list1,my_list2)]

        print('title: ',title)
        name_list = name_list[:-1]

    return title


def both_class_and_dict():
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            dict.__init__(self, *args, **kwargs)
            # self.x = "Flintstones"
            

    a = AttrDict()  # {"fred": "male", "wilma": "female", "barney": "male"})

    return a
    
    # example:
    # a = AttrDict({"fred": "male", "wilma": "female", "barney": "male"})
    # a.x = "Wilma"

#%% compare two files (e.g., txt files or py source codes)

def compare(file1_path, file2_path):
    from difflib import Differ

    with open(file1_path) as file_1, open(file2_path) as file_2:
        differ = Differ()
    
        for line in differ.compare(file_1.readlines(), file_2.readlines()):
            print(line)

#%% list files in directory (recursive)

import os

def list_files_recursive(path = '.', my_list = [], specific = None):

    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path)
        else:
            my_list.append(full_path)

    if specific is not None:
        if type(specific) is str:  # example: specific = '.npy'
            my_list = [l for l in my_list if l.endswith(specific)]
        elif specific == 'dir':
            my_list = [l for l in my_list if os.path.isdir(l)]

    return my_list

#%% 2. deconvolve_lambdas:

# old version:
def deconvolve_lambdas_old(g,js,lambdas):

    dict_lambdas = {}

    for i1,s1 in enumerate(g.keys()):
        dict_lambdas[s1] = {}
        for i2,s2 in enumerate(g[s1].keys()):
            dict_lambdas[s1][s2] = lambdas[js[i1][i2]:js[i1][i2+1]]

    return dict_lambdas

# new version:
def deconvolve_lambdas(data_n_experiments,lambdas):

    dict_lambdas = {}

    ns = 0

    for s1 in data_n_experiments.keys():#enumerate(g.keys()):
        dict_lambdas[s1] = {}
        for s2 in data_n_experiments[s1].keys():#enumerate(g[s1].keys()):
            dict_lambdas[s1][s2] = lambdas[ns:(ns+data_n_experiments[s1][s2])]
            ns+=data_n_experiments[s1][s2]

    return dict_lambdas

