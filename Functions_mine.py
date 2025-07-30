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

#%% 3. class statistics, including: compute Kish size

class statistics():

    # compute inverse Kish size, which is the effective n. of frames
    # remember: kish of (1/2,0,1/2) is different from kish of (1/2,1/2), namely, 0 frames do matter!
    def compute_kish_inv(weights, weights0=None, if_geom=False):
        if weights0 is None: # w.r.t. uniform distribution
            if not if_geom: # arithmetic average (usual Kish size)
                kish = np.sum(weights**2)*len(weights)
            else: # geometric average (relative entropy)
                
                # 0*log0 = 0
                weights_non0 = weights[np.argwhere(weights!=0).flatten()]

                kish = np.exp(np.sum(weights_non0*np.log(weights_non0)))*len(weights)
        else: # w.r.t. weights0
            if not if_geom:
                kish = np.sum(weights**2/weights0)
            else:
                wh = np.argwhere(weights!=0).flatten()
                weights_non0 = weights[wh]
                kish = np.exp(np.sum(weights_non0*np.log(weights_non0/weights0[wh])))

        eff_n_frames = 1/kish

        return eff_n_frames

    def angles_from_sincos(sin, cos):

        angles = np.arcsin(sin)
        wh = np.argwhere(sin < 0)
        angles[wh] = np.pi - angles[wh]

        angles = np.mod(angles, 2*np.pi)

        wh = np.argwhere((angles > np.pi) & (angles < 2*np.pi))
        angles[wh] = angles[wh] - 2*np.pi

        return angles

    def detect_convergence(time_series, threshold_fact = 50., which_method = 1, if_plot = False):
        """
        There are several ways to detect convergence of a time series (beyond visual inspection),
        for example you can compute averages on time windows and then compare them.
        Here, `threshold_fact` is not required.

        Alternatively (`which_method = 2`), we can compute the variation of the cumulative averages
        and check where it goes under a threshold. This variation is equal to:

        $ \delta m(j) = m(j + 1) - m(j) = ... = 1/(j + 1) * (x_{j+1} - m(j))$

        where $m(j)$ is the cumulative average up to frame n. $j$.
        """

        time_series = np.asarray(time_series)

        if which_method == 1:
            std = np.std(time_series[-5000:])
            diffs = np.abs(time_series - time_series[-1])/std
            wh = np.argwhere(diffs < 7)

            b = ( len(wh)/len(diffs) > 0.99 )
            # this is to avoid sporadic deviations above the threshold will affect the convergence detection

            if b : position = 0
            else : position = wh[np.where(np.ediff1d(wh) != 1)][-1]

            if if_plot:
                plt.figure()
                plt.plot(diffs)
                plt.plot([0, len(diffs)], 7*np.ones(2), 'k')

            return position

        elif which_method == 2:
            cumulative_avg = np.cumsum(time_series) / np.arange(1, len(time_series) + 1)
            diff = np.ediff1d(cumulative_avg)

            my_max = np.max(diff)
            threshold = my_max/threshold_fact

            wh = np.argwhere(diff < threshold)
            b = ( wh[-1] == len(diff) - 1 )[0]

            if if_plot:
                plt.figure()
                plt.plot(diff)
                plt.plot([0, len(diff)], threshold*np.ones(2))
                plt.title('diff')

            if b :
                position = wh[np.where(np.ediff1d(wh) != 1)][-1]
                print('convergence found at %s' % position)
                return position
            else:
                print('convergence not detected')
                return None 

    def block_analysis(x, size_blocks=None):
    
        size = len(x)
        mean = np.mean(x)
        std = np.std(x)/np.sqrt(size)

        delta = 1
        if size_blocks is None:
            size_blocks = np.arange(1, np.int64(size/2) + delta, delta)

        n_blocks = []
        epsilon = []

        for size_block in size_blocks:
            n_block = int(size/size_block)
            
            # a = 0 
            # for i in range(n_block):
            #     a += (np.mean(x[(size_block*i):(size_block*(i+1))]))**2
            # 
            # epsilon.append(np.sqrt((a/n_blocks[-1] - mean**2)/n_blocks[-1]))

            block_averages = []
            for i in range(n_block):
                block_averages.append(np.mean(x[(size_block*i):(size_block*(i+1))]))
            block_averages = np.array(block_averages)

            n_blocks.append(n_block)
            epsilon.append(np.sqrt((np.mean(block_averages**2)-np.mean(block_averages)**2)/n_block))
        
        return mean, std, epsilon, n_blocks, size_blocks

#%% Metropolis algorithm

def run_Metropolis(x0, proposal, energy_function, *, kT=1, n_steps=100):

    """Metropolis algorithm with customizable proposal and energy_function. For example:
    
    ```
    def proposal(x0, L, val1=0.3, val2=0.7):

        moves = np.random.uniform(size=len(x0))
            
        x1 = x0
        x1[moves < val1] -= 1
        x1[moves > val2] += 1

        x1 = np.mod(x1, L)  # periodic boundary conditions

        return x1

    proposal_full = {}
    proposal_full['fun'] = proposal
    proposal_full['args'] = (L, 0.3, 0.7)
    ```

    ```
    def compute_energy(xs, ene0 = +2, ene1 = +1):
        sorted_x0 = np.sort(x0)
        vec = np.ediff1d(sorted_x0)
        vec = np.append(vec, sorted_x0[0] + L - sorted_x0[-1])
        energy = len(np.where(vec == 0)[0])*ene0 + len(np.where(vec == 1)[0])*ene1
        return energy

    energy_function_full = {'fun': compute_energy, 'args': (+2, +1)}
    ```
    """

    traj = []
    # time = []
    ene = []
    av_alpha = 0

    traj.append(x0)
    # time.append(0)
    u0 = energy_function['fun'](x0)
    ene.append(u0)
    # print('u0: ', u0)

    for i_step in range(n_steps):

        x_try = proposal['fun'](x0, *proposal['args'])
        u_try = +energy_function['fun'](x_try, *energy_function['args'])
        # print('u_try: ', u_try)

        alpha = np.exp(-(u_try-u0)/kT)

        # print(alpha)
        
        if alpha > 1: alpha=1
        if alpha > np.random.random():
            av_alpha += 1
            x0 = +x_try
            u0 = +u_try
        
        # traj.append(x0)
        
        # to avoid overwriting!
        traj.append([])
        traj[-1] = +x0
        
        # time.append(i_step)
        ene.append(u0)

        # print(traj)

    av_alpha = av_alpha/n_steps
    
    return np.array(traj), np.array(ene), av_alpha


#%% 4. class thermodynamics (including free energy difference)

class thermodynamics():

    # compute free energy difference
    # dE_AB is E(B)-E(A), in unit of temperature if if_adimensional
    def compute_DG(temperature,weights_A,dE_AB,if_adimensional=True):

        weights_A = weights_A/np.sum(weights_A)
        
        if if_adimensional: beta_dE_AB = dE_AB
        else: beta_dE_AB = dE_AB/temperature
        
        DG = -temperature*np.log(np.sum(weights_A*np.exp(-beta_dE_AB)))
        
        return DG

#%% 5. class my_plots
# it includes:
# - two_scale_plot (plot with 2 y scales)

# see also:
# https://github.com/cxli233/FriendsDontLetFriends/tree/main?tab=readme-ov-file#friends-dont-let-friends-use-boxpot-for-binomial-data


import matplotlib.pyplot as plt

class my_plot_scripts():

    """ font size:
    
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    matplotlib.rc('font', **font)

    """

    #%% 1. two_scale_plot
    # examples for two_scale_plot:

    # colors1 = ['b','g','y','c','r','k']
    # labels1 = ['C6','N6','H61','N1','C10','H101']
    # colors2 = ['b']
    # labels2 = ['$V_{\eta}$']

    # xs = np.arange(10)
    # ys1 = []
    # for i in range(6):
    #     ys1.append(np.random.rand(len(xs)))

    # ys2 = [np.random.rand(len(xs))]

    def two_scale_plot(xs,ys1,labels1,colors1,ys2,labels2,colors2):

        plt.figure(figsize=(19.20,10.80))

        # FIGURE 1
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        for i in range(len(ys1)):
            ax1.plot(xs, ys1[i], '.-', color=colors1[i], label=labels1[i])

        ax1.set_ylim([-0.4,0.32])
        ax2.set_ylim([1.3,2.9])

        ax1.tick_params(axis='both', which='major', labelsize=13)
        ax2.tick_params(axis='both', which='major', labelsize=13)

        # FIGURE 2
        for i in range(len(ys2)):
            ax2.plot(xs, ys2[i], 'D', color=colors2[i], label=labels2[i])
        

        ax1.set_xscale('log')
        
        #plt.set_xlabel('log(alphaspha)', fontsize=15)
        #plt.ylabel('lambda valphasue', fontsize=15)
        #ax1.set_title('A <-> m6A replicas distribution', fontsize  =15 )
        ax1.set_xlabel(r'$\alpha$ ($e^{-2}$)', fontsize  =18)
        ax1.set_ylabel(r'$\Delta$Q ($e$)', fontsize  =18)
        ax2.set_ylabel(r'$V_{\eta}$ ($kJ/mol$)', fontsize  =18)
        #ax2.set_ylim(0,100)
        ax1.legend(loc='upper right', fontsize=13, ncol=2)
        ax2.legend(bbox_to_anchor=(1.0, 0.47), fontsize=15)
        # plt.savefig('fig.png', format='png', dpi=500, bbox_inches='tight')
        plt.show()

    #%% 2. returns list of first n default colors/markers
    def default_colors(n=10):
        colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink',
            'tab:gray','tab:olive','tab:cyan']
        return colors[:n]
    
    def default_markers(n=None):
        markers = ['.','v','s','*','D','o','^','<','>','p','P','+','X','d']
        if n is not None: markers = markers[:n]
        return markers

    #%% 3. plot flatten dictionary (with keys in legend)
    
    def plot_flatten_dict(my_dict):
        l = 0
        for s1 in my_dict.keys():
            for s2 in my_dict[s1].keys():
                l_new = l+len(my_dict[s1][s2])
                plt.plot(np.arange(l,l_new),my_dict[s1][s2],'.',label='%s %s' % (s1,s2))
                l = l_new
        
        plt.legend()

    #%% 4. plot with double labeling
    # it can be improved by including the possibility for different x for each my_dict[name1][name2]
    # in principle, you can add more legends

    def plot_DataFrame(df):

            barWidth = 1/(len(df.columns) + 1)

            plt.subplots(figsize=(12, 8)) 

            brs = []
            brs.append(np.arange(len(df.iloc[:, 0])))
    
            plt.bar(brs[-1], df.iloc[:, 0], label=df.columns[0], width=barWidth)  # edgecolor ='grey', color ='tab:blue')

            for i in range(1, len(df.columns)):
                    brs.append([x + barWidth for x in brs[-1]])
                    plt.bar(brs[-1], df.iloc[:, i], label=df.columns[i], width=barWidth)  # edgecolor ='grey', color ='tab:blue')

            plt.xticks([r + barWidth*(len(df.columns) - 1)/2 for r in range(len(df.iloc[:, 0]))], list(df.index))
            # plt.xticks([r + barWidth for r in range(len(df.iloc[:, 0]))], list(df.index))
            # plt.xlabel(list(df.index))
            
            # plt.xlabel('Branch', fontweight ='bold', fontsize = 15) 
            # plt.ylabel('Students passed', fontweight ='bold', fontsize = 15) 
            # plt.xticks([r + barWidth for r in range(len(df['Aduri'].iloc[:-1]))], names_charges)

            plt.legend()
            plt.gca().xaxis.grid(True)
            # plt.grid()
            # plt.show()

            return

    def plot_double_legend(x, my_dict, *, add_label2 = '', loc1 = None, loc2 = None):
        colors = my_plot_scripts.default_colors()
        markers = my_plot_scripts.default_markers()
        for i in range(len(markers)): markers[i] = markers[i] + '--'

        plt.figure()

        plot_lines = []
        keys1 = list(my_dict.keys())
        keys2 = {}

        for i, name1 in enumerate(keys1):
            l = {}
            keys2[name1] = list(my_dict[name1].keys())
            for j, name2 in enumerate(keys2[name1]):
                vec = my_dict[name1][name2]
                l[name2], = plt.plot(x, vec, markers[j], color=colors[i])
                
            plot_lines.append(unwrap_dict(l))

        if loc1 is not None: loc = loc1
        else: loc = 2

        legend1 = plt.legend(plot_lines[0], keys2[keys1[0]], loc=loc)
        plt.gca().add_artist(legend1)

        if loc2 is not None: loc = loc2
        else: loc = 3

        labels = [add_label2 + str(keys1[i]) for i in range(len(keys1))]
        plt.legend([l[0] for l in plot_lines], labels, loc=loc)

    def plot_3d_with_arrows():
        from matplotlib.patches import FancyArrowPatch
        from mpl_toolkits.mplot3d import proj3d
        from mpl_toolkits.mplot3d.axes3d import Axes3D

        (x0, y0, z0) = (1.5, 2, 4)

        N = 40

        # theta = np.random.rand(N)*2*np.pi
        theta = np.linspace(-4*np.pi, -1.5*np.pi, N)#2*np.pi, N)
        phi = theta
        # np.random.shuffle(theta)
        # z = np.linspace(-2, 2, 100)
        r = 8/np.arange(1,N+1,1)#z**2 + 1
        x = x0 + r*np.sin(theta)*np.cos(phi)
        y = y0 + r*np.sin(theta)*np.sin(phi)
        z = z0 + r*np.cos(theta)

        upto = 15
        x = x[:-upto]
        y = y[:-upto]
        z = z[:-upto]

        ax = plt.figure(figsize=(10, 8)).add_subplot(projection='3d')
        plt.title(r'gradient descent of $\min\,\chi^2(\alpha,\beta,\gamma)$', fontsize=20)

        ax.set_xlabel(r'$\log\,\alpha$', fontsize=20)
        # ax.tick_params(labelleft = False, left = False)
        # ax.set_xticks("")#color = 'w')
        ax.set_ylabel(r'$\log\,\beta$', fontsize=20)
        ax.set_zlabel(r'$\log\,\gamma$', fontsize=20)
        ax.plot(x, y, z, '.', label='parametric curve')

        ax.plot(x[-1], y[-1], z[-1], 'Xk', markersize=10)  # , color = 'b')
        # ax.legend()

        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
                FancyArrowPatch.draw(self, renderer)

        for i in range(len(x)-1):

            a = Arrow3D([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], mutation_scale=10, lw=3, 
                arrowstyle='-|>', color='r')
            ax.add_artist(a)


        # ax.set_xlim([1.2, 1.6])
        # ax.set_ylim([1.8, 2.2])
        # ax.set_zlim([3,5])

        # ax.set_xlim(left = 0)
        # ax.set_ylim(bottom = 0)
        # Axes3D.set_zlim(ax, bottom = 0)

        ax.xaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.yaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.zaxis.set_pane_color((1.0,1.0,1.0,1.0))

        # plt.draw()
        # plt.show()

    def plot_2d_dataframe(df):
        from matplotlib.colors import LogNorm
    
        # if df.columns.name=='alpha': alphas = df.columns
        # if df.index.name=='gamma': gammas = df.index
        alphas = df.columns
        gammas = df.index
    
        X,Y = np.meshgrid(alphas,gammas)
    
        #s='test_red.chi2'#_sameobs'
        #s='train_red.chi2'
        #a=np.array(data_ERFF['mean'][s])
        plt.figure(figsize=(6,6))
        plt.rcParams['font.size'] = 13
        #plt.imshow(b,cmap='jet',interpolation='none')
        # b_show = df.iloc[::-1]
    
        #np.array2string(np.array(b_show.columns))
        # cols=['0.01','0.1','1','2','5','10','20','50','100','1e3','1e4','1e5','1e6','inf']
        cols = ['{:.0e}'.format(alpha) for alpha in alphas]
        rows = ['{:.0e}'.format(gamma) for gamma in gammas]
        # rows=(['0']+cols)[::-1]
        # rows
    
        plt.imshow(df, cmap='jet', interpolation='none', norm=LogNorm(vmin=np.nanmin(np.array(df)), vmax=np.nanmax(np.array(df)))) # b or np.log(b)
        #plt.imshow(b_show,cmap='jet',interpolation='none')#,vmin=np.min(np.array(b_show)),vmax=np.max(np.array(b_show)))
        #plt.imshow(np.log(np.array(b).T),cmap='jet',interpolation='none') # interpolation='lanczos'
        plt.colorbar()
        plt.xlabel(r'$\alpha$')
        plt.xticks(range(len(alphas)),cols,rotation=90) # b_show.columns
        plt.ylabel(r'$\gamma$')
        plt.yticks(range(len(gammas)),rows)#b_show.index)
        # plt.title(r'$\alpha,\zeta$ angles force field correction')
        #plt.title(r'red. $\chi^2$, trainingest (new observables)')#new observables')
        #plt.title(r'-$S_{rel}[P|P_0]$ training')
        plt.show()

    def save_pdf(path, name):
        '''saving images as .pdf allows to avoid grainy images'''
        
        if not path[-1] == '/':
            path = path + '/'
        
        plt.savefig(path + name + '.pdf', format='pdf', bbox_inches='tight')

    
    def plot_with_interrupted_x_axis(ns, x, ys, labels, colors=None, lines=None, d : float=.015,
        delta : float=0.2, figsize : tuple=(6, 4)):
        """
        Plot `y` vs. `x` values `for y in ys` with interrumption of x axis beyond `ns[0]` and `ns[1]`
        and stop x axis at `ns[2]`.
        The variable `ns` is either a list/array of float values (in this case the indices are got by
        `np.argmin`) or integer values (in this case they are exactly the indices of `x`) corresponding to
        breaking x axis.

        Example:
        ```
        ns = [-100, -70, -25, 10, 30]
        x = np.linspace(ns[0], ns[-1])
        
        m = 5
        ys = [np.random.rand(len(x)) for i in range(m)]

        labels = ['%i' % i for i in range(m)]
        colors = my_plot_scripts.default_colors(m)
        lines = ['.-']*m

        fig, ax1, ax2 = plot_with_interrumpted_x_axis(ns[1:4], x, ys, labels, colors=colors, lines=lines)
        plt.suptitle('(d) non-normalized posterior density', y=1.0)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.15, 0.5))
        fig.text(0.5, 0.0, '$\lambda$', ha='center')

        fig.savefig('my_break_image.pdf', format='pdf', bbox_inches='tight')
        ```
        """

        # if single array/list of values to plot
        try: len(ys[0])
        except: ys = [ys]

        for i in range(3):
            if type(ns[i]) is int:
                print('the first input variable is a list of x values, not of indices -- otherwise, just put it as a list of integers')
            elif type(ns[i]) is float:
                print('the first input variable is a list of indices, not of x values -- otherwise, just put it as a list of float')
                ns[i] = np.argmin(np.abs(x - ns[i]))
        
        # specify color and line
        assert (colors is None) or len(colors) == len(ys), 'error in length colors'
        assert (lines is None) or len(lines) == len(ys), 'error in length lines'
        if colors is None : colors = ['']*len(ys)
        if lines is None : lines = ['']*len(ys)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize, gridspec_kw={'wspace': 0.05,
            'width_ratios': [1, 1]})  # , 'height_ratios': [1, 1]})

        for i, y in enumerate(ys):
            ax1.plot(x[:ns[0]], y[:ns[0]], color=colors[i], linestyle=lines[i], label=labels[i])
            ax1.set_xlim([x[0], x[ns[0]]])

            ax2.plot(x[ns[1]:ns[2]], y[ns[1]:ns[2]], color=colors[i], linestyle=lines[i])
            ax2.set_xlim([x[ns[1]], x[ns[2]]])

        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params(axis='y', left=False)

        # add a diagonal break marker
        delta = 1.
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((delta - d, delta + d), (-d, +d), **kwargs)
        ax1.plot((delta - d, delta + d), (delta - d, delta + d), **kwargs)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (-d, +d), **kwargs)
        ax2.plot((-d, +d), (delta - d, delta + d), **kwargs)

        ax2.yaxis.set_tick_params(labelleft=False)

        ax1.grid()
        ax2.grid()

        plt.tight_layout()

        # plt.suptitle('(d) non-normalized posterior density')

        # for the legend:
        # handles, labels = ax1.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center')
        
        return fig, ax1, ax2



