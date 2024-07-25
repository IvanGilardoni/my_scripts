""" User-defined Python functions employed in basic staffs """


import numpy as np

#%% 0. USEFUL READINGS
'''

- how to avoid overwriting and make copy correctly: https://realpython.com/copying-python-objects/
 BE CAREFUL: if you deepcopy a class, then this does not mean you are doing a deepcopy of its attributes;
 for example, if class has a dictionary has an attribute, then you are not doing a deepcopy of dictionary!!!

'''
#%% 1. BASIC OPERATIONS WITH LIST, DICTIONARIES AND SO ON
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

import matplotlib.pyplot as plt

class my_plots():

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

            fig = plt.subplots(figsize=(12, 8)) 

            brs = []
            brs.append(np.arange(len(df.iloc[:, 0])))
    
            plt.bar(brs[-1], df.iloc[:, 0], label = df.columns[0], width = barWidth) # edgecolor ='grey', color ='tab:blue')

            for i in range(1, len(df.columns)):
                    brs.append([x + barWidth for x in brs[-1]])
                    plt.bar(brs[-1], df.iloc[:, i], label=df.columns[i], width = barWidth) # edgecolor ='grey', color ='tab:blue')

            plt.xticks([r + barWidth for r in range(len(df.iloc[:, 0]))], list(df.index))
            # plt.xlabel(list(df.index))
            
            # plt.xlabel('Branch', fontweight ='bold', fontsize = 15) 
            # plt.ylabel('Students passed', fontweight ='bold', fontsize = 15) 
            # plt.xticks([r + barWidth for r in range(len(df['Aduri'].iloc[:-1]))], names_charges)

            plt.legend()
            plt.gca().xaxis.grid(True)
            # plt.grid()
            # plt.show()

            return

    def plot_double_legend(x,my_dict,*,add_label2='',loc1=None,loc2=None):
        colors = my_plots.default_colors()
        markers = my_plots.default_markers()
        for i in range(len(markers)): markers[i] = markers[i]+'--'

        plt.figure()
        plot_lines = []
        keys1 = list(my_dict.keys())
        keys2 = {}
        for i,name1 in enumerate(keys1):
            l = {}
            keys2[name1] = list(my_dict[name1].keys())
            for j,name2 in enumerate(keys2[name1]):

                vec = my_dict[name1][name2]
                l[name2], = plt.plot(x,vec,markers[j],color=colors[i])
                
            plot_lines.append(unwrap_dict(l))

        if loc1 is not None: loc = loc1
        else: loc = 2
        legend1 = plt.legend(plot_lines[0], keys2[keys1[0]], loc=loc)
        plt.gca().add_artist(legend1)

        if loc2 is not None: loc = loc2
        else: loc = 3
        labels = [add_label2+str(keys1[i]) for i in range(len(keys1))]
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

        ax = plt.figure(figsize=(10,8)).add_subplot(projection = '3d')
        plt.title(r'gradient descent of $\min\,\chi^2(\alpha,\beta,\gamma)$', fontsize = 20)

        ax.set_xlabel(r'$\log\,\alpha$', fontsize = 20)
        # ax.tick_params(labelleft = False, left = False)
        # ax.set_xticks("")#color = 'w')
        ax.set_ylabel(r'$\log\,\beta$', fontsize = 20)
        ax.set_zlabel(r'$\log\,\gamma$', fontsize = 20)
        ax.plot(x, y, z, '.', label='parametric curve')

        ax.plot(x[-1], y[-1], z[-1], 'Xk', markersize = 10)#, color = 'b')
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

            a = Arrow3D([x[i],x[i+1]],[y[i],y[i+1]],[z[i],z[i+1]], mutation_scale = 10, lw = 1, arrowstyle = '-|>', color = 'r')
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


''' Preliminary tests for Python packages: 

    - name: Pyflakes
      run: |
        pip install --upgrade pyflakes
        pyflakes MDRefine
    - name: Pylint
      run: |
        pip install --upgrade  pylint
        pylint -E MDRefine
    - name: Flake8
      run: |
        pip install --upgrade flake8
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
        flake8 MDRefine bin --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics | tee flake8_report.txt

'''
