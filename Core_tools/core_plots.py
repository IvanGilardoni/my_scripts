""" my_plots
# it includes:
# - two_scale_plot (plot with 2 y scales)

# see also:
# https://github.com/cxli233/FriendsDontLetFriends/tree/main?tab=readme-ov-file#friends-dont-let-friends-use-boxpot-for-binomial-data
"""

import numpy as np
import matplotlib.pyplot as plt
from .basic import unwrap_dict

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



