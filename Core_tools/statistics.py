""" statistics, including: compute Kish size """

import numpy as np
import matplotlib.pyplot as plt

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
        else : position = wh[np.where(np.ediff1d(wh) != 1)][-1][0]

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
            position = wh[np.where(np.ediff1d(wh) != 1)][-1][0]
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


# compute free energy difference
# dE_AB is E(B)-E(A), in unit of temperature if if_adimensional
def compute_DG(temperature,weights_A,dE_AB,if_adimensional=True):

    weights_A = weights_A/np.sum(weights_A)
    
    if if_adimensional: beta_dE_AB = dE_AB
    else: beta_dE_AB = dE_AB/temperature
    
    DG = -temperature*np.log(np.sum(weights_A*np.exp(-beta_dE_AB)))
    
    return DG
