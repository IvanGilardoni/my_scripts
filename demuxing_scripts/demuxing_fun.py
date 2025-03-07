""" demuxing """

import time, os
import numpy as np
import MDAnalysis as mda
from scipy.spatial import distance

#%% define main function

def demuxing(threshold, n_replicas, sorted_paths_traj, if_skip_first_frame, path_print, n_print = 10000):
    """
    This function is based on computing distances between couples (before, after).
    However, if there are PBCs, 'sqeuclidean' is not the correct thing to do:
    rather, you can compute the n. of particles moving less than a threshold
    and then do linear assignment based on that.
    In this way, giving the same weight to each displacement bigger than the threshold,
    you will manage to get rid of the particles close to the boundary (rho*L^3 vs. rho*L^2*delta).
    
    Notice that distance.cdist requires x, x_new to have shape (M x N) where M is the n. of replicas
    and N is the n. of total coordinates.

    ----------------
    Parameters:

    threshold : float or str
        If float, the function used for `distance.cdist` (which is not a distance) is:
        count how many particles move less than `threshold`.
        If string, use `distance.cdist` with `metric = threshold` (e.g., `'sqeuclidean'`).

    n_replicas : int
        The n. of replicas.

    sorted_paths_traj : list
        Sorted list of string with the path of the subtrajectories.

    if_skip_first_frame : bool
        Boolean, if True skip the first frame of each subtrajectory (except the first subtrajectory)
        because it is the same as the last of the previous subtrajectory

    path_print : str
        Path where the output will be saved.

    n_print : int
        How often (as n. of frames) to print progression in the demuxing.
    """

    start = time.time()

    repl_indices = []
    times = []

    repl_indices.append(np.arange(n_replicas))

    if (type(threshold) is float) or (type(threshold) is int):
        my_fun = lambda x1, x2 : len(np.where(np.abs(x1 - x2) < threshold)[0])
        """ ensure it is big for close frames (otherwise you will have to change argmax to argmin) """
        # return  # just to ensure it is argmin (without an additional assert repeated over each frame)
    else:
        my_fun = threshold  # 'sqeuclidean' for instance
        print('in this case you must change argmax to argmin!')
        return

    for n_subtraj in range(2):  # len(sorted_paths_traj)):

        print('subtrajectory n. ', n_subtraj)
        
        xtc_read = []  # xtc_read[NR][n_frame][:]
        for n_rep in range(n_replicas):
            xtc_read.append(mda.coordinates.XTC.XTCReader(sorted_paths_traj[n_subtraj] % n_rep))

        if n_subtraj == 0:
            x = []
            for n_rep in range(n_replicas):
                x.append(xtc_read[n_rep][0][:].flatten())
            x = np.array(x)

        # skip the first frame if if_skip_first_frame for all the subtrajectories after the first one
        # and also for the first subtrajectory but because you have already read the first frame
        if (if_skip_first_frame and (n_subtraj != 0)) or (n_subtraj == 0): n_frame0 = 1
        else: n_frame0 = 0

        len_subtraj = len(xtc_read[0])
        
        for n_frame in range(n_frame0, len_subtraj):
            x_new = []
            for n_rep in range(n_replicas):
                x_new.append(xtc_read[n_rep][n_frame][:].flatten())
            
            diff = distance.cdist(x, x_new, my_fun)
            """ diff[i, j] is the distance between x[i] and x_new[j] """

            x = np.array(x_new)

            ind = np.argmax(diff, axis=0)
            # ind = np.argmin(diff, axis=0)
            """ ind[i] is the index of x_new which is the closest to x[i]
            ind[i] with axis=0 answers to: "where does x_new[i] come from?" 
            instead ind[i] with axis=1 answers to: "where does x_new[i] go to?
            so the correct one is axis=0"
            """

            # you should add a check on unique assignment!
            assert len(np.unique(ind)) == n_replicas
            # if len(np.unique(ind)) != n_replicas:
            #     print('error')

            repl_indices.append(repl_indices[-1][ind])

            if (n_frame % n_print) == 0:
                times.append(time.time() - start)
                print('n frames: ', n_frame)

    """ `repl_indices` is replica_index (its columns are themperatures) """
    np.savetxt(path_print + '/replica_index', repl_indices, fmt='%i', delimiter=',')
    np.savetxt(path_print + '/time', times, delimiter=',')
    np.savetxt(path_print + '/n_subtraj', np.array([n_subtraj]), fmt='%d')

    print(n_subtraj)

    return np.array(repl_indices)

#%% define path for subtrajectories

sequence = 'CCCC'
n_replicas = 24

curr_dir = '/net/sbp/srnas2/tfrahlki/Simulations/%s_TREMD/Production/%s' % (sequence, sequence)

paths_traj = [s for s in os.listdir(curr_dir) if s.startswith('traj_comp0')]  #  and s.endswith('0002.xtc'))]

paths_traj.sort()

# first element is traj_comp0.xtc
paths_traj.insert(0, paths_traj[-1])
paths_traj = paths_traj[:-1]

# substitute replica number with %s
for i in range(len(paths_traj)):
    index = 9
    replacement = '%s'

    text_list = list(paths_traj[i])
    text_list[index] = replacement

    paths_traj[i] = ''.join(text_list)

paths_traj = [(curr_dir + '/' + s) for s in paths_traj]

#%% run demuxing

path_print = '../../demuxing_results'

my_rep_index = demuxing(1e-3, n_replicas, paths_traj, True, path_print, n_print=10)
