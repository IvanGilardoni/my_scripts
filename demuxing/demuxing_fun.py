""" demuxing function """

import time, os, datetime
import numpy as np
import MDAnalysis as mda
from scipy.spatial import distance

#%% reciprocal lattice

def reciprocal_lattice(ts : np.ndarray, which_return : str = ''):
    """compute the primitive vectors of the reciprocal lattice and their sum
    
    ----------------
    Parameters:

    ts : numpy.ndarray
        Numpy array of the triclinic lattice systems: namely, `(a, b, c, alpha, beta, gamma)` where
        `a, b, c` are the lengths of the primitive vectors of the real-space lattice and
        `alpha, beta, gamma` are the corresponding opposite angles in sexagesimal degree
        (`alpha` is the angle between primitive vectors of length `b, c` and so on for the others).
        It is returned by `MDAnalysis.coordinates.XTC.XTCReader(xtc_path).ts`.

    which_return : str
        If `'primitive vectors'`, then return only the matrix with the primitive vectors of the real-space lattice.
        Elif `'sum'`, then return only the sum of the three primitive vectors of the reciprocal lattice.
        Otherwise return: the first two outputs and also the primitive vectors of the reciprocal lattice.
    """

    ts = ts.astype(np.float64)
    ts[-3:] = ts[-3:]/180*np.pi

    a1 = np.array([ts[0], 0, 0])
    a2 = ts[1]*np.array([np.cos(ts[-1]), np.sin(ts[-1]), 0])

    norm_c_x = np.cos(ts[-2])
    norm_c_y = (np.cos(ts[-3]) - np.cos(ts[-2])*np.cos(ts[-1]))/np.sin(ts[-1])
    norm_c_z = np.sqrt(1 - norm_c_x**2 - norm_c_y**2)

    a3 = ts[2]*np.array([norm_c_x, norm_c_y, norm_c_z])

    matrix = np.zeros((3, 3))
    matrix[:, 0] = a1
    matrix[:, 1] = a2
    matrix[:, 2] = a3

    if which_return == 'primitive vectors': return matrix

    cell_vol = ts[0]*ts[1]*a3[2]*np.sin(ts[-1])

    # always equal to:
    cell_vol_check = np.dot(a1, np.cross(a2, a3))
    assert np.abs(cell_vol - cell_vol_check) < 1e-6, 'error in cell volume: %f != %f' % (cell_vol, cell_vol_check)

    # always equal to:
    cell_vol_check = np.linalg.det(matrix)
    assert np.abs(cell_vol - cell_vol_check) < 1e-6, 'error in cell volume: %f != %f' % (cell_vol, cell_vol_check)

    k = np.array([ts[1]*a3[2]*np.sin(ts[-1]), (ts[1]*np.cos(ts[-1]) + ts[0])*a3[2], 
        ts[0]*(ts[1]*np.sin(ts[-1]) - a3[1]) + ts[1]*(a3[1]*np.cos(ts[-1]) - a3[0]*np.sin(ts[-1]))])

    k = k*2*np.pi/cell_vol

    if which_return == 'sum': return k
    else:

        # reciprocal lattice vectors
        g1 = 2*np.pi/cell_vol*np.cross(a2, a3)
        g2 = 2*np.pi/cell_vol*np.cross(a3, a1)
        g3 = 2*np.pi/cell_vol*np.cross(a1, a2)

        # always equal to:
        k_check = g1 + g2 + g3
        assert np.sum((k - k_check)**2) < 1e-6, 'error in the sum of reciprocal primitive vectors: %f != %f' % (k, k_check)

        g_matrix = np.zeros((3, 3))
        g_matrix[:, 0] = g1
        g_matrix[:, 1] = g2
        g_matrix[:, 2] = g3
        return matrix, g_matrix, k


def cosine_distance(x1, x2, rec_vec):
    """cosine distance for triclinic systems, defined as `1 - cos(rec_vec*diff)`, summed over all the particles
    
    ------------------
    Parameters:
        x1, x2 : numpy.ndarray
            Numpy 1d. arrays with the three coordinates x, y, z of each particle at positions 0, 1, 2 mod. 3
        
        rec_vec : numpy.ndarray
            Numpy 1d. array with the sum of the three primitive vectors of the reciprocal lattice.    
    """

    # unit_length = 51.0491
    diff = np.abs(x2 - x1)
    # arg = 2*np.pi*(diff[::3] + diff[1::3] + np.sqrt(2)*diff[2::3])/unit_length
    
    arg = rec_vec[0]*diff[::3] + rec_vec[1]*diff[1::3] + rec_vec[2]*diff[2::3]
    cos_dists = 1 - np.cos(arg)
    cos_dist = np.mean(cos_dists)
    # cos_dist = 1 - np.cos(np.sum(arg))

    return cos_dist  # , cos_dists


def periodical_distance(x1, x2, matrix, inv_matrix):
    # for single particles:
    # s_12 = np.dot(inv_matrix, x2 - x1)
    # s_12 -= np.round(s_12)
    # dist = np.linalg.norm(np.dot(matrix, s_12))

    # else:
    diff = x2 - x1
    diff = np.reshape(diff, (-1, 3))

    s_12 = np.dot(inv_matrix, diff.T)
    s_12 -= np.round(s_12)

    dist = np.mean(np.linalg.norm(np.dot(matrix, s_12), axis=0))

    return dist

#%% define main functions

def demuxing_single_move(x, x_new, my_fun):

    diff = distance.cdist(x, x_new, my_fun)
    """ diff[i, j] is the distance between x[i] and x_new[j] """

    ind = np.argmin(diff, axis=0)
    """ ind[i] is the index of x_new which is the closest to x[i]
    ind[i] with axis=0 answers to: "where does x_new[i] come from?" 
    instead ind[i] with axis=1 answers to: "where does x_new[i] go to?
    so the correct one is axis=0"
    """

    return ind


def demuxing(distance_fun, n_replicas, sorted_paths_traj, if_skip_first_frame, path_print, n_print = 10000):
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

    threshold : str or tuple
        If string, use `distance.cdist` with `metric = threshold` (e.g., `'sqeuclidean'`).
        Else, `threshold` is a tuple; it includes:
            - `('count', value)` with `value` a float (count how many particles move more than `value`)
            - `('periodical', ts)` with `ts` the numpy array for the triclinic lattic systems
            (euclidean distance with Periodic Boudary Conditions)
            - `('cos', ts)` with `ts` the numpy array for the triclinic lattice system ("cosine distance",
            taking into account the PBCs).

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

    """ 1. define the distance used """
    assert (type(distance_fun) is str) or (type(distance_fun) is tuple), 'error in threshold'

    if type(distance_fun) is str:
        my_fun = distance_fun  # 'sqeuclidean' for instance
    else:
        if distance_fun[0] == 'count':
            my_fun = lambda x1, x2 : 1/len(np.where(np.abs(x1 - x2) < distance_fun[1])[0])
            """ ensure this def. to be small for close frames
            (otherwise you will have to change argmax to argmin) """
        
        elif distance_fun[0] == 'cos':
            rec_vec = reciprocal_lattice(distance_fun[1], 'sum')
            my_fun = lambda x1, x2 : cosine_distance(x1, x2, rec_vec)

        elif distance_fun[0] == 'periodical':
            h = reciprocal_lattice(distance_fun[1], 'primitive vectors')
            h_inv = np.linalg.inv(h)
            my_fun = lambda x1, x2 : periodical_distance(x1, x2, h, h_inv)

    """ 2. cycle over the subtrajectories and do demuxing with the distance defined above """

    if not os.path.isdir(path_print): os.mkdir(path_print)
    if path_print[-1] != '/': path_print = path_print + '/'

    s = datetime.datetime.now()
    date = s.strftime('%Y_%m_%d_%H_%M_%S')

    for n_subtraj in range(len(sorted_paths_traj)):

        print('subtrajectory n. %s / %s' % (str(n_subtraj), str(len(sorted_paths_traj))))
        
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
            
            ind = demuxing_single_move(x, x_new, my_fun)

            # you should add a check on unique assignment!
            assert len(np.unique(ind)) == n_replicas
            # if len(np.unique(ind)) != n_replicas:
            #     print('error')

            x = np.array(x_new)

            repl_indices.append(repl_indices[-1][ind])

            if (n_frame % n_print) == 0:
                times.append(time.time() - start)
                print('n frames: ', n_frame)

        """ `repl_indices` is replica_index (its columns are temperatures) """

        # save at the end of each subtrajectory and overwrite to the previous saved file
        np.savetxt(path_print + 'replica_index_' + date, repl_indices, fmt='%i', delimiter=',')
        np.savetxt(path_print + 'time_' + date, times, delimiter=',')
        np.savetxt(path_print + 'n_subtraj_' + date, np.array([n_subtraj]), fmt='%d')

        print(n_subtraj)

    return np.array(repl_indices)


