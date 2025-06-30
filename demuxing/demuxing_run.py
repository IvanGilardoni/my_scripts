import os
import numpy as np
import MDAnalysis as mda

from demuxing_fun import demuxing

#%% define path for subtrajectories

sequence = 'GACC'
n_replicas = 24

curr_dir = '/net/sbp/srnas2/tfrahlki/Simulations/%s_TREMD/Production/%s' % (sequence, sequence)

paths_traj = [s for s in os.listdir(curr_dir) if s.startswith('traj_comp0')]  #  and s.endswith('0002.xtc'))]
paths_traj = [s for s in paths_traj if not (s.endswith('full_proc.xtc') or s.endswith('full.xtc'))]

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

path_print = '../../demuxing_results_%s' % sequence

# dim = np.array([51.0491, 51.0491, 51.049175, 59.999985, 59.999985, 90.])
dim = mda.coordinates.XTC.XTCReader(paths_traj[0] % 0).ts.dimensions

my_rep_index = demuxing(('count', 1e-1), n_replicas, paths_traj, True, path_print, n_print=1000)
