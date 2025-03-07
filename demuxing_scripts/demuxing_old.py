# %% [markdown]
# ## Demuxing

# %% [markdown]
# see Thorben's thesis:
# - 24 replicas from 275 K to 400 K
# - 1 micros per replica
# - 300 K corresponds to replica 6 (counting from 1)

# %%
import numpy as np
import time
import MDAnalysis as mda
from scipy.spatial import distance
import sys

# %% [markdown]
# ## Input data
# General data for all the tetramers:
# - their names (Sequences)
# - the n. of subtrajectories
# - the n. of replicas (in this case, it should be equal for all the tetramers)

# %%

Sequences=['UCAAUC','UCUCGU']
js=[310,100] # n. of subtrajectory

#n_seq=int(sys.argv[1])
n_seq=0
Sequence=Sequences[n_seq]
n_max_subtraj=2#js[n_seq]

#curr_dir='/net/sbp/srnas2/tfrahlki/Simulations'

# to see the n. of subtrajectories of AAAA let's say:
# ll traj6.AAAA.*proc.xtc

# to see the n. of replicas (choose a fragment of trajectory, let's say 0176):
# ll traj*.part0176.xtc
n_replicas=24

# %% [markdown]
# Input data for demuxing:
# - path_ref: the path of the reference (topology .pdb), useless if you extract coordinates directly from .xtc
# - paths_traj: the paths of the trajectories
# - start_frame: skip the first frame, because it is equal to the last of the previous subtraj or because it is the first subtraj and you skipped it
# - max_step: useful only if you evaluate the n. of atoms moving less than max_step

# %%
# for n_seq in [0]:#range(len(Sequences)):
# attention: the n. of subtrajectories may be different for different

#Sequence=Sequences[n_seq]
curr_dir='/net/sbp/srnas2/tfrahlki/Simulations/%s_TREMD/Production/%s' % (Sequence,Sequence)
print('tetramer: ',Sequence)
print(curr_dir)

path_ref=curr_dir+'/reference.pdb'

l=[]
for i in range(2,n_max_subtraj+1):
    s='000%s'%i
    s=s[-4:]
    l.append(s)

paths_traj=[]
paths_traj.append(curr_dir+'/traj_comp%s.xtc') # % NR
for i in range(len(l)):
    paths_traj.append(curr_dir+'/traj_comp%s.part'+l[i]+'.xtc')



# %%
start_frame=1

# max_step=5


# %% [markdown]
# # Algorithm
# Assumption: thermodynamic limit
# 
# input:
# - n_replicas
# - paths_traj: the paths of the trajectories (with %s for the different replicas)
# - start_frame: starting frame for each subtraj (simple case of frames to skip: the i-th of each subtrajectory)
# - if_local: if you want to copy subtrajectories in the same folder of your python script, to run faster (?, anyway you can choose only MW atoms, for instance)
# - N: every N frames save required time
# - userdoc: the folder path where you save output data
# 
# algorithm:
# - permutations = np.arange(n_replicas) # first frame: 1,2...n_replicas
# - for n_sub in range(N_subtraj):
#     - if if_local: copy subtraj(n_sub) in your local folder (as .xtc)
#     - if n_sub==0: initialize first frame x 
#     - for n_frame in range(start_frame+1,len(subtraj)):
#         - x_new = new frame 
#         - cost = distance.cdist(x,x_new,"sqeuclidean")
#         - (rows,cols1)=linear_sum_assignment(diff1) # or equivalently cols1=np.argmin(diff1,axis=1)
#         - permutations.append(permutations[-1][cols2])   
# 
# Two ways to read coordinates from frames:
# - 1st way:
#     - univ=mda.Universe(path_ref,paths_traj[0] % NR)
#     - univ.trajectory[n_frame]
#     - x1=univ.atoms.positions
# - 2nd way: (it seems better)
#     - xtc_read=mda.coordinates.XTC.XTCReader(paths_traj[0] % NR)
#     - x1=+xtc_read[n_frame][:]
#     - here x=+xtc_read[:][:] is the array of coordinates (all atoms, all frames)
# 
# output:
# - demuxed frame indices
# - also the time required for each N frames
# 

# %%
print(paths_traj)
print(start_frame)

# %%
#@numba.jit
def demuxing(Sequence,n_replicas,paths_traj,start_frame,userdoc,N=10000): # if_local
    permutations=[]
    times=[]
    permutations.append(np.arange(n_replicas))

    start=time.time()

    for n_subtraj in range(len(paths_traj)):
        
        xtc_read=[] # xtc_read[NR][n_frame][:]
        for NR in range(n_replicas):
            xtc_read.append(mda.coordinates.XTC.XTCReader(paths_traj[n_subtraj] % NR))
                
        n_frame0=start_frame
        if n_subtraj==0:
            x=[]
            for NR in range(n_replicas):
                x.append(xtc_read[NR][start_frame][:].flatten())
            x=np.array(x)
            n_frame0=start_frame+1 # if already initialised, start from next frame
        
        len_subtraj=len(xtc_read[0])
        for n_frame in range(1,2):#n_frame0,len_subtraj):
            x_new=[]
            for NR in range(n_replicas):
                x_new.append(xtc_read[NR][n_frame][:].flatten())


            diff = distance.cdist(x,x_new,"sqeuclidean")
            #print('differences: ',diff/1e7)

            x=np.array(x_new)
            #(rows,cols)=linear_sum_assignment(diff)
            # or equivalently:
            cols=np.argmin(diff,axis=1)

            permutations.append(permutations[-1][cols])

            if (n_frame%N)==0:
                times.append(time.time()-start)
                print('n frames: ',n_frame)

    np.savetxt(userdoc+'/demuxed%s' % Sequence,permutations,delimiter=',')
    np.savetxt(userdoc+'/time%s' % Sequence,times,delimiter=',')

# %% [markdown]
# ### do demux

# %%
import os
from pathlib import Path

#userdoc = os.path.join(os.path.expanduser("~"),'2_tetramers/demuxing')
userdoc='demuxing_hexamers'
Path(userdoc).mkdir(parents=True, exist_ok=True)

demuxing(Sequence,n_replicas,paths_traj,start_frame,userdoc)
