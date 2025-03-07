import numpy as np
import time
import MDAnalysis as mda
from scipy.spatial import distance
import pandas

# %%
Sequence='UCUCGU'
n_replicas=24

curr_dir='/net/sbp/srnas2/tfrahlki/Simulations/%s_TREMD/Production/%s' % (Sequence,Sequence)

n_max_subtraj=100 # 310 for UCAAUC, 100 for UCUCGU (?)

start_frame=1

# %%

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

def demuxing(n_replicas,paths_traj,start_frame,userdoc,N=10000): # if_local
    permutations=[]
    times=[]
    permutations.append(np.arange(n_replicas))

    start=time.time()

    my_fun = lambda x1, x2 : len(np.where(np.abs(x1 - x2) < 0.1)[0])

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
        for n_frame in range(n_frame0,len_subtraj):
            x_new=[]
            for NR in range(n_replicas):
                x_new.append(xtc_read[NR][n_frame][:].flatten())


            # diff = distance.cdist(x,x_new,"sqeuclidean")
            diff = distance.cdist(x, x_new, my_fun)

            # If there are PBCs, 'sqeuclidean' is not the correct thing to do:
            # rather, you can compute the n. of particles moving less than a threshold
            # and then do linear assignment based on that.
            # In this way, giving the same weight to each displacement bigger than the threshold,
            # you will manage to get rid of the particles close to the boundary (rho*L^3 vs. rho*L^2*delta).
            # Notice that distance.cdist requires x, x_new to have shape (M x N) where M is the n. of replicas
            # and N is the n. of total coordinates.

            #print('differences: ',diff/1e7)

            x=np.array(x_new)
            #(rows,cols)=linear_sum_assignment(diff)
            # or equivalently:
            cols=np.argmin(diff,axis=1)

            permutations.append(permutations[-1][cols])

            if (n_frame%N)==0:
                times.append(time.time()-start)
                print('n frames: ',n_frame)

        np.savetxt(userdoc+'/demuxed',permutations,fmt='%d',delimiter=',')
        np.savetxt(userdoc+'/time',times,delimiter=',')
        np.savetxt(userdoc+'/n_subtraj',np.array([n_subtraj]),fmt='%d')
        print(n_subtraj)

# %%

import os
from pathlib import Path

#userdoc = os.path.join(os.path.expanduser("~"),'2_tetramers/demuxing')
userdoc='demuxing_%s' % Sequence
Path(userdoc).mkdir(parents=True, exist_ok=True)

permutations, times = demuxing(n_replicas,paths_traj,start_frame,userdoc,N=1000)