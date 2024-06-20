################################# python3 run for demuxing ######################################

import MDAnalysis as mda
import numpy as np
import time
import os
from pathlib import Path
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

Sequences=['AAAA','CAAU','CCCC','GACC','UUUU']

# for n_seq in [0]:#range(len(Sequences)):
# attention: the n. of replicas may be different for different
n_subtrajs=[310,307,270,310,305]
n_seq=4

n_replicas=24

Sequence=Sequences[n_seq]
curr_dir='/net/sbp/srnas2/tfrahlki/Simulations/%s_TREMD/Production/%s' % (Sequence,Sequence)
print('tetramer: ',Sequence)
print(curr_dir)

#path_ref=curr_dir+'/reference.pdb'

l=[]

for i in range(2,n_subtrajs[n_seq]+1):
    s='000%s'%i
    s=s[-4:]
    l.append(s)

paths_traj=[]
paths_traj.append(curr_dir+'/traj_comp%s.xtc') # % NR
for i in range(len(l)):
    paths_traj.append(curr_dir+'/traj_comp%s.part'+l[i]+'.xtc')
##paths_traj.append('subtraj/subtraj%s.xtc') # % NR
print(paths_traj)

start_frame=1 # starting frame: for each subtrajectory, start from the second frame (depending on file writing)

#max_step=5 # needed if you want to distinguish replicas based on the number of atoms shifted more than max_step
           # not needed if you distinguish replicas based on the mean difference of coordinates

def demux_old(paths_traj,path_ref,max_step,n_replicas,start_frame):
    permutations=[]
    speed=[]
    permutations.append(np.arange(n_replicas))

    univ=mda.Universe(path_ref,paths_traj[0] % 0) # replica 0, just to know the n. of atoms
    ncoord=np.shape(univ.atoms.positions.flatten())[0] # select_atoms or atoms
    print('3N = (with N n. of atoms): ',ncoord)

    # initialization of x
    x=np.zeros((n_replicas,ncoord))

    univ=[]
    for NR in range(0,n_replicas):
        univ.append(mda.Universe(path_ref,paths_traj[0] % NR))
        univ[NR].trajectory[start_frame]
        x[NR,:]=univ[NR].atoms.positions.flatten() # atoms or select_atoms("name MW")

    for i in range(len(paths_traj)): # index of subtrajectory
        univ=[]
        for NR in range(0,n_replicas):
            print(NR)
            # the first time, it is useless
            univ.append(mda.Universe(path_ref,paths_traj[i] % NR))
            #univ.append(mda.Universe(path_ref,paths_traj[0] % NR).select_atoms("name MW"))
            print(univ)
        nframes=len(univ[0].trajectory)
        
        start=time.time()
        for j in range(start_frame+1,nframes):#len(10):#,len(univ)): # for on the frames


            x_new=np.zeros((n_replicas,ncoord))
            dx=np.zeros((n_replicas,n_replicas,ncoord))

            for NR2 in range(n_replicas):
                univ[NR2].trajectory[j]
                x_new[NR2,:]=univ[NR2].atoms.positions.flatten()
                #x_new[NR2,:]=univ[NR2].trajectory[j].positions.flatten()
                
                ###############
                ##for NR in range(n_replicas):
                ##    dx[NR2,NR,:]=np.abs(x[NR,:]-x_new[NR2,:])
                
            ##x=x_new

            ##diff1=np.mean(dx,axis=2)
            ################
            
            # you can also compute the average removing the let's say 20% largest distances
            # or compute distances only for atoms in the bulk
        #print(diff1)

            ##i0,i1,i2=np.where(dx<max_step)
            ##diff2=np.histogram2d(i0,i1,bins=np.arange(n_replicas+1))[0]
        #print(diff2)

            ######## (rows,cols1)=linear_sum_assignment(diff1)

            ##(rows,cols2)=linear_sum_assignment(1/diff2)
            #print(1/diff2)
            if (j%100)==0:
                #print(j)
                #print(diff1)
            ##    print(permutations[-1])
            ##    print(cols2)
                #print(time.time()-start)
                times=time.time()-start
                speed.append(j/times)
                print('speed (n. frames/sec.): ',speed[-1])
                
            #if np.array_equal(cols1,cols2):
            #########permutations.append(permutations[-1][cols1])
            #else:
            #    print('error')
            #    break
    
        #####np.savetxt(os.path.join(userdoc,"demuxed"),permutations,delimiter=',')
        ########np.savetxt(os.path.join(userdoc,"speed"),speed,delimiter=',')
    return permutations,speed

#import numba
#@numba.jit
def demux(n_replicas,paths_traj,start_frame,userdoc,N=100): # if_local
    permutations=[]
    times=[]
    check=[]
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

        #def f(x1,x2):
        #    return np.sum(1/(np.exp(5-(x1-x2))+1))

        for n_frame in range(n_frame0,len_subtraj):
            x_new=[]
            for NR in range(n_replicas):
                x_new.append(xtc_read[NR][n_frame][:].flatten())
            x_new=np.array(x_new)

            #start2=time.time()
            diff = distance.cdist(x,x_new,"sqeuclidean")
            #print('time1:',time.time()-start2) # 0.027 sec.
            
            #start2=time.time()
            #diff = distance.cdist(x,x_new,f)#lambda u, v: np.sum(1/(np.exp(5-(u-v))+1)))
            #print('time2:',time.time()-start2) # 0.38 sec.
            #print(diff)
            
            x=x_new

            check.append(np.max((np.min(diff,axis=1)-np.partition(diff,1,axis=1)[:,1])/np.min(diff,axis=1)))

            #print((np.min(diff,axis=1)-np.partition(diff,1,axis=1)[:,1])/np.min(diff,axis=1))
            #print(check)

            #print(np.min(diff,axis=1)/1e6)
            #print(np.partition(diff,1,axis=1)[:,1]/1e6)
            print(diff/1e6)

            (rows,cols)=linear_sum_assignment(diff)
            # or equivalently:
            #cols=np.argmin(diff,axis=1)
            print(cols)

            permutations.append(permutations[-1][cols])
            print(permutations[-1][cols])

            if (n_frame%N)==0:
                times.append(time.time()-start)
                #print(times[-1])
        
        np.savetxt(userdoc+'/replica_temp',permutations,delimiter=',')
        np.savetxt(userdoc+'/time',times,delimiter=',')
        np.savetxt(userdoc+'/check',check,delimiter=',')
    #return permutations,time
    print('done')


userdoc = os.path.join(os.path.expanduser("~"),'2_tetramers/demuxing%s' % Sequences[n_seq])
Path(userdoc).mkdir(parents=True, exist_ok=True)

demux(n_replicas,paths_traj,start_frame,userdoc)