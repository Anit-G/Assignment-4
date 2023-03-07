import numpy as np
from scipy.spatial.distance import cdist,euclidean, hamming
import pickle
import matplotlib.pyplot as plt

def cost_fn(a,b):
    return euclidean(a,b)

def DTW_distance(s,t,method='euclidean'):
    """
        Args:
            s:
                The sample sequence that the cost is being measured against
            t:
                The target sequence that the cost is being measured for

        Return:
            Distance cost:
                Returns the normalized distance cost using Dynamic Time Warping of the cost matrix
    https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html
    """

    n, m = s.shape[1],t.shape[1]
    DTW = np.zeros((n+1,m+1))

    # Initialize the cost matrix boundaries
    DTW[:,0] = np.inf
    DTW[0,:] = np.inf
    DTW[0,0] = 0

    # perform Dynamic time warping:
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = cost_fn(s[:,i-1],t[:,j-1])
            DTW[i,j] = cost + min(DTW[i-1,j],DTW[i-1,j-1],DTW[i,j-1])
    
    return DTW[n,m]/(n+m)

def cost_matrix(a,b,method='euclidean'):
    """
        Compute the cost matrix using a and b using a euclidean distance
    """
    a,b = np.atleast_2d(a,b)
    C = cdist(a.T,b.T,metric=method)
    return C

def DTW_align(s,t,method='m1'):
    """
        Align t to s using DTW and find alignment cost

        Args:
            s:      Sequence
            t:      Target
            method: 
                Method used when warping the target to sequence using 
                the aligned path found from DTW

                Note: m1 method picks the last value from the targets
                      m2 method takes the average values of the targets

                      m2 yields a slightly lesses distance cost otherwise
                      they are the same

        Return:
            op: the optimal warping path
    """
    C = cost_matrix(s,t)
    N,M = C.shape

    # Initialize the matrix
    DTW = np.zeros((N,M))
    DTW[1:,0] = C[1:,0]+DTW[:-1,0]
    DTW[0,1:] = C[0,1:]+DTW[0,:-1]

    # Perform DTW
    for i in range(1,N):
        for j in range(1,M):
            DTW[i,j] = C[i,j] + min(DTW[i-1,j],DTW[i-1,j-1],DTW[i,j-1])

    # Find optimal path through the DTW matrix/search trills
    n = N-1
    m = M-1
    op = [(n,m)]
    while n>0 or m>0:
        if n==0:
            cl = (0,m-1)
        elif m==0:
            cl = (n-1,0)
        else:
            val = min(DTW[n-1,m],DTW[n-1,m-1],DTW[n,m-1])
            if val == DTW[n-1,m-1]:
                cl = (n-1,m-1)
            elif val == DTW[n-1,m]:
                cl = (n-1,m)
            else:
                cl = (n,m-1)
        op.append(cl)
        (n,m) = cl
    op.reverse()
    
    # compute the aligned target
    aligned = np.zeros((s.shape[0],s.shape[1]))

    if method=='m1':
        # Method based on taking the last value
        for i,(n,m) in enumerate(op):
            aligned[:,n] = t[:,m]

    elif method=='m2':
        # Method Based on averaging across horzontal lines in the path
        op = np.array(op)
        temp = np.split(op[:,1], np.unique(op[:, 0], return_index=True)[1][1:])
        for i,k in enumerate(temp):
            aligned[:,i] = np.mean([t[:,j] for j in k],axis=0)
    else:
        print("WARNING The alignment method choosen in not specified")

    alignment_cost = sum([C[i,j] for (i,j) in op])/(N+M)    # The normalized alignment cost
    dist_cost = DTW[-1,-1]/(N+M)                             # The normalized distance cost    

    return DTW, aligned, np.array(op), alignment_cost, dist_cost

def DTW_template_alignment(m,ts):
    """
        We will be doing template alignment of ts to m

        Args:
            m:
            Master choosen from the avaliable templates (by default choose the one with median number of frames)
            
            ts:
            Targets or all the other templates that need to be aligned to the master
        
        Return:
            All the aligned templates
    """
    temp = [m]
    for t in ts:
        D, al_t, p, acost, dcost = DTW_align(m,t)
        temp.append(al_t)

    # average the Vectors across the templates
    temp = np.array(temp)

    return np.mean(temp,axis=0)

if __name__ == "__main__":
    # c = DTW_distance([1, 3, 9, 2, 1],[2, 0, 0, 8, 7, 2])
    # print(c)

    with open('Team7_data.txt','rb') as f:
        Data = pickle.load(f)
        f.close() 
    
    s = Data['2_dev']['nh_2.wav']
    t = Data['2_dev']['ra_2.wav']
    method = 'm1'


    DTW, aligned_t, path, al_cost, dist_cost = DTW_align(s,t,method)
    print("Statistics of Initial Alignment")
    print(f"Sequence Shape: {s.shape}\nTarget Shape: {t.shape}")
    print(f"Alignment Cost: {al_cost}")
    print(f"Distance Cost: {dist_cost}")
    print(f"Aligned Target: {aligned_t.shape}")
    print(f"Optimal Path Length: {len(path)}")

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    ax1.imshow(DTW,cmap='gray_r', origin='lower', aspect='equal')
    ax1.plot(path[:,1],path[:,0], marker='o', color='r')
    ax1.set_title(f'Search trilles with the Optimal Path\nAlignment Cost: {al_cost:.3f}\nDistance Cost: {dist_cost:.3f}')
    ax1.set_xlabel('Target')
    ax1.set_ylabel('Sequence')


    DTW, aligned_t2, path, al_cost, dist_cost = DTW_align(s,aligned_t,method)
    print("\n\nStatistics for Validation with aligned_t")
    print(f"Aligned Target Shape: {aligned_t.shape}")
    print(f"Alignment Cost: {al_cost}")
    print(f"Distance Cost: {dist_cost}")
    print(f"New Aligned Target: {aligned_t2.shape}")
    print(f"Optimal Path Length: {len(path)}")

    ax2.imshow(DTW,cmap='gray_r', origin='lower', aspect='equal')
    ax2.plot(path[:,1],path[:,0], marker='o', color='r')
    ax2.set_title(f'Search trilles with the Optimal Path for\nValidation Check of Aligned Target\nAlignment Cost: {al_cost:.3f}\nDistance Cost: {dist_cost:.3f}')
    ax2.set_xlabel('Target')
    ax2.set_ylabel('Sequence')

    plt.figure()
    DTW, _, path, al_cost, dist_cost = DTW_align(s,s)
    plt.imshow(DTW,cmap='gray_r', origin='lower', aspect='equal')
    plt.plot(path[:,1],path[:,0], marker='o', color='r')
    plt.title(f'Search trilles with the optimal path for Self Check\nAlignment Cost: {al_cost:.3f}\nDistance Cost: {dist_cost:.3f}')
    plt.xlabel('Target')
    plt.ylabel('Sequence')
    plt.show()