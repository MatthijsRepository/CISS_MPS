import numpy as np
from MPS_TimeOp import MPS

def initialize_halfstate(N, d, chi):
    """ Initializes the MPS into a product state of uniform eigenstates """
    Lambda_mat = np.zeros((N+1,chi))
    Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
    
    Lambda_mat[:,0] = 1
    Gamma_mat[:,:,0,0] = 1/np.sqrt(d)
    
    #.locsize[:,:] = 1
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum(d**arr, chi)
    locsize[np.where(locsize==0)] = chi
    return Gamma_mat, Lambda_mat, locsize

def initialize_flipstate(N, d, chi):
    """ Initializes the MPS into a product of alternating up/down states """
    Lambda_mat = np.zeros((N+1,chi))
    Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
    
    Lambda_mat[:,0] = 1
    
    for i in range(0,N,2):
        Gamma_mat[i,0,0,0] = 1
    for i in range(1,N,2):
        Gamma_mat[i,d-1,0,0] = 1
           
    #.locsize[:,:] = 1
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum(d**arr, chi)
    locsize[np.where(locsize==0)] = chi
    return Gamma_mat, Lambda_mat, locsize

def initialize_up_or_down(N, d, chi, up):
    """ Initializes the MPS into a product state of up or down states """
    Lambda_mat = np.zeros((N+1,chi))
    Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
    if up:  #initialize each spin in up state
        i=0 
    else:   #initialize each spin in down state
        i=d-1
    Lambda_mat[:,0] = 1
    Gamma_mat[:,i,0,0] = 1
    
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum(d**arr, chi)
    locsize[np.where(locsize==0)] = chi
    return Gamma_mat, Lambda_mat, locsize

def initialize_LU_RD(N, d, chi, scale_factor):
    """ Initializes the MPS linearly from up at the leftmost site to down at the rightmost site """
    """ scale_factor is a variable that defines the peak up/down values taken """
    Lambda_mat = np.zeros((N+1,chi))
    Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
    
    Lambda_mat[:,0] = 1
    
    temp = 1-np.arange(N)/(N-1)
    temp = temp-0.5
    
    
    Gamma_mat[:,0,0,0] = np.sqrt(temp*scale_factor + 0.5)
    Gamma_mat[:,d-1,0,0] = np.sqrt(-1*temp*scale_factor + 0.5)
    
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum(d**arr, chi)
    locsize[np.where(locsize==0)] = chi
    return Gamma_mat, Lambda_mat, locsize



def create_maxmixed_normstate(N, d, newchi):
    """ Creates vectorised density matrix of an unnormalized maximally mixed state, used to calculate the trace of a vectorised density matrix """
    """ since to obtain rho11 + rho22 you must take inner [1 0 0 1] [rho11 rho12 rho21 rho22]^T without a factor 1/sqrt(2) in front """
    lambdas = np.zeros((N+1,newchi))
    lambdas[:,0]= 1
    
    gammas = np.zeros((N,d**2,newchi,newchi), dtype=complex)
    diagonal = (1+d)*np.arange(d)
    gammas[:,diagonal, 0, 0] = 1        #/2  #/np.sqrt(2)
    
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,newchi**2)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum((d**2)**arr, newchi**2)
    
    NORM_state = MPS(0, N, d**2, newchi, True)
    NORM_state.Gamma_mat = gammas
    NORM_state.Lambda_mat = lambdas
    NORM_state.locsize = locsize
    return NORM_state

    
def calculate_thetas_singlesite(state):
    """ contracts lambda_i gamma_i lambda_i+1 (:= theta) for each site and returns them, used for the NORM_state """
    """ NOTE: only works for NORM_state since there the result is the same for all sites! """
    """ This function is used to prevent redundant calculation of these matrices """
    #Note, the lambda matrices are just a factor 1, it is possible to simply return a reshaped gamma matrix
    #temp = np.tensordot(np.diag(state.Lambda_mat[0,:]), state.Gamma_mat[0,:,:,:], axes=(1,1)) #(chi, d, chi)
    #return np.tensordot(temp, np.diag(state.Lambda_mat[1,:]),axes=(2,0)) #(chi,d,chi)
    return state.Gamma_mat[0].transpose(0,2,1)

def calculate_thetas_twosite(state):
    """ contracts lambda_i gamma_i lambda_i+1 gamma_i+1 lambda_i+2 (:= theta) for each site and returns them, used for the NORM_state """
    """ NOTE: only works for NORM_state since there the result is the same for all sites! """
    """ This function is used to prevent redundant calculation of these matrices """
    temp = np.tensordot(np.diag(state.Lambda_mat[0,:]), state.Gamma_mat[0,:,:,:], axes=(1,1)) #(chi, d, chi)
    temp = np.tensordot(temp,np.diag(state.Lambda_mat[1,:]),axes=(2,0)) #(chi, d, chi) 
    temp = np.tensordot(temp, state.Gamma_mat[1,:,:,:],axes=(2,1)) #(chi, d, chi, d) -> (chi,d,d,chi)
    return np.tensordot(temp,np.diag(state.Lambda_mat[2,:]), axes=(3,0)) #(chi, d, d, chi)










