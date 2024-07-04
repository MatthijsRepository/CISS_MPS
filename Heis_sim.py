import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

import pickle
import time

### These imports are done in the 'main()' function to avoid multiprocessing-related errors

#from MPS_TimeOp_no_locsize import MPS, Time_Operator
#from MPS_TimeOp import MPS, Time_Operator

#from MPS_initializations import *


##############################################################################################

def load_state(folder, name, new_ID):
    from MPS_TimeOp import MPS
    """ loads a pickled state from folder 'folder' with name 'name' - note: name must include .pkl """
    filename = folder + name
    with open(filename, 'rb') as file:  
        loaded_state = pickle.load(file)
    globals()[loaded_state.name] = loaded_state
    
    loaded_state.ID = new_ID
    if loaded_state.is_density:
        loaded_state.name = "DENS"+str(new_ID)
    else: 
        loaded_state.name = "MPS"+str(new_ID)
    return loaded_state
    
def create_superket(State, newchi):
    from MPS_TimeOp import MPS
    """ create MPS of the density matrix of a given MPS """
    gammas, lambdas, locsize = State.construct_vidal_supermatrices(newchi)
    
    name = "DENS" + str(State.ID)
    newDENS = MPS(State.ID, State.N, State.d**2, newchi, True)
    newDENS.Gamma_mat = gammas
    newDENS.Lambda_mat = lambdas
    newDENS.locsize = locsize
    globals()[name] = newDENS
    return newDENS

##############################################################################################

def global_apply_twosite(TimeOp, normalize, Lambda_mat, Gamma_mat, locsize, d, chi):
        """ Applies a two-site operator to sites i and i+1 """
        #theta = self.contract(i,i+1) #(chi, chi, d, d)
        theta = np.tensordot(np.diag(Lambda_mat[0,:locsize[0]]), Gamma_mat[0,:,:locsize[0],:locsize[1]], axes=(1,1)) #(chi, d, chi)
        theta = np.tensordot(theta,np.diag(Lambda_mat[1,:locsize[1]]),axes=(2,0)) #(chi, d, chi) 
        theta = np.tensordot(theta, Gamma_mat[1,:,:locsize[1],:locsize[2]],axes=(2,1)) #(chi,d,d,chi)
        theta = np.tensordot(theta,np.diag(Lambda_mat[2,:locsize[2]]), axes=(3,0)) #(chi, d, d, chi)   
        #operator is applied, tensor is reshaped
        TimeOp = np.reshape(TimeOp, (d,d,d,d))
        theta_prime = np.tensordot(theta, TimeOp,axes=([1,2],[2,3])) #(chi,chi,d,d)     
        theta_prime = np.reshape(np.transpose(theta_prime, (2,0,3,1)),(d*locsize[0], d*locsize[2])) #first to (d, chi, d, chi), then (d*chi, d*chi)
        X, Y, Z = np.linalg.svd(theta_prime); Z = Z.T

        if normalize:
            Lambda_mat[1,:locsize[1]] = Y[:locsize[1]] * 1/np.linalg.norm(Y[:locsize[1]])
        else:
            Lambda_mat[1,:locsize[1]] = Y[:locsize[1]]
        
        #truncation, and multiplication with the inverse lambda matrix of site i, where care is taken to avoid divides by 0
        X = np.reshape(X[:d*locsize[0], :locsize[1]], (d, locsize[0], locsize[1])) 
        inv_lambdas  = Lambda_mat[0, :locsize[0]].copy()
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        tmp_gamma = np.tensordot(np.diag(inv_lambdas),X[:,:locsize[0],:locsize[1]],axes=(1,1)) #(chi, d, chi)
        Gamma_mat[0,:,:locsize[0],:locsize[1]] = np.transpose(tmp_gamma,(1,0,2))
        
        #truncation, and multiplication with the inverse lambda matrix of site i+2, where care is taken to avoid divides by 0
        Z = np.reshape(Z[:d*locsize[2], :locsize[1]], (d, locsize[2], locsize[1]))
        Z = np.transpose(Z,(0,2,1))
        inv_lambdas = Lambda_mat[2, :locsize[2]].copy()
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        Gamma_mat[1,:,:locsize[1],:locsize[2]] = np.tensordot(Z[:,:locsize[1],:locsize[2]], np.diag(inv_lambdas), axes=(2,0)) #(d, chi, chi)
        return (Lambda_mat[1], Gamma_mat)
        
def TEBD_Heis_multi(State, TimeOp, diss_index, diss_TimeOp, normalize, diss_bool):
    """ Performing the TEBD steps in parallel using python's 'pool' method """
    for j in [0,1]:
        new_matrices = p.starmap(global_apply_twosite, [(TimeOp, normalize, State.Lambda_mat[i:i+3], State.Gamma_mat[i:i+2], State.locsize[i:i+3], State.d, State.chi) for i in range(j, State.N-1, 2)])
        for i in range(j, State.N-1, 2):
            State.Lambda_mat[i+1] = new_matrices[int(i//2)][0]
            State.Gamma_mat[i:i+2] = new_matrices[int(i//2)][1]
        
    if diss_bool:
        for i in range(len(diss_index)):
            State.apply_singlesite(diss_TimeOp[i], diss_index[i])
    pass 

##############################################################################################

def init_TimeOp():
    from MPS_TimeOp import Time_Operator
    """ Initialize time operator for the XXZ chain """
    TimeEvol_obj = Time_Operator(N, d, diss_bool, True)
    
    TimeEvol_obj.Ham_XXZ = np.zeros((d**4,d**4), dtype=complex)

    TimeEvol_obj.Ham_XXZ += JXY * TimeEvol_obj.calc_dens_Ham_term(Sx, Sx, True)
    TimeEvol_obj.Ham_XXZ += JXY * TimeEvol_obj.calc_dens_Ham_term(Sy, Sy, True)
    TimeEvol_obj.Ham_XXZ += JZ * TimeEvol_obj.calc_dens_Ham_term(Sz, Sz, True)
    
    TimeEvol_obj.TimeOp_XXZ = TimeEvol_obj.Create_TimeOp(TimeEvol_obj.Ham_XXZ, dt, use_CN)
    
    #Note, we must pass an imaginary time here, because the create_TimeOp function multiplies by -1j
    TimeEvol_obj.add_dissipative_term_singlesite(0, np.sqrt(s_coup)*Sp, dt, use_CN)
    TimeEvol_obj.add_dissipative_term_singlesite(N-1, np.sqrt(s_coup)*Sm, dt, use_CN)
    return TimeEvol_obj


def time_evolution(TimeEvol_obj, State, steps, track_Sz):
    """ Perform the time evolution steps and calculate the observables """
    if TimeEvol_obj.is_density != State.is_density:
        print("Error: time evolution operator type does not match state type (MPS/DENS)")
        return
    print(f"Starting time evolution of {State}")
    
    middle_site = int(np.round(State.N/2-1))
    
    if track_Sz:
        State.Sz_expvals = np.zeros((State.N, steps))
    
    t1 = time.time()
    for t in range(steps):
        #if (t%2000==0 and t>0 and save_state_bool):
        #    State.store()
        if (t%20==0 and t>0):
            #print(str(t) + " / " + str(steps) + " (" + str(np.round(t/steps*100, decimals=0)) + "% completed)")
            print(str(t) + " / " + str(steps) + " (" + str(np.round(t/steps*100, decimals=0)) + "% completed), approx. " + str(np.round((steps/t - 1)*(time.time()-t1), decimals=0)) + "s left" )
        
            
        State.normalization = np.append(State.normalization, State.calculate_vidal_inner(State))
        if State.is_density:
            State.trace = np.append(State.trace, State.calculate_vidal_inner(NORM_state))
            
        if track_Sz:
            State.Sz_expvals[:,t], temp_trace = State.expval_chain(np.kron(Sz, np.eye(2)), NORM_state)
            State.Sz_expvals[:,t] *= 1/temp_trace
        if track_currents:
            State.current_left_up = np.append(State.current_left_up, np.real( State.expval_twosite(spin_current_op, middle_site, NORM_state, normalize) ))
            State.current_right_up = np.append(State.current_right_up, np.real( State.expval_twosite(spin_current_op, middle_site+1, NORM_state, normalize) ))
            if State.is_density:
                State.current_left_up[-1] *= 1/State.trace[t]
                State.current_right_up[-1] *= 1/State.trace[t]
        
        #State.TEBD_Heis(TimeEvol_obj.TimeOp_XXZ, TimeEvol_obj.diss_index, TimeEvol_obj.diss_TimeOp, normalize, diss_bool)
        TEBD_Heis_multi(State, TimeEvol_obj.TimeOp_XXZ, TimeEvol_obj.diss_index, TimeEvol_obj.diss_TimeOp, normalize, diss_bool)
    pass


def plot_results(State):
    """ Plot the time evolution results """
    plt.plot(State.Lambda_mat[int(State.N/2)], linestyle="", marker=".")
    plt.title(f"Singular values of site {int(State.N/2)}")
    plt.grid()
    plt.show()
    
    plt.plot(State.normalization)
    plt.xlabel("Timesteps")
    plt.ylabel("Normalization")
    plt.show()
    
    if State.is_density:
        plt.plot(State.trace)
        plt.xlabel("Timesteps")
        plt.ylabel("Trace")
        plt.show()
        
    if hasattr(State, 'Sz_expvals'):
        for i in range(State.N):
            plt.plot(State.Sz_expvals[i], label=f"Site {i}")
        plt.xlabel("Timesteps")
        plt.ylabel("<Sz>")
        plt.grid()
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.show()
    if plot_Sz_final:
        Sz_vals, temp_trace = State.expval_chain(np.kron(Sz, np.eye(2)), NORM_state)
        Sz_vals *= 1/temp_trace
        plt.plot(Sz_vals, marker=".", linestyle="")
        plt.xlabel("site")
        plt.ylabel("<Sz>")
        plt.grid()
        plt.show()
        
        #plt.plot(State.Sz_expvals[:,-1], linestyle="", label=".")
        #plt.show()
    
    if track_currents:
        plt.plot(State.current_left_up, label="In")
        plt.plot(State.current_right_up, label="Out")
        plt.plot( (State.current_left_up + State.current_right_up)/2, label="Average current")
        plt.xlabel("Timesteps")
        plt.ylabel("Current")
        plt.legend()
        plt.grid()
        plt.show()
    pass
    



##############################################################################################

max_cores = 5

t0 = time.time()
#### Simulation variables
N=14
d=2
chi=10      #MPS truncation parameter
newchi=60   #DENS truncation parameter

im_steps = 0
im_dt = -0.03j
steps=1000
dt = 0.02

normalize = True
use_CN = False #choose if you want to use Crank-Nicolson approximation
diss_bool = True
track_Sz = False
track_currents = True
plot_Sz_final = True

#### Hamiltonian and Lindblad constants
h=0
JXY=1#1
JZ=0.5

s_coup=1

#### Spin matrices
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
Sx = np.array([[0,1], [1,0]])
Sy = np.array([[0,-1j], [1j,0]])
Sz = np.array([[1,0],[0,-1]])

#### Spin current operators for DENS objects
#spin_current_op = 1/2 * ( np.kron( np.kron(Sx, np.eye(d)) , np.kron(Sy, np.eye(d))) - np.kron( np.kron(Sy, np.eye(d)) , np.kron(Sx, np.eye(d))) )
spin_current_op = 1j* ( np.kron( np.kron(Sp, np.eye(d)) , np.kron(Sm, np.eye(d))) - np.kron( np.kron(Sm, np.eye(d)) , np.kron(Sp, np.eye(d))) )

#### NORM_state initialization
NORM_state = None


#### Loading and saving states
loadstate_folder = "data\\"
loadstate_filename = ""

savestate_folder = "data\\"

save_state_bool = False
load_state_bool = False

##############################################################################################

def main():
    #Import is done here instead of at the beginning of the code to avoid multiprocessing-related errors
    from MPS_initializations import initialize_halfstate, initialize_LU_RD
    from MPS_TimeOp import MPS
    from MPS_initializations import create_maxmixed_normstate, calculate_thetas_singlesite, calculate_thetas_twosite
    
    global NORM_state
    NORM_state = create_maxmixed_normstate(N, d, newchi)
    NORM_state.singlesite_thetas = calculate_thetas_singlesite(NORM_state)
    NORM_state.twosite_thetas = calculate_thetas_twosite(NORM_state)

        
    #load state or create a new one
    if load_state_bool:
        DENS1 = load_state(loadstate_folder, loadstate_filename, 1)
    else:
        MPS1 = MPS(1, N,d,chi, False)
        #MPS1.Gamma_mat[:,:,:,:], MPS1.Lambda_mat[:,:], MPS1.locsize[:] = initialize_halfstate(N,d,chi)
        MPS1.Gamma_mat[:,:,:,:], MPS1.Lambda_mat[:,:], MPS1.locsize[:] = initialize_LU_RD(N,d,chi, scale_factor = 0.8 )
        #temp = np.zeros((d,chi,chi))
        #temp[0,0,0] = np.sqrt(4/5)
        #temp[1,0,0] = 1/np.sqrt(5)
        #MPS1.Gamma_mat[0] = temp
        
        DENS1 = create_superket(MPS1, newchi)
    #DENS1.locsize = np.ones(DENS1.N+1, dtype=int) * newchi
    
    #creating time evolution object
    TimeEvol_obj1 = init_TimeOp()
    
    time_evolution(TimeEvol_obj1, DENS1, steps, track_Sz)
    
    plot_results(DENS1)
    
    if save_state_bool:
        DENS1.store(savestate_folder, "", True)
    pass
    

t0 = time.time()

if __name__=="__main__":
    p = Pool(processes=max_cores)
    main()
    p.close()

elapsed_time = time.time()-t0
print(f"Elapsed simulation time: {elapsed_time}")







