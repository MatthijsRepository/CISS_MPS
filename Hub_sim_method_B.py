import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

import pickle
import time

import traceback


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




       
def TEBD_Hub_multi(State, incl_SOC, TimeOp_Coul, TimeOp_hopping, TimeOp_SOC, diss_index, diss_TimeOp, normalize, diss_bool):
    """ Performing the TEBD steps in parallel using python's 'pool' method """
    
    #Coulomb interactions
    new_matrices = p.starmap(global_apply_twosite, [(TimeOp_Coul, normalize, State.Lambda_mat[i:i+3], State.Gamma_mat[i:i+2], State.locsize[i:i+3], State.d, State.chi) for i in range(0, State.N-1, 2)])
    for i in range(0, State.N-1, 2):
        State.Lambda_mat[i+1] = new_matrices[int(i//2)][0]
        State.Gamma_mat[i:i+2] = new_matrices[int(i//2)][1] 
    
    #Hopping interactions
    for i in range(3):
            for j in range(i, State.N-2, 3):
                State.apply_threesite(TimeOp_hopping, j, normalize)
    
    if incl_SOC:
        for i in range(np.shape(TimeOp_SOC)[0]): #Application of a sixsite operator
                theta = State.contract(2*i, 2*i+5) #(chi, chi, d, ..., d)
                theta = np.tensordot(theta, TimeOp_SOC[i], axes=([2,3,4,5,6,7],[6,7,8,9,10,11]))
                State.decompose_contraction(theta, 2*i, normalize)
            
    if diss_bool:
        for i in range(len(diss_index)):
            if len(diss_TimeOp[i][0])==State.d:
                State.apply_singlesite(diss_TimeOp[i], diss_index[i])
            else:
                State.apply_twosite(diss_TimeOp[i], diss_index[i], normalize)
    pass 


def calc_currents_bond(State, site, trace, normalize): 
    """ calculates the current crossing the Hubbard bond located to the left of 'site'  """
    """ also calculates the currents leaving the given site, for comparison in case of the Hubbard model """
    in_up = np.real( State.expval_three_six(hop_cur_op, site-2, NORM_state, normalize, True) ) / trace
    in_down = np.real( State.expval_three_six(hop_cur_op, site-1, NORM_state, normalize, True) ) / trace
    out_up = np.real( State.expval_three_six(hop_cur_op, site, NORM_state, normalize, True) ) / trace
    out_down = np.real( State.expval_three_six(hop_cur_op, site+1, NORM_state, normalize, True) ) / trace
    
    if incl_SOC:
        if site<=2:
            SOC_cur_1=False
        else:
            SOC_cur_1 = State.expval_three_six(SOC_cur_op[int(site/2)-2], site-4, NORM_state, normalize, False) / trace
        if site>=State.N-2:
            SOC_cur_1=False
        else:   
            SOC_cur_2 = State.expval_three_six(SOC_cur_op[int(site/2)-1], site-2, NORM_state, normalize, False) / trace
    else:
        SOC_cur_1 = None
        SOC_cur_2 = None
    return in_up, in_down, out_up, out_down, SOC_cur_1, SOC_cur_2



def calc_current_profile(State, trace, normalize): 
    """ calculates the current through each bond in the Hubbard chain """
    up_currents = np.array([])
    down_currents = np.array([])
    SOC_currents = np.array([])
    
    for i in range(0,State.N-2,2):
        up_currents = np.append(up_currents, np.real( State.expval_three_six(hop_cur_op, i, NORM_state, normalize, True)  / trace ))
        down_currents = np.append(down_currents, np.real( State.expval_three_six(hop_cur_op, i+1, NORM_state, normalize, True) / trace )) 
    
    current_profile = up_currents + down_currents
    
    if incl_SOC:
        for i in range(0,State.N-4,2):
            SOC_currents = np.append(SOC_currents, np.real(State.expval_three_six(SOC_cur_op[int(i/2)], i, NORM_state, normalize, False) ) / trace)
            current_profile[int(i/2):int(i/2)+2] += SOC_currents[int(i/2)]
    
    #SOC_cur_1 = np.real(State.expval_three_six(SOC_cur_op[0], 0, NORM_state, normalize, False) ) / trace
    #SOC_cur_2 = np.real(State.expval_three_six(SOC_cur_op[1], 2, NORM_state, normalize, False) ) / trace
    #current_profile[0:2] += SOC_cur_1
    #current_profile[1:3] += SOC_cur_2
    
    #avg_cur = np.average(current_profile)
    #abs_diff = max(current_profile) - min(current_profile)
    #rel_diff = abs_diff / avg_cur
    #print(cur_profiles)
    #print(f"Average current:        {np.round(avg_cur, decimals=6)}")
    #print(f"Max difference: abs:    {np.round(abs_diff, decimals=6)}, rel: {np.round(rel_diff, decimals=6)}")
    return up_currents, down_currents, current_profile


##############################################################################################

def Op_Sz_Op(Op1, Op2, num_Sz):
    Op1 = np.kron(Op1, np.eye(d))
    for i in range(num_Sz):
        Op1 = np.kron(Op1, np.kron(-1*Sz,np.eye(d)))
    return np.kron(Op1, np.kron(Op2, np.eye(d)))

def init_current_operators(v_list, incl_SOC):
    """ Initialize current operators as defined in section 2.2 of the report """
    global hop_cur_op, SOC_cur_op
    
    hop_cur_op = -t_hopping * 1j* ( -Op_Sz_Op(Sp, Sm, 1) +  Op_Sz_Op(Sm, Sp, 1))
    hop_cur_op = np.reshape(hop_cur_op, (d**2,d**2,d**2, d**2,d**2,d**2))

    if incl_SOC:
        SOC_cur_op = np.zeros((2, (d**2)**6, (d**2)**6), dtype=complex)
        for i in range(2):  #Last two entries of v_list are used
            temp = s_SOC * 1j * (v_list[-2+i,2] * 1j * -1*Op_Sz_Op(Sp, Sm, 3) + np.conj(v_list[-2+i,2] * 1j) * Op_Sz_Op(Sm, Sp, 3))
            SOC_cur_op[i] += np.kron(temp, np.eye(d**2))
            
            temp = s_SOC * 1j * ((v_list[-2+i,0] - 1j*v_list[-2+i,1]) * 1j * -1*Op_Sz_Op(Sp, Sm, 4) + np.conj((v_list[-2+i,0] - 1j*v_list[-2+i,1]) * 1j) * Op_Sz_Op(Sm, Sp, 4))
            SOC_cur_op[i] += temp
            
            temp = s_SOC * 1j * ((v_list[-2+i,0] + 1j*v_list[-2+i,1]) * 1j * -1*Op_Sz_Op(Sp, Sm, 2) + np.conj((v_list[-2+i,0] + 1j*v_list[-2+i,1]) * 1j) * Op_Sz_Op(Sm, Sp, 2))
            SOC_cur_op[i] += np.kron(np.kron(np.eye(d**2), temp), np.eye(d**2))
            
            temp = s_SOC * 1j * (-1*v_list[-2+i,2] * 1j * -1*Op_Sz_Op(Sp, Sm, 3) +  np.conj(-1*v_list[-2+i,2] * 1j) * Op_Sz_Op(Sm, Sp, 3))
            SOC_cur_op[i] += np.kron(np.eye(d**2), temp)
        SOC_cur_op = np.reshape(SOC_cur_op, (2, d**2,d**2,d**2,d**2,d**2,d**2, d**2,d**2,d**2,d**2,d**2,d**2))
    pass


def init_TimeOp():
    from MPS_TimeOp import Time_Operator
    """ Initialize time operator for the XXZ chain """
    TimeEvol_obj = Time_Operator(N, d, diss_bool, True)
    
    #Coulomb terms
    TimeEvol_obj.Ham_Coul = U_coulomb * TimeEvol_obj.calc_dens_Ham_term(num_op, num_op, True)
    TimeEvol_obj.TimeOp_Coul = TimeEvol_obj.Create_TimeOp(TimeEvol_obj.Ham_Coul, dt, use_CN)
    
    #Hopping terms
    TimeEvol_obj.Ham_hop = np.zeros((d**6, d**6), dtype=complex)
    TimeEvol_obj.Ham_hop += -t_hopping/2 * TimeEvol_obj.calc_dens_Ham_term_threesite(Sx, -1*Sz, Sx)
    TimeEvol_obj.Ham_hop += -t_hopping/2 * TimeEvol_obj.calc_dens_Ham_term_threesite(Sy, -1*Sz, Sy)
    TimeEvol_obj.TimeOp_hop = TimeEvol_obj.Create_TimeOp(TimeEvol_obj.Ham_hop, dt, use_CN)
    
    #SOC terms
    if incl_SOC:
        from helix import create_helix_coordinates, create_v_vectors
        helix = create_helix_coordinates(L, R, pitch, dPhi, N_per_cycle, is_righthanded)
        v_list = create_v_vectors(helix)
        TimeEvol_obj.Ham_SOC = TimeEvol_obj.calc_sixsite_SOC(v_list, s_SOC)
        TimeEvol_obj.TimeOp_SOC = np.zeros((len(v_list[:,0]), d**2, d**2, d**2, d**2, d**2, d**2, \
                                 d**2, d**2, d**2, d**2, d**2, d**2), dtype=complex)
        for i in range(len(v_list[:,0])):
            Op = TimeEvol_obj.Create_TimeOp(TimeEvol_obj.Ham_SOC[i], dt, use_CN)
            Op = np.reshape(Op, (d**2, d**2, d**2, d**2, d**2, d**2, \
                                 d**2, d**2, d**2, d**2, d**2, d**2))
            TimeEvol_obj.TimeOp_SOC[i] = Op
    else: 
        v_list=None
        TimeEvol_obj.TimeOp_SOC = None
    
    #Dissipative terms
    
    #Equivalent, but specific for each site which operators act on it
    #TimeEvol_obj.add_dissipative_term_singlesite(0, np.array([np.sqrt(up_factor)*mu_plus*Sp, np.sqrt(up_factor)*mu_min*Sm]), dt, use_CN)
    #TimeEvol_obj.add_dissipative_term_twosite(0, np.array([np.sqrt(down_factor)*mu_plus*Sp, np.sqrt(down_factor)*mu_min*Sm]), dt, use_CN, [True,True], [True,True])
    #TimeEvol_obj.add_dissipative_term_twosite(N-2, np.array([mu_min*Sp, mu_plus*Sm]), dt, use_CN, [True,True], [False,False])
    #TimeEvol_obj.add_dissipative_term_singlesite(N-1, np.array([mu_min*Sp, mu_plus*Sm]), dt, use_CN)
    
    TimeEvol_obj.add_dissipative_term_twosite(0, np.array([np.sqrt(up_factor)*mu_plus*Sp, np.sqrt(up_factor)*mu_min*Sm, np.sqrt(down_factor)*mu_plus*Sp, np.sqrt(down_factor)*mu_min*Sm]), \
                                              dt, use_CN, [False,False, True,True], [False,False,True,True])
    TimeEvol_obj.add_dissipative_term_twosite(N-2, np.array([mu_min*Sp, mu_plus*Sm, mu_min*Sp, mu_plus*Sm]), \
                                              dt, use_CN, [True,True, False,False], [False,False, True,True])
    return TimeEvol_obj, v_list


def time_evolution(TimeEvol_obj, State, steps, track_Sz):
    """ Perform the time evolution steps and calculate the observables """
    if TimeEvol_obj.is_density != State.is_density:
        print("Error: time evolution operator type does not match state type (MPS/DENS)")
        return
    print(f"Starting time evolution of {State}")
    
    if track_VNEE:
        State.VNEE = np.zeros((N-1, steps))
    
    if track_n:
        State.n_expvals = np.zeros((State.N, steps))
    
    t1 = time.time() #Time evolution start time
    for t in range(steps):
        if (t%20==0 and t>0):
            print(str(t) + " / " + str(steps) + " (" + str(np.round(t/steps*100, decimals=0)) + "% completed), approx. " + str(np.round((steps/t - 1)*(time.time()-t1), decimals=0)) + "s left" )
            
        State.normalization = np.append(State.normalization, State.calculate_vidal_inner(State))
        if State.is_density:
            State.trace = np.append(State.trace, State.calculate_vidal_inner(NORM_state))
        
        if track_VNEE:
            Schmidt_vals = np.multiply(State.Lambda_mat[1:State.N], State.Lambda_mat[1:State.N])
            mask = Schmidt_vals > 1e-14
            result = np.zeros(np.shape(Schmidt_vals))
            result[mask] = -1 * Schmidt_vals[mask] * np.log(Schmidt_vals[mask])
            State.VNEE[:,t] = np.sum(result, axis=1)
            
        if track_n:
            State.n_expvals[:,t], temp_trace = State.expval_chain(np.kron(num_op, np.eye(2)), NORM_state)
            State.n_expvals[:,t] *= 1/temp_trace
        
        if track_currents:
            in_up, in_down, out_up, out_down, SOC_cur_1, SOC_cur_2 = calc_currents_bond(State, current_site_index, State.trace[-1], normalize)
            State.current_left_up = np.append(State.current_left_up, in_up)
            State.current_left_down = np.append(State.current_left_down, in_down)
            State.current_right_up = np.append(State.current_right_up, out_up)
            State.current_right_down = np.append(State.current_right_down, out_down)
            if incl_SOC:
                State.SOC_current_1 = np.append(State.SOC_current_1, SOC_cur_1)
                State.SOC_current_2 = np.append(State.SOC_current_2, SOC_cur_2)
        
        #Perform timestep
        #State.TEBD_Hub_method_B(incl_SOC, TimeEvol_obj.TimeOp_Coul, TimeEvol_obj.TimeOp_hop, TimeEvol_obj.TimeOp_SOC, TimeEvol_obj.diss_index, TimeEvol_obj.diss_TimeOp, normalize, diss_bool)
        TEBD_Hub_multi(State, incl_SOC, TimeEvol_obj.TimeOp_Coul, TimeEvol_obj.TimeOp_hop, TimeEvol_obj.TimeOp_SOC, TimeEvol_obj.diss_index, TimeEvol_obj.diss_TimeOp, normalize, diss_bool)
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
    
    if hasattr(State, 'VNEE'):
        plt.figure(dpi=200)
        for i in range(int(N/2)-1):
            plt.plot(np.linspace(0, steps*dt, steps), State.VNEE[i], label=f"$S_{i+1}$ ($S_{State.N-i-1}$)")
        plt.plot(np.linspace(0, steps*dt, steps), State.VNEE[int(N/2)-1], label=f"$S_{int(N/2)}$")

        handles, labels = plt.gca().get_legend_handles_labels()
        handles.reverse()
        labels.reverse()
        plt.legend(handles, labels)
        
        plt.xlabel("Time")
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.ylabel("VNEE")
        #plt.grid()
        plt.show()
        
    if hasattr(State, 'n_expvals'):
        for i in range(State.N):
            plt.plot(State.n_expvals[i], label=f"Site {i}")
        plt.xlabel("Timesteps")
        plt.ylabel("<n>")
        plt.grid()
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.show()
        
    if plot_n_final:
        if hasattr(State, 'n_expvals'):
            occup = State.n_expvals[:,-1]
        else:
            occup, temp_trace = State.expval_chain(np.kron(num_op, np.eye(d)), NORM_state)
            occup *= 1/temp_trace
        even = np.arange(0, State.N, 2)
        odd = np.arange(1, State.N, 2)
        plt.plot(occup[even], linestyle="", marker=".", label="Even sites")
        plt.plot(occup[odd], linestyle="", marker=".", label="Odd sites")
        plt.xlabel("Physical sites")
        plt.ylabel("<n>")
        plt.legend()
        plt.grid()
        plt.show()

    
    
    up_currents, down_currents, current_profile = calc_current_profile(State, State.trace[-1], normalize)
    #plt.plot(np.arange(1,len(up_currents)+1), up_currents, linestyle="", marker="^", markersize=3)
    #plt.plot(np.arange(1,len(down_currents)+1), down_currents, linestyle="", marker="v", markersize=3)
    plt.plot(np.arange(1,len(current_profile)+1), current_profile, linestyle="", marker=".")
    plt.grid()
    plt.xlabel("Site")
    plt.xticks(np.arange(1,len(current_profile)+1))
    plt.ylabel("Current")
    plt.show()
    
    if track_currents:
        plt.plot(State.current_left_up, label="In")
        plt.plot(State.current_right_up, label="Out")
        #plt.plot( (State.current_left_up + State.current_right_up)/2, label="Average current")
        if incl_SOC:
            plt.plot(State.current_SOC_1, label="1")
            plt.plot(State.current_SOC_2, label="2")
        plt.title("Current through 'up' channel")
        plt.xlabel("Timesteps")
        plt.ylabel("Current")
        plt.legend()
        plt.grid()
        plt.show()
        
        
        plt.plot(State.current_left_down, label="In")
        plt.plot(State.current_right_down, label="Out")
        #plt.plot( (State.current_left_down + State.current_right_down)/2, label="Average current")
        if incl_SOC:
            plt.plot(State.current_SOC_1, label="1")
            plt.plot(State.current_SOC_2, label="2")
        plt.title("Current through 'down' channel")
        plt.xlabel("Timesteps")
        plt.ylabel("Current")
        plt.legend()
        plt.grid()
        plt.show()
    pass
    



##############################################################################################

max_cores = 4

t0 = time.time()
#### Simulation variables
N =         8
d =         2
chi =       16      #MPS truncation parameter
newchi =    65   #DENS truncation parameter

#im_steps =  0
#im_dt =     -0.03j
steps =     800
dt =        0.02


#current_site_index = int(np.ceil(N/4)*2-2)  #Site of which we will track the currents in and out of
                                            #Corresponds to N/2-2 if L is even, N/2-1 if L is odd

normalize =         True  #maintain normalization, must be set to true
use_CN =            False #choose if you want to use Crank-Nicolson approximation
diss_bool =         True  #whether to include dissipation
incl_SOC =          False  #whether to include SOC

track_currents =    True  #track the currents through and over a given site, specified by current site
current_site_index =4      #index of the Hubbard site through which to check the current

track_VNEE =        False #Whether to track VNEE
track_n =           False #Whether to track site occupation over all timesteps
plot_n_final =      True  #Whether to plot the occupation after time evolution is completed, can still be used if track_n is False


#### Helix constants
L =                 int(N/2)
R =                 1
pitch =             2
dPhi =              None
N_per_cycle =       L
is_righthanded =    True


#### Hamiltonian and coupling constants
t_hopping =         1   #NOTE: t must be incorporated for spin current operator
U_coulomb =         1
s_SOC =             1

s_coup =            1
mu =                0.2
polarization_is_up= True



if incl_SOC:
    polarization_factor = 5
else:
    polarization_factor = 1
    
    
mu_plus = np.sqrt(s_coup*(1+mu))
mu_min = np.sqrt(s_coup*(1-mu))

if polarization_is_up:
    up_factor = 1
    down_factor = 1 / polarization_factor
else:
    up_factor = 1 / polarization_factor
    down_factor = 1
    



#### Spin matrices
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
Sx = np.array([[0,1], [1,0]])
Sy = np.array([[0,-1j], [1j,0]])
Sz = np.array([[1,0],[0,-1]])

#### JW-transformed number operator
num_op = np.array([[1,0],[0,0]])

#### Swap operator, used such that swaps can be performed using the global_apply_twosite function
swap_op = np.zeros((d**4,d**4))
for i in range(d**2):
    for j in range(d**2):
        swap_op[i*d**2 + j, j*d**2 +i] = 1


#### Spin current operators for DENS objects
#spin_current_op = -t_hopping*  1/2 * ( np.kron( np.kron(Sx, np.eye(d)) , np.kron(Sy, np.eye(d))) - np.kron( np.kron(Sy, np.eye(d)) , np.kron(Sx, np.eye(d))) )
#spin_current_op = -t_hopping * 1j* ( np.kron( np.kron(Sp, np.eye(d)) , np.kron(Sm, np.eye(d))) - np.kron( np.kron(Sm, np.eye(d)) , np.kron(Sp, np.eye(d))) )

#spin_current_op = -t_hopping * 1j* ( np.kron( np.kron(Sp, np.eye(d)) ,np.kron(np.kron(-1*Sz, np.eye(d)) , np.kron(Sm, np.eye(d)))) - np.kron( np.kron(Sm, np.eye(d)) ,np.kron(np.kron(-1*Sz, np.eye(d)) , np.kron(Sp, np.eye(d)))) )
hop_cur_op = None
SOC_cur_op = None

#### NORM_state definition, is initialized in main() function due to multiprocessing reasons
NORM_state = None





#### Loading and saving states
save_state_bool = False
load_state_bool = False

loadstate_folder = "data\\"
loadstate_filename = ""

if incl_SOC:
    savestate_folder = "data\\"
    savestring = "mu" + str(int(mu*10)) + "_uf" + str(int(up_factor*10)) + "_s" + str(int(s_SOC*10)) + "_"
else:
    savestate_folder = "data\\"
    savestring = "mu" + str(int(mu*10)) + "_"


##############################################################################################

def main():
    #Import is done here instead of at the beginning of the code to avoid multiprocessing-related errors
    from MPS_TimeOp import MPS
    from MPS_initializations import initialize_halfstate, initialize_LU_RD
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
    
    #creating time evolution object
    TimeEvol_obj1, v_list = init_TimeOp()
    
    init_current_operators(v_list, incl_SOC)
    
    time_evolution(TimeEvol_obj1, DENS1, steps, track_n)
    
    
    try:
        plot_results(DENS1)
    except:
        pass
    print(f"Left current: {np.round(DENS1.current_left_up[-1] + DENS1.current_left_down[-1], decimals=6)}")
    print(f"Right current: {np.round(DENS1.current_right_up[-1] + DENS1.current_right_down[-1], decimals=6)}")
    if incl_SOC:
        print(f"SOC from site: {np.round(DENS1.SOC_current_1[-1],decimals=6)}")
        print(f"SOC over: {np.round(DENS1.SOC_current_2[-1],decimals=6)}")
    
    if save_state_bool:
        DENS1.store(savestate_folder, savestring, False)
    pass
    

t0 = time.time()

if __name__=="__main__":
    p = Pool(processes=max_cores)
    try:
        main()
    except:
        print(traceback.format_exc())
    p.close()

del hop_cur_op, SOC_cur_op

elapsed_time = time.time()-t0
print()
print(f"Elapsed simulation time: {elapsed_time}")







