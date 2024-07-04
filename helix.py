import numpy as np
import matplotlib.pyplot as plt

def create_helix_coordinates(L, R, pitch, dPhi, N_per_cycle, is_righthanded):
    """ Creates an equidistantly spaced helix using the provided parameters """
    if dPhi==None:  #If dPhi is not specified, we use N_per_cycle
        dPhi = 2*np.pi/N_per_cycle
    else:
        N_per_cycle = 2*np.pi/dPhi
    
    coordinates = np.zeros((3,L))
    
    angles = np.arange(L)*dPhi
    dZ = pitch/N_per_cycle
    
    if is_righthanded==False:
        angles *= -1
    
    coordinates[0] = R*np.cos(angles)
    coordinates[1] = R*np.sin(angles)
    coordinates[2] = dZ*np.arange(L)
    coordinates = np.around(coordinates, decimals=10)
    return coordinates.transpose()

def create_v_vectors(lattice):
    """ Calculates the v-vectors as defined in section 1.3.2 of the report """
    #Creating d_m+1 vectors
    d_m1 = lattice[1:] - lattice[:-1]  
    d_m1 = d_m1 / np.linalg.norm(d_m1, axis=1)[:, np.newaxis]
    
    #Creating d_m+2 vectors
    d_m2 = lattice[2:] - lattice[:-2]
    d_m2 = d_m2 / np.linalg.norm(d_m2, axis=1)[:, np.newaxis]
    
    #Calculating v_m vectors
    return np.cross(d_m1[:-1], d_m2)













