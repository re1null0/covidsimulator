import pandas as pd
import numpy as np
from numpy.random import RandomState, SeedSequence
from sklearn.neighbors import KDTree
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import random
import simulator

class Particles():
    def __init__(self, simulator):
        ''' 
        An object of class Particles is characterized by five parameters: 
            x (array): position coordinates on a 2D map for each particle in the array
            v (array): velocities of the particles in the array
            ages (array): an age distribution array of the particles
            time_cur_states (array): the total time present at the current state for each particle
            epidemic_state (array): the epidemic state of each particle in the array is encoded as:
            0: Susceptible, 1: Exposed, 2: Infected,
            3: Recovered Immunized, 4: Dead, 7: Severely Infected
        '''
        
        self.model = pd.read_csv('model.csv', index_col=0)
        
        # Randomly initializing the position coordinates. 
        # Offset by 0.5 to provide smooth distribution.
        #np.random.seed(simulator.SEED+0*simulator.number_of_iter)
        self.x = np.loadtxt('x.txt', dtype=np.float64)
        
        # Randomly initialization of the particle velocities.         
        # Offset by 0.5 to provide smooth distribution.
        #np.random.seed(simulator.SEED+1*simulator.number_of_iter)
        self.v =  np.loadtxt('v.txt', dtype=np.float64)
        
        # Initialize the age distribution based on the age groups and age pyramid defined in the class Simulator
        self.ages = np.transpose(self.model['ages'].to_numpy().reshape(1, -1))
        # np.random.seed(simulator.SEED+2*simulator.number_of_iter) dont change
        # np.random.shuffle(self.ages)
        
        # Initialize the time array for all particles
        self.time_cur_state = np.transpose(self.model['time'].to_numpy().reshape(1, -1))
        
        # Initialize the epidemic state of the particles
        self.epidemic_state = np.transpose(self.model['state'].to_numpy().reshape(1, -1))
        
        # Randomly initilalize particles to be at the  exposed state to start the epidemic
        # np.random.seed(simulator.SEED+3*simulator.number_of_iter) dont change
        # np.put(self.epidemic_state,np.random.choice(range(simulator.NUMBER_OF_PARTICLES*1), simulator.INITIAL_EXPOSED, replace=False),1)
        
        self.vac = np.zeros((simulator.NUMBER_OF_PARTICLES, 1)) 
        self.vac_imm = np.zeros((simulator.NUMBER_OF_PARTICLES, 1)) 
        self.vac_groups = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
        # The dictionary and dataframe are used to store the information of each epidemic state at every iteration
        self.states= {}
        self.app = np.transpose(self.model['app'].to_numpy().reshape(1, -1))
        self.df = pd.DataFrame(columns=["ind", "days", "exposed", "infected", "qua", "iso", "severe infected", "recovered", "dead", "susceptible", "total cases"])
        self.test_res = np.transpose(self.model['testing'].to_numpy().reshape(1, -1))
        self.n_id = simulator.NUMBER_OF_PARTICLES
        self.n_time = 2
        with open('cell.npy', 'rb') as f:
            self.contact_cell = np.load(f, allow_pickle=True)
        #self.contact_cell = np.empty((self.n_id, self.n_time), dtype=list) dont change
        self.temp = np.zeros((simulator.NUMBER_OF_PARTICLES, 1))
    
    def update_states(self, i, simulator):
        '''
        The class method updates the dictionary of epidemic states 
        and stores these results with the provided iteration in the dataframe.
        Susceptible = 0; Exposed = 1; Infected = 2; Severe Infected = 7; Recovered = 3;
        Dead = 4.
        
        Parameters
        ----------
        i (int): An iteration number
        simulator (class object): An object of class Simulator that contains 
        simulation parameters (e.g. delta_t)

        Returns
        -------
        None.

        '''

        self.states = {'exposed': np.where(np.copy(self.epidemic_state)==1, self.epidemic_state, 0), 
                       'infected': np.where(np.copy(self.epidemic_state)==2, self.epidemic_state, 0),
                       'true_qua': np.where(np.copy(self.epidemic_state)==5, self.epidemic_state, 0),
                       'false_qua': np.where(np.copy(self.epidemic_state)==9, self.epidemic_state, 0),
                       'true_iso': np.where(np.copy(self.epidemic_state)==6, self.epidemic_state, 0),
                       'false_iso': np.where(np.copy(self.epidemic_state)==8, self.epidemic_state, 0),
                       'recovered': np.where(np.copy(self.epidemic_state)==3, self.epidemic_state, 0),
                       'dead': np.where(np.copy(self.epidemic_state)==4, self.epidemic_state, 0),
                       'severe_inf':np.where(np.copy(self.epidemic_state)==7, self.epidemic_state, 0),
                       'susceptible':(simulator.NUMBER_OF_PARTICLES - np.count_nonzero(np.copy(self.epidemic_state))),
                       'total_cases':((np.count_nonzero(self.epidemic_state)))
                           }
        

        self.df.loc[i,:] = [i, i*simulator.delta_t, np.count_nonzero(self.states['exposed']), np.count_nonzero(self.states['infected']),
                            (np.count_nonzero(self.states['true_qua'])+ np.count_nonzero(self.states['false_qua'])),
                            (np.count_nonzero(self.states['true_iso'])+np.count_nonzero(self.states['false_iso'])),
                 np.count_nonzero(self.states['severe_inf']), np.count_nonzero(self.states['recovered']),
                 np.count_nonzero(self.states['dead']), np.count_nonzero(self.states['susceptible']), self.states['total_cases']-np.count_nonzero(self.states['exposed'])]

    def plot(self, simulator,i):
        '''
        arr = np.arange(0,i - 1, 1)
        time_vec = arr * simulator.delta_t;
        plt.plot(time_vec, np.where(np.copy(self.epidemic_state)==1, self.epidemic_state, 0), 'y')
        plt.plot(time_vec, np.where(np.copy(self.epidemic_state)==2, self.epidemic_state, 0))
        plt.plot(time_vec, simulator.NUMBER_OF_PARTICLES)
        plt.plot(time_vec, simulator.NUMBER_OF_PARTICLES)
        plt.plot(time_vec, simulator.NUMBER_OF_PARTICLES)
        '''
        ax = plt.gca()
        #self.df.plot(x ='days', y='susceptible', kind = 'line', color = 'blue', ax=ax) 
        self.df.plot(x ='days', y='exposed', kind = 'line', color = 'yellow', ax=ax)
        self.df.plot(x ='days', y='infected', kind = 'line', color = 'orange', ax=ax)
        self.df.plot(x ='days', y='qua', kind = 'line', color = 'magenta', ax=ax)
        #self.df.plot(x ='days', y='false_qua', kind = 'line', color = 'blue', ax=ax)
        self.df.plot(x ='days', y='iso', kind = 'line', color = 'cyan', ax=ax)
        # self.df.plot(x ='days', y='false_iso', kind = 'line', color = 'magenta', ax=ax)
        self.df.plot(x ='days', y='severe infected', kind = 'line', color = 'brown', ax=ax)
        self.df.plot(x ='days', y='recovered', kind = 'line', color = 'green', ax=ax)
        self.df.plot(x ='days', y='dead', kind = 'line', color = 'red', ax=ax)
        self.df.plot(x ='days', y='total cases', kind = 'line', color = 'black', ax=ax) 
        plt.title("Epidemic states at {}/{} iterations".format(i, simulator.number_of_iter))
        plt.xlabel('Time (days)')
        plt.ylabel('Number of particles')
        plt.savefig('./plot/states_{}.png'.format(i))
        return plt.show()
        
        '''
        The plot function visualizes the epidemic dynamic curves for each state

        Parameters
        ----------
        simulator (class object): An object of class Simulator that contains 
        simulation parameters (e.g. NUMBER_OF_PARTICLES)

        Returns
        -------
        A plot figure
        '''

    def vac_per_iter(self, i, simulator):
        rand_num = np.random.uniform(0, 1, 1)
        if rand_num[0]>(simulator.VAC_FLOAT - simulator.VAC_FLOOR):
            vac_per_iter = simulator.VAC_FLOOR
            return vac_per_iter
        else:
            vac_per_iter = simulator.VAC_CEIL
            return vac_per_iter


    def update_velocities(self, i, simulator): # Particle class method
        '''
        The class method updates velocities of particles. If a velocity for a particle exceeds
        the magnitude of self.INIT_V_MAX, it set to zero. The velocity for dead and severely
        infected particles is also set zero.
        isolated
        
        Parameters
        -------
        i (int): An iteration number 
        simulator (class object): An object of class Simulator that contains simulation parameters 
        
        Returns
        -------
        None
         
        '''
        
        if i%simulator.KDT_FREQ==0 :#| i%simulator.KDT_FREQ==1:
            #np.random.seed(simulator.SEED+4*simulator.number_of_iter)
            self.v = self.v + simulator.speed_gain*(np.random.random((simulator.NUMBER_OF_PARTICLES, 2))-0.5)
            self.v = np.where((self.epidemic_state==4) | (self.epidemic_state==7) 
                              | (self.epidemic_state==5) | (self.epidemic_state==9) 
                              | (self.epidemic_state==6) | (self.epidemic_state==8), 0, self.v)
            self.v = np.where(self.v < -simulator.INIT_V_MAX, 0, self.v)
            self.v = np.where(self.v > simulator.INIT_V_MAX, 0, self.v)
    

    def update_coordinates(self, simulator):  
        self.x = self.x + self.v*simulator.delta_t
        self.x = np.where(self.x > 1, -1, self.x)
        self.x = np.where(self.x < -1, 1, self.x)

        '''
        DONE
        The function to update the coordinates of the particles at each iteration.
        The particles stay inside of the 2D boundaries, set to [-1,1] for both dimensions. 
        If a particle reaches one of the borders, it is sent to the opposite side. 
        For example, if x > 1, then update to x = -1.
        
        Parameters:
        simulator (class object): An object of class Simulator that contains simulation parameters 
         
        Returns
        -------
        None
        '''

        
    def get_new_cases_ids(self, i, simulator):
        '''
        This function randomly selects a proportion of indexes representing contagious 
        particles (exposed, infected, severe infected) based on their disease
        transmittion rates.
        
        Parameters
        -------
        i (int): a number of the current iteration
        simulator (class object): An object of class Simulator that contains simulation parameters 
         
        Returns
        -------
        new_cases (list): list of particles that spread infection
         
        '''
        exposed_id = np.array(np.nonzero(self.states['exposed'])[0])
        #np.random.seed(simulator.SEED+5*simulator.number_of_iter+i)
        ind_end_exp = np.random.randint(len(exposed_id), size= int(np.ceil(simulator.TRANSMISSION_RATE_EXPOSED*len(exposed_id))))
        #ind_end_exp = np.random.choice(len(exposed_id), int(np.ceil(simulator.TRANSMISSION_RATE_EXPOSED*len(exposed_id))), replace=False)
        
        qua_id = np.array(np.nonzero(self.states['true_qua'])[0])
       # np.random.seed(simulator.SEED+6*simulator.number_of_iter+i)
        #ind_end_qua = np.random.choice(len(qua_id), int(np.ceil(simulator.TRANSMISSION_RATE_QUA*len(qua_id))), replace=False)
        ind_end_qua = np.random.randint(len(qua_id), size= int(np.ceil(simulator.TRANSMISSION_RATE_QUA*len(qua_id))))
        
        infected_id = np.array(np.nonzero(self.states['infected'])[0])
    
        sev_infected_id = np.array(np.nonzero(self.states['severe_inf'])[0])
       # np.random.seed(simulator.SEED+7*simulator.number_of_iter+i)     
        ind_end_sev = np.random.randint(len(sev_infected_id), size= int(np.ceil(simulator.TRANSMISSION_RATE_SEVERE_INFECT*len(sev_infected_id))))
        #ind_end_sev = np.random.choice(len(sev_infected_id), int(np.floor(simulator.TRANSMISSION_RATE_SEVERE_INFECT*len(sev_infected_id))), replace=False)
        
        isolated_id = np.array(np.nonzero(self.states['true_iso'])[0])
       # np.random.seed(simulator.SEED+8*simulator.number_of_iter+i)
        ind_end_iso = np.random.randint(len(isolated_id), size= int(np.ceil(simulator.TRANSMISSION_RATE_QUA*len(isolated_id))))
        #ind_end_iso = np.random.choice(len(isolated_id), int(np.ceil(simulator.TRANSMISSION_RATE_QUA*len(isolated_id))), replace=False)
        
        array = np.hstack((exposed_id[ind_end_exp], qua_id[ind_end_qua], infected_id, sev_infected_id[ind_end_sev], isolated_id[ind_end_iso])).ravel()
        #print(array)
        contact_ids = [int(i) for sublist in KDTree(self.x, leaf_size=2, metric="manhattan").query_radius(self.x[array], 
                        r=simulator.init_cont_threshold)  for i in sublist]
 
        new_cases = [contact_ids[x] for x 
                              in (np.where(self.epidemic_state[contact_ids]==0))[0]]
        
        return new_cases

    def get_contact(self, i, simulator):
        exposed_id = np.array(np.nonzero(self.states['exposed'])[0])
        #np.random.seed(0+10*simulator.number_of_iter+i)
        ind_end_exp = np.random.randint(len(exposed_id), size= int(np.ceil(simulator.TRANSMISSION_RATE_EXPOSED*len(exposed_id))))
        #ind_end_exp = np.random.choice(len(exposed_id), int(np.ceil(simulator.TRANSMISSION_RATE_EXPOSED*len(exposed_id))), replace=False)

        infected_id = np.array(np.nonzero(self.states['infected'])[0])
    
        sev_infected_id = np.array(np.nonzero(self.states['severe_inf'])[0])
       # np.random.seed(0+11*simulator.number_of_iter+i)     
        ind_end_sev = np.random.randint(len(sev_infected_id), size= int(np.ceil(simulator.TRANSMISSION_RATE_SEVERE_INFECT*len(sev_infected_id))))
        #ind_end_sev = np.random.choice(len(sev_infected_id), int(np.floor(simulator.TRANSMISSION_RATE_SEVERE_INFECT*len(sev_infected_id))), replace=False)
        
        isolated_id = np.array(np.nonzero(self.states['true_iso'])[0])
       # np.random.seed(0+12*simulator.number_of_iter+i)
        ind_end_iso = np.random.randint(len(isolated_id), size= int(np.ceil(simulator.TRANSMISSION_RATE_QUA*len(isolated_id))))
        #ind_end_iso = np.random.choice(len(isolated_id), int(np.ceil(simulator.TRANSMISSION_RATE_QUA*len(isolated_id))), replace=False)
        
        array = np.hstack((exposed_id[ind_end_exp],infected_id, sev_infected_id[ind_end_sev], isolated_id[ind_end_iso])).ravel()
        #print(array)
        if len(array) > 1:
            contacts = [int(i) for sublist in KDTree(self.x, leaf_size=2, metric="manhattan").query_radius(self.x[array], r=simulator.init_cont_threshold)  for i in sublist]
            
            contact_ids = KDTree(self.x, leaf_size=2, metric="manhattan").query_radius(self.x[array], 
                            r=simulator.init_cont_threshold, count_only=False)
            
            for k in range(len(contact_ids)):
                if len(contact_ids[k])<=0:
                    continue
                curr_ind = array[k]
                if ((self.app[curr_ind]==1) & ((self.epidemic_state[curr_ind]==1) | (self.epidemic_state[curr_ind]==0)| (self.epidemic_state[curr_ind]==2)| (self.epidemic_state[curr_ind]==7))):
                    temp_cont = []
                    temp_time = []
                    for j in range (1, len(contact_ids[k])):
                        cont_ind = contact_ids[k][j]
                        if ((self.app[cont_ind]==1) & (((self.epidemic_state[cont_ind]==1)) | (self.epidemic_state[cont_ind]==0)| (self.epidemic_state[cont_ind]==2)| (self.epidemic_state[cont_ind]==7))):
                            
                            temp_cont.append(cont_ind)
                            temp_time.append(i*simulator.delta_t)
                        self.contact_cell[curr_ind, 0] = temp_cont
                        self.contact_cell[curr_ind, 1] = temp_time
            self.temp[contacts] = 1
            contact_ids = np.where((self.temp==1)&(self.epidemic_state==0))
    
            return contact_ids, self.contact_cell, contacts
    
    