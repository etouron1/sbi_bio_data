import time
import mlxp
import time
import os

from utils import assign_device, get_dtype, instantiate

import pprint

pp = pprint.PrettyPrinter(indent=4)

from contact_matrix import get_contact_matrix_all_chr

class Simulation:
    """
    """
    def __init__(self,config,logger):
        """
        Initializes the Simulation class with the provided configuration and logger.

        Parameters:
        - config : all the args of the simulator.
        - logger : the DNA 3D config and the raw contact matrix
        """

        self.logger = logger
        self.args = config
        self.device = assign_device(self.args.system.device)
        self.dtype = get_dtype(self.args.system.dtype)
        self.round = 1
        


    def simulate_C_1_round(self):
        """
        Compute the raw contact matrix from 1 DNA configuration

        Return:
        - the DNA configuration
        - the raw contact_matrix
        """
        
        self.simulator = instantiate(self.args.simulator.name)(self.args.simulator.args)#init the simulator
        start_time = time.time()
        X = self.simulator.simulate_DNA() # DNA_config : list of 3D coor for each bead [[b_x, b_y, b_z], b in B]
        end_time = time.time()
        print("total time: "+ str((end_time-start_time)))
        C = get_contact_matrix_all_chr('y', self.args.simulator.args.sep, [X]) #compute the raw contact matrix from X
        return X, C
            
    def simulate_C_n_rounds(self):
        """
        Compute the raw contact matrix from n DNA configuration

        Output:
        - the last DNA configuration in the pickle file 'DNA_config'
        - the raw contact matrix in the pickle file 'raw_contact_matrix'
        """
        try :
            start_round = self.logger.load_artifacts(artifact_name='next_start_round', artifact_type='pickle')
            print(f"Load checkpoint, starting from round {start_round}")
        except:
            start_round=1
            print(f"Failed to load checkpoint, starting from round {start_round}")
        
        for i in range(start_round,self.args.simulator.args.n_simu+1):
            print(f"round {i}/{self.args.simulator.args.n_simu}")
            if i==1:
                X, C = self.simulate_C_1_round()
            else:
                C = self.logger.load_artifacts(artifact_name='tmp_contact_matrix', artifact_type='pickle')
                X, C_tmp = self.simulate_C_1_round()
                C += C_tmp
            self.logger.log_artifacts(C, artifact_name='tmp_contact_matrix', artifact_type='pickle')
            self.logger.log_artifacts(i+1, artifact_name='next_start_round', artifact_type='pickle')

        self.logger.log_artifacts(X, artifact_name='DNA_config', artifact_type='pickle')
        self.logger.log_artifacts(C, artifact_name='raw_contact_matrix', artifact_type='pickle')

        print("Removing temporary files (contact matrix and start id)")
        os.remove(f"data/{self.logger.log_id}/artifacts/pickle/next_start_round")
        os.remove(f"data/{self.logger.log_id}/artifacts/pickle/tmp_contact_matrix")
        print("Successfull simulation")




        
@mlxp.launch(config_path='configs')#, seeding_function=set_seed)
def simulate_C(ctx: mlxp.Context)->None:
    simulation = Simulation(ctx.config, ctx.logger)
    if ctx.config['simulator']['args']['type']=='y':
            to_print = f"Simulation of yeast " 
    elif ctx.config['simulator']['args']['type']=='p':
            to_print = f"Simulation of parasite "
    else:
        print("unknown type")
        return
    print(to_print + f"at a resolution of {ctx.config['simulator']['args']['sep']} pb for {ctx.config['simulator']['args']['n_simu']} rounds")
    simulation.simulate_C_n_rounds()


if __name__ == "__main__":
    for i in range(1):
        simulate_C()
    
