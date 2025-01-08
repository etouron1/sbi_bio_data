import os
import sys
import math
import IMP
import IMP.core
import IMP.atom
import IMP.display
import IMP.algebra
import numpy as np
import time


# Number of letters  ACTG per chromosoms: 
chr_seq_parasite = {"chr01": 640851, "chr02": 947102, "chr03": 1067971, "chr04": 1200490,
           "chr05": 1343557, "chr06": 1418242, "chr07": 1445207, "chr08": 1472805,
           "chr09": 1541735, "chr10": 1687656, "chr11": 2038340,
           "chr12": 2271494, "chr13": 2925236, "chr14": 3291936}

# Centromers middle position: single centromere per chromosoms
chr_cen_parasite = {"chr01": 459191, "chr02": 448856, "chr03": 599144, "chr04": 643179,
           "chr05": 456668, "chr06": 479886, "chr07": 810620, "chr08": 300205,
           "chr09": 1243266, "chr10": 936752, "chr11": 833107,
           "chr12": 1283731, "chr13": 1169395, "chr14": 1073139}

sep_parasite = 3200

chr_seq_yeast_mean = {'chr01': 240000, 'chr02': 820000, 'chr03': 320000, 'chr04': 1540000, 
                 'chr05': 580000, 'chr06': 280000, 'chr07': 1100000, 'chr08': 570000, 
                 'chr09': 440000, 'chr10': 750000, 'chr11': 670000, 'chr12': 1080000, 
                 'chr13': 930000, 'chr14': 790000, 'chr15': 1100000, 'chr16': 950000}

chr_seq_yeast_duan = {"chr01": 230209, "chr02": 813179, "chr03": 316619, "chr04": 1531918, 
           "chr05": 576869, "chr06": 270148, "chr07": 1090947, "chr08": 562644,
            "chr09": 439885, "chr10": 745746, "chr11": 666455, "chr12": 1078176,
            "chr13": 924430, "chr14": 784334, "chr15": 1091290, "chr16": 948063}

chr_cen_yeast_duan = {'chr01': 151584, 'chr02': 238325, 'chr03': 114499, 'chr04': 449819, 
                 'chr05': 152103, 'chr06': 148622, 'chr07': 497042, 'chr08': 105698, 
                 'chr09': 355742, 'chr10': 436418, 'chr11': 439889, 'chr12': 150946, 
                 'chr13': 268149, 'chr14': 628877, 'chr15': 326703, 'chr16': 556070}

chr_seq_yeast_tjong = {"chr01": 230208, "chr02": 813136, "chr03": 316613, "chr04": 1531914,
           "chr05": 576869, "chr06": 270148, "chr07": 1090944, "chr08": 562639,
           "chr09": 439885, "chr10": 745446, "chr11": 666445, "chr12": 1078173,
           "chr13": 924430, "chr14": 784328, "chr15": 1091285, "chr16": 948060}

chr_cen_yeast_tjong = {"chr01": 151526, "chr02": 238226, "chr03": 114437, "chr04": 449762,
           "chr05": 152036, "chr06": 148563, "chr07": 496980, "chr08": 105638,
           "chr09": 355684, "chr10": 436060, "chr11": 439831, "chr12": 150886,
           "chr13": 268086, "chr14": 628812, "chr15": 326643, "chr16": 556010}



chr_seq_yeast = chr_seq_yeast_duan
chr_cen_yeast = chr_cen_yeast_duan


#sep_yeast = 10000

sep_yeast = 3200

def get_num_beads_and_start(chr_seq, sep):

    """ return a dict {chr : nb of beads}, a dict {chr : number of the start bead}, the total number of beads for all chr"""

    chr_bead = {}    # number of beads for each chromosome
    nbead = 0		# total number of beads for all chromosomes
    bead_start = {}  # bead label starts of a chr

    for i in chr_seq.keys(): # attention sorted()
        n = int(chr_seq[i] // sep) + int(chr_seq[i] % sep!=0)
        chr_bead[i] = n # number of beads for chromosome i
        
        nbead = nbead + n # total number of beads for all chromosmes
        bead_start[i] = nbead - n # the start bead for chr i
    return chr_bead, bead_start, nbead



def mdstep(model, constraints, xyzr, t, step):
    """ 
    do a molecular dynamics optimization step of a model on a set of contraints 
    modify in place the 3D coor of the DNA
    
    Input:
    - t : temperature, 
    - step : max iterations
    
    """

    sf = IMP.core.RestraintsScoringFunction(constraints) # Create a scoring function on a list of restraints. 
    o = IMP.atom.MolecularDynamics(model) # molecular dynamics optimizer
    md = IMP.atom.VelocityScalingOptimizerState(model, xyzr, t) # Maintains temperature during molecular dynamics by velocity scaling (to consume less energy).
    #This OptimizerState, when used with the MolecularDynamics optimizer, implements a simple thermostat by periodically rescaling the velocities
    o.set_scoring_function(sf) # add the scoring function to the optimizer
    o.add_optimizer_state(md)
    #each time the config of the DNA change, update the state
    s = o.optimize(step) # do the optimization for max step = step iterations, returns the final score of the loss
    #modif in place of the DNA configuration
    o.remove_optimizer_state(md)
    return s


def cgstep(model, constraints, step=1000):
    """ 
    do a conjugate gradient optimization step of a model on a set of constraints 
    modify in place the 3D coor of the DNA
    
    Input:
    - step: max iterations
    
    """
    
    sf = IMP.core.RestraintsScoringFunction(constraints) # Create a scoring function on a list of constraints.
    o = IMP.core.ConjugateGradients(model) # conjugate gradients optimizer
    o.set_scoring_function(sf) # add the scoring function to the optimizer
    f = o.optimize(step) # do the optimization for max step = step iterations, returns the final score of the loss
    return f


class VRSM_Simulator:

    def __init__(self,args):

        self.args = args
        if args.type=='p':
            self.chr_seq = chr_seq_parasite
            self.chr_cen = chr_cen_parasite
        if args.type=='y':
            self.chr_seq = chr_seq_yeast
            self.chr_cen = chr_cen_yeast
        
        self.telo_thick = args.telo_thick # 50 nm envelope thickness for telomere (nl)
        self.nuclear_rad = args.nuclear_rad # 1000 nm nuclear radius for trophozoite stage (nm)
        self.centro_rad = args.centro_rad # 300 nm centromere radius sphere (nm)
        self.nucleolus_center_x = args.nucleolus_center_x # (1200,0,0) nucleolus center
        self.nucleolus_rad = args.nucleolus_rad # 1600 nm nucleolus radius (nm)
        self.sep = args.sep # 3200 pb separation between 2 consecutive beads (pb)
        self.chr_bead, self.bead_start, self.nbead = get_num_beads_and_start(self.chr_seq,self.sep) #dict chr : nb bead, dict chr : num start bead, total nb beads
        self.r = args.radius # 15 nm radius of each bead (nm)
        self.lb = args.lb  # length of bond (nm) 30 : two consecutive beads at a distance of 30 nm
        self.kbend = args.kbend # 0.2 strength for angular restraint (kcal/mol)
        
        IMP.set_log_level(IMP.SILENT)
        self.m = IMP.Model() #create a model to store a collection of "particles" or data container

        self.build_IMP()


    def build_IMP(self):

        self.xyzr = IMP.core.create_xyzr_particles(self.m, self.nbead, self.r) # return a list of nbead coordinates x,y,z and a radius for each bead r
        
        corner1 = IMP.algebra.Vector3D(-self.nuclear_rad, 
                                       -self.nuclear_rad, 
                                       -self.nuclear_rad)
        corner2 = IMP.algebra.Vector3D(self.nuclear_rad, 
                                       self.nuclear_rad, 
                                       self.nuclear_rad)
        box = IMP.algebra.BoundingBox3D(corner1, corner2) # create a 3D box of length 2 nuclear radius
        
        # from random import random
        # rdummy=int(random()*10000)
        # for i in range(rdummy):
        #     ranvec = IMP.algebra.get_random_vector_in(box)

        #-------------------------- initialization --------------------------------------#
        for i in range(self.nbead):

            p0 = self.xyzr[i] #3D coordinate for bead i
            IMP.atom.Mass.setup_particle(p0, 1) 
            p = IMP.core.XYZR(p0) #decorator to set the coor
            coor = IMP.algebra.get_random_vector_in(box) #get a 3D random coor in the box radius with uniform density
            p.set_coordinates(coor) # bead i takes this coor
            p.set_coordinates_are_optimized(True)
        #-------------------------------------------------------------------------------#
        
        self.rdnaStart = self.bead_start['chr12'] + 140 #begin rDNA
        self.rdnaEnd = self.bead_start['chr12'] + 147 #not included as rDNA
        self.rdna1 = self.rdnaStart + 2 #last bead 1st chain
        self.rdna2 = self.rdnaStart + 3 #begin 2nd chain



    def build_first_constraints(self):
        """
        Create the first set of constraints : 
        - chromatin chain restraint: br
        - chromatin chain excluded volume: evr
        - NE (nuclear envelope): rcell
        - centromere localization: rcentro
        - nucleolus excluded volume: rnucleolus

        """

        chain = IMP.container.ListSingletonContainer(self.m, self.xyzr) #store a list of ParticleIndexes

        #------------- chromatin chain restraint u_ch ----------------------------------#
        #u_ch = 1/2*k*(|r_i-r_{i+1}|-d)^2 , d = 30 nm  
        # Create bonds for consecutive beads in a string
        bonds = IMP.container.ListSingletonContainer(self.m) #store a list of ParticleIndexes
        for id in self.chr_seq.keys():#for each chr Attention : sorted()
            istart = self.bead_start[id] #num of bead start in chr id
            iend = istart + self.chr_bead[id] #num of bead end in chr id
            bp = IMP.atom.Bonded.setup_particle(self.xyzr[istart]) #Bonded : decorator for a particle which has bonds, first particule p1
            for i in range(istart + 1, iend): #for each bead in the chr
                if i != self.rdna2:
                    bpr = IMP.atom.Bonded.setup_particle(self.xyzr[i]) # second Bonded particule p2
                    b = IMP.atom.create_custom_bond(bp, bpr, self.lb, 2) # connect the two wrapped particles by a custom bond, (p1, p2, length, stiffness)
                    # two particule are at 30 nm one from another
                    bonds.add(b.get_particle())
                    bp = bpr # step forward in chr
                else:
                    IMP.atom.Bonded.setup_particle(self.xyzr[self.rdna2])

        # Restraint for bonds
        bss = IMP.atom.BondSingletonScore(IMP.core.Harmonic(0, 1)) # Store a set of objects, Harmonic function (symmetric about the mean), 
        #simple score modeling an harmonic oscillator. The score is 0.5 * k * x * x, where k is the 'force constant' and x is the distance from the mean. 
        #mean : 0, force constant : k = 1
        br = IMP.container.SingletonsRestraint(bss, bonds) #apply the score for each bead
        # Applies a SingletonScore to each Singleton in a list.
        # This restraint stores the used particle indexes in a ParticleIndexes. 
        # The container used can be set so that the list can be shared with other containers (or a nonbonded list can be used).
        #--------------------------------------------------------------------------------#

        #------------- chromatin chain excluded volume restraint u_exc ----------------------------------#
        #u_exc = 1/2*k*(|r_i-r_j|-d)^2 if |r_i-r_j|<d, d=30 nm
        # Set up excluded volume
        evr = IMP.core.ExcludedVolumeRestraint(chain) #Prevent a set of particles and rigid bodies from inter-penetrating
        #------------------------------------------------------------------------------------------------#

        #------------- NE (nuclear envelope) restraint u_nuc ----------------------------------#
        #u_nuc = 1/2*k*(|r_i-nuc_center| - nuc_radius)^2 if |r_i-cent_radius| > nuc_radius, 0 else
        # Set up cap
        self.center = IMP.algebra.Vector3D(0, 0, 0) # the nucleus is centered at (0,0,0), r_nucleus = 850 nm
        ubcell = IMP.core.HarmonicUpperBound(self.nuclear_rad, 1.0) # Upper bound harmonic function (non-zero when feature > mean)
        # mean / reference distance : nuclear radius , k : spring constant 1.0
        sscell = IMP.core.DistanceToSingletonScore(ubcell, self.center) #Use an IMP::UnaryFunction to score a distance to a point. 
        # distance between r_i and the reference point : nuclear center

        rcell = IMP.container.SingletonsRestraint(sscell, chain) #compute the score for each bead
        # Applies a SingletonScore to each Singleton in a list. 
        #--------------------------------------------------------------------------------------#


        #------------- Centromere localization restraint u_cen ----------------------------------#

        # u_cen = 1/2*k*(|r_i-centro_center|-centro_radius)^2 if |r_i-centro_center| > centro_radius, 0 else
        # the centromere are in a sphere centered at (700,0,0), r_centro = 300 nm
        centro = IMP.algebra.Vector3D(self.nuclear_rad - self.centro_rad, 0, 0) #center of centromere localization : (800,0,0)
        
        listcentro = IMP.container.ListSingletonContainer(self.m) #create a list of 3D coor for centromeres (Store a list of ParticleIndexes) 
        for k in self.chr_seq.keys(): #for each chr Attention : sorted()
            
            centro_id = self.bead_start[k] + self.chr_cen[k]//self.sep # bead id in chr k for the centromere position
            listcentro.add(self.xyzr[centro_id]) # get the 3D coor for bead j
       
        ubcen = IMP.core.HarmonicUpperBound(self.centro_rad, 1.0) # Upper bound harmonic function
        # mean / reference distance : centromere sphere radius , k : spring constant 1.0
        sscen = IMP.core.DistanceToSingletonScore(ubcen, centro) # Use an IMP::UnaryFunction to score a distance to a point.
        # distance between r_i and the reference point : centromere sphere center

        rcentro = IMP.container.SingletonsRestraint(sscen, listcentro) #compute the score for each centromere bead
        #Applies a SingletonScore to each Singleton in a list. 
        # this restraint stores the used particle indexes in a ParticleIndexes. The container used can be set so that the list can be shared with other containers (or a nonbonded list can be used).
        #----------------------------------------------------------------------------------------#

        #------------- Nucleolus excluded volume restraint u_onu ----------------------------------#

        # # u_onu = 1/2*k*(|r_i-centro_nucleolus|-nucleolus_radius)^2 if |r_i-centro_nucleolus| > nucleolus_radius, 0 else
        # # the nucleolus is a sphere centered at (1200,0,0), r_nucleolus = 1600 nm
        # nucleolus_center = IMP.algebra.Vector3D(self.nucleolus_center_x, 0, 0)

        # ubnucleolus = IMP.core.HarmonicUpperBound(self.nucleolus_rad, 1.0) # Upper bound harmonic function
        # # mean / reference distance : nucleolus radius , k : spring constant 1.0
        # ssnucleolus = IMP.core.DistanceToSingletonScore(ubnucleolus, nucleolus_center) # Use an IMP::UnaryFunction to score a distance to a point.
        # # distance between r_i and the reference point : nucleolus center

        # rnucleolus = IMP.container.SingletonsRestraint(ssnucleolus, chain) #compute the score for each bead
        # #Applies a SingletonScore to each Singleton in a list. 
        # # this restraint stores the used particle indexes in a ParticleIndexes. The container used can be set so that the list can be shared with other containers (or a nonbonded list can be used).
        #----------------------------------------------------------------------------------------#

        # evr : excluded volume restraint, br : consecutive bead restraint, rcell : nuclear sphere restraint, rcentro : centromere sphere restraint, rnucleolus : nucleolus exclusion

        self.constraints = [evr, br, rcell, rcentro]#, rnucleolus]
 

    def simulate_before_telomere_constraint(self):
        """ do the optimization on the set of constraints (consecutive beads restaint, volume exclusion, nuclear sphere, centromere sphere)"""
        
        mdstep(self.m, self.constraints, self.xyzr, 1000000, 500)
        mdstep(self.m, self.constraints, self.xyzr, 500000, 500)
        mdstep(self.m, self.constraints, self.xyzr, 300000, 500)
        mdstep(self.m, self.constraints, self.xyzr, 100000, 500)
        mdstep(self.m, self.constraints, self.xyzr, 5000, 500)

        score = cgstep(self.m, self.constraints, 1000)
        #score of mdstep > score of cgstep
        print('conj gradient score before telomere restraint: ', score)


    def build_telomere_constraint(self):
        """
        Create the telomere localization constraint: rt 
        """

        #------------- Telomere localization restraint u_tel ----------------------------------#
        # Telomeres near nuclear envelope thickness 50
        # u_tel = 1/2*k*(|r_i-nuc_cent|-env_radius)^2 if |r_i-nuc_cent| < env_radius, 0 else
        telo = IMP.container.ListSingletonContainer(self.m) # list of the telomere 3D coor (Store a list of ParticleIndexes)
        for k in self.chr_seq.keys(): # for each chr Attention : sorted()
            telo_1_id = self.bead_start[k] # start bead telomere for chr k
            telo.add(self.xyzr[telo_1_id]) # get the 3D coor + radius of the bead (15 nm)

            telo_2_id = telo_1_id + self.chr_bead[k]- 1 # end bead telomere for chr k
            telo.add(self.xyzr[telo_2_id]) # # get the 3D coor + radius of the bead (15 nm)

        envelope = self.nuclear_rad - self.telo_thick #compute the envelope radius (thickness: 50 nm, envelope radius: 850-50 = 800 nm)

        tlb = IMP.core.HarmonicLowerBound(envelope, 1.0) #Lower bound harmonic function (non-zero when feature < mean)
        # mean / reference distance : envelope sphere radius , k : spring constant 1.0
        sst = IMP.core.DistanceToSingletonScore(tlb, self.center) # Use an IMP::UnaryFunction to score a distance to a point.
        # distance between r_i and the reference point : nuclear sphere center

        rt = IMP.container.SingletonsRestraint(sst, telo) #compute the score for each telomere bead
        #Applies a SingletonScore to each Singleton in a list. 
        #-------------------------------------------------------------------------------------#

        #rt : telomere restraint
        self.constraints = self.constraints + [rt]
    
    def build_nucleolus_constraint(self):

        #------------- Nucleolus excluded volume restraint u_onu ----------------------------------#

        # # u_onu = 1/2*k*(|r_i-centro_nucleolus|-nucleolus_radius)^2 if |r_i-centro_nucleolus| > nucleolus_radius, 0 else
        # # the nucleolus is a sphere centered at (1200,0,0), r_nucleolus = 1600 nm
        
        not_rDNA = IMP.container.ListSingletonContainer(self.m)
        for i in range(self.nbead):
            if i < self.rdnaStart or i >= self.rdnaEnd:
                p = self.xyzr[i]
                not_rDNA.add(p)
        # rDNA chr12 near nucleolus
        rDNA = IMP.container.ListSingletonContainer(self.m) #rDNA is just the bead start and the bead start and bead start + 1
        rDNA.add(self.xyzr[self.rdna1])
        rDNA.add(self.xyzr[self.rdna2])

        nucleolus_center = IMP.algebra.Vector3D(self.nucleolus_center_x, 0, 0)

        in_nucleolus = IMP.core.HarmonicLowerBound(self.nucleolus_rad, 0.5)  #can also try lowerbound nucleolus_rad-rncutoff
        # Lower bound harmonic function : 1/2*k*(x-nuc_rad)^2 if x-nuc_rad < 0
        # # mean / reference distance : nucleolus radius , k : spring constant 0.5
        ss_nucleolus = IMP.core.DistanceToSingletonScore(in_nucleolus, nucleolus_center) 
        #apply the harmonic fct in_nucleolus = 1/2*k*(x-nuc_rad)^2 to x the distance between a point and the nucleolus center
        # Use an IMP::UnaryFunction to score a distance to a point.
        # # distance between r_i and the reference point : nucleolus center
        rnucleolus = IMP.container.SingletonsRestraint(ss_nucleolus,rDNA) #compute the score for each bead of the rDNA
        #m.add_restraint(rnucleolus) #7
        self.constraints = self.constraints + [rnucleolus]

        # Set up Nucleolis #
        out_nucleolus = IMP.core.HarmonicUpperBound(self.nucleolus_rad, 0.5) 
        # Upper bound harmonic function 1/2*k*(x-nuc_rad)^2 if x-nuc_rad > 0
        ss_nucleolus_1 = IMP.core.DistanceToSingletonScore(out_nucleolus, nucleolus_center)
        #apply the harmonic fct in_nucleolus = 1/2*k*(x-nuc_rad)^2 to x the distance between a point and the nucleolus center
        rnucleolus_1 = IMP.container.SingletonsRestraint(ss_nucleolus_1,not_rDNA) #compute the score for each bead not in the rDNA
        #m.add_restraint(rnucleolus_1) #3
        self.constraints = self.constraints + [rnucleolus_1]
        #----------------------------------------------------------------------------------------#


    def simulate_before_angle_constraint(self):
        """ do the optimization on the set of constraints:

        - consecutive beads restaint: br
        - volume exclusion: evr
        - nuclear sphere: rcell
        - centromere sphere: rcentro
        - nucleolus constraint: rnucleolus
        - telomere restraint: rt
        
        """
        
        mdstep(self.m, self.constraints, self.xyzr, 500000, 5000)
        mdstep(self.m, self.constraints, self.xyzr, 300000, 5000)
        mdstep(self.m, self.constraints, self.xyzr, 5000, 10000)
        score = cgstep(self.m, self.constraints, 500)
        print('conj gradient score before angle restraint', score)
    

    def build_angle_constraint(self):
        """
        Create the stiffness constraint: ars 
        """

        #------------- Chromatin persistence length u_angle ----------------------------------#
        # Angle Restraint
        angle = math.pi
        angle_set = [] #set 
        noangle = [i for i in self.bead_start.values()]  # do not apply angle restraints on the start bead of each chr
        ars = [] #angle restraints set for all the involved beads
        for i in range(self.nbead - 1):
            ieval = i + 1
            if ieval in noangle:
                continue
            elif i in noangle:
                continue
            else:

                pot = IMP.core.Harmonic(angle, self.kbend) #Harmonic function (symmetric about the mean) (mean, spring constant)
                #1/2*k_angle(|r_i-\mu|-d)^2
                ar = IMP.core.AngleRestraint(self.m, pot, self.xyzr[i - 1], self.xyzr[i], self.xyzr[i + 1]) #get the 3D coord of the 3 consecutive beads 
                # Angle restraint between three particles. (model, score function, p1,p2,p3)
                ars.append(ar)
                angle_set.append(ar)
        #-----------------------------------------------------------------------------------#

        constraints = self.constraints + ars
        return constraints

    def simulate_final(self, constraints):
        """ do the optimization on the set of constraints:
        - consecutive beads restaint: br
        - volume exclusion: evr
        - nuclear sphere: rcell
        - centromere sphere: rcentro 
        - nucleolus constraint: rnucleolus
        - telomere restraint: rt
        - angle restraint
        
        do a last cgstep optimization without the angle constraint
        """

        mdstep(self.m, constraints, self.xyzr, 50000,500)
        mdstep(self.m, constraints, self.xyzr, 25000,500)
        mdstep(self.m, constraints, self.xyzr, 20000,1000)
        mdstep(self.m, constraints, self.xyzr, 10000,1000)
        mdstep(self.m, constraints, self.xyzr, 5000,3000)
        mdstep(self.m, constraints, self.xyzr, 2000,5000)
        mdstep(self.m, constraints, self.xyzr, 1000,7000)
        mdstep(self.m, constraints, self.xyzr, 500,10000)
        score = cgstep(self.m, constraints, 2500)
        print('conj gradient score with angle:', score)
        # mdstep(self.m, constraints, self.xyzr, 500000, 5000)
        # mdstep(self.m, constraints, self.xyzr, 300000, 5000)
        # mdstep(self.m, constraints, self.xyzr, 5000, 10000)
        # score = cgstep(self.m, constraints, 500)

        # score = cgstep(self.m, self.constraints, 1000) # do not optimize on angle
        score = cgstep(self.m, self.constraints, 1000) # do not optimize on angle

        print('Final conj gradient score:', score)	

	
    def get_X(self):
        """
        Return the DNA configuration

        Return:
        - a list [[chr_id,b_x, b_y, b_z], for b in list of beads]
        """
        X = [] # the 3D config
        
        for i in range(self.nbead):
            X.append([self.xyzr[i].get_x(), self.xyzr[i].get_y(), self.xyzr[i].get_z()]) # config 3D with num_chr, 3D coor. the indice i is the bead's number 
        return X



    def simulate_DNA(self):
        """
        Build the constraints and optimize by gradually adding the constraints

        Return:
        - the DNA configuration
        """
        self.build_first_constraints()
        self.simulate_before_telomere_constraint()
        self.build_telomere_constraint()
        self.build_nucleolus_constraint()
        self.simulate_before_angle_constraint()
        constraints = self.build_angle_constraint()
        self.simulate_final(constraints)
        return self.get_X()
  
