from mdl_NAVIGATE_AZ import BVC_NAV as AZIMUTH
from mdl_NAVIGATE_Memory import BVC_NAV as MEM
from mdl_NAVIGATE_HP import BVC_NAV as HP
from mdl_NAVIGATE import BVC_NAV as NO_MEM
from mdl_NAVIGATE_COMBO import BVC_NAV as COMBO
from mdl_HPmax import BVC_NAV as HPmax
from mdl_NAVIGATE_HP_interceps import BVC_NAV as Default_inter
import tensorflow as tf
import numpy as np
import time
from TimerFunc import tic, toc, TicTocGenerator
TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

#save simulation parameters
save_path = "ModelComparison" #path name
OldSim=0   #change this if path already have n older simulations
sims2execute= 10 #total environments to create
sim_repeat = 1 #repetitions each simulation (to compare variables)
plt_flag=0  #set to 0 or 1 to create video of the session
vid_flag=plt_flag
sim_time = 45  # maximal allowed simulation time in sec
seed = 0

####  navigation parameters
telen_neurons = 10000  #total neurons available for navigation encoding
TT=[1000,2000,5000,10000,20000,50000] #optional : change population size per rep, require uncommenting a line in the run loop
RateScale=[1,1,1,1,1,1,1,1] #some simulations can use a variable version of this variable, here it's just a default value

####  environment properties

# n_corals=100 #### optional - high complexity task
n_corals=np.random.randint(20,25)  #hoe many corals will be in the environment
n_c=n_corals #keep an unrandomized version of this variable (for the specific simulations it is randomized in the loop)

arena_size=[4,4] #arena size in [m]. can be changed in the loop.
Goal_min_max_pos = [arena_size[0]/4, arena_size[1]/2*.9]  # feeding position range (will be chosen randomly)
coral_pos_range = [-arena_size[0]/2*.95,arena_size[1]/2*.95] # corals position range (will be chosen randomly)


# run on GPU
with tf.compat.v1.Session() as sess:
    for RUN in range (0,sims2execute):

        ####   define current environment
        corals_pos = np.random.uniform(coral_pos_range[0],coral_pos_range[1],size=[n_corals,2])
        Goal_rnd_pos = np.random.uniform(Goal_min_max_pos[0], Goal_min_max_pos[1], size=[1, 2])
        Goal_pos=[Goal_rnd_pos[0][0],Goal_rnd_pos[0][1]]
        #define food quadrant
        Goal_quadrant_sign=[[1,1],[1,-1]][np.random.randint(0,2)]#,[-1,1],[-1,-1]][np.random.randint(0,4)]
        Goal_pos=[Goal_pos[0] * Goal_quadrant_sign[0],Goal_pos[1] * Goal_quadrant_sign[1]]
        # and agent initial position accordingly
        agent_init_pos = [-1.85*Goal_quadrant_sign[0], -1.85*Goal_quadrant_sign[1]]


        for rep in range(0,sim_repeat):

            ######## optional simulation type #####:

        #### change population size ###
            # telen_neurons=TT[rep]
        #### change complexity ###
            # n_corals = (rep+1)*5
            # corals_pos = np.random.uniform(coral_pos_range[0], coral_pos_range[1], size=[n_corals, 2])
        ###### change arena size ###
            # n_c0=5 #initial conditions- number of corals is smallest arena
            # AS0=[2,2] #initial conditions- smallest arena size
            # st0=(45/2) #initial conditions- simulation duration smallest arena
            # arena_size = [AS0[0] + rep*2, AS0[1] + rep*2] #current arena size
            # arena_size_ratio = (arena_size[0] * arena_size[1]) / (AS0[0] * AS0[1])
            # n_corals= int(n_c0 * arena_size_ratio) #current number of corals, proportional to arena size change
            # sim_time = st0 * (rep+1) #current simulation max duration, increases linearly as the ideal path does
            # Goal_min_max_pos = [arena_size[0]/2-1,arena_size[1]/2*0.9]  # feeding position range (will be chosen randomly)
            # coral_pos_range = [-arena_size[0]/2 ,arena_size[1]/2] #[-arena_size[0]/2 +.2 ,arena_size[1]/2 -.2] #[-arena_size[0]/2*.8,arena_size[1]/2*.8] # coral position range (will be chosen randomly)
            # corals_pos = np.random.uniform(coral_pos_range[0],coral_pos_range[1],size=[n_corals,2])
            # Goal_rnd_pos = np.random.uniform(Goal_min_max_pos[0], Goal_min_max_pos[1], size=[1, 2])
            # Goal_pos=[Goal_rnd_pos[0][0],Goal_rnd_pos[0][1]]
            # #define food quadrant %replace end of line to enabl all 4 corners
            # Goal_quadrant_sign=[[1,1],[1,-1]][np.random.randint(0,2)]#,[-1,1],[-1,-1]][np.random.randint(0,4)]
            # Goal_pos=[Goal_pos[0] * Goal_quadrant_sign[0],Goal_pos[1] * Goal_quadrant_sign[1]]
            # # and agent initial position accordingly
            # agent_init_pos = [-(arena_size[0]/2+.1)*Goal_quadrant_sign[0],
            #                   -(arena_size[1]/2+.1)*Goal_quadrant_sign[1]]
        ### test maximal rates ###
            # RateScale = [0.1, 0.5, 1, 2, 10]
        #### test attractor ###
            # Goal_pos=[agent_init_pos[0], agent_init_pos[1]+0.5]


        # previously tested constant parameters
            ang_bin=15
            df=2
            eps=.03
            free_turn_bins=1#[1,2,1,2]
            max_bins_turn=3#[5,5,6,6] #corresponds to ang_bin*max_bins_turn turning range
            m_taus = [3]#[1,3,5] #modulate steer_time-constant
            steer_thres=-1.18 #this will yield thresholds in the range 0.2:0.3



        #### choose navigation strategy by commenting/uncommenting blocks ###:

            # # run with no memory:
            tic()
            model = NO_MEM(m_tau=m_taus[0],steer_thres=steer_thres,telen_neurons=telen_neurons, ang_bin=ang_bin, corals_pos=corals_pos,
                            Goal_pos=Goal_pos, df=df, eps=eps,Path=save_path,
                            free_turn_bins=free_turn_bins, max_bins_turn=max_bins_turn, seed=seed,
                            sim_time=sim_time, sim_num=sim_num,
                            plt=plt_flag, vid=vid_flag,agent_init_pos=agent_init_pos)
            model.Start()
            sim_num+=1
            toc()


             # run once with Azimuth:
            tic()
            model=AZIMUTH(m_tau=m_taus[0],steer_thres=steer_thres,telen_neurons=telen_neurons,ang_bin=ang_bin, corals_pos=corals_pos,
                                        Goal_pos=Goal_pos, df=df, eps=eps, Path=save_path,
                                        free_turn_bins=free_turn_bins, max_bins_turn=max_bins_turn, seed=seed,
                                        sim_time=sim_time,sim_num=sim_num,
                          plt=plt_flag,vid=vid_flag,agent_init_pos=agent_init_pos)
            model.Start()
            sim_num += 1
            toc()

            # run with HP:

            tic()
            model = HP(arena_size=arena_size,m_tau=m_taus[0],steer_thres=steer_thres,telen_neurons=telen_neurons, ang_bin=ang_bin, corals_pos=corals_pos,
                        Goal_pos=Goal_pos, df=df, eps=eps, RateScale=RateScale[rep],
                        free_turn_bins=free_turn_bins, max_bins_turn=max_bins_turn, seed=seed,
                        sim_time=sim_time, sim_num=sim_num, Path=save_path,
                            plt=plt_flag, vid=vid_flag,agent_init_pos=agent_init_pos)
            model.Start()
            sim_num+=1
            toc()

            # run with HP_max:

            tic()
            model = HPmax(m_tau=m_taus[0],steer_thres=steer_thres,telen_neurons=telen_neurons, ang_bin=ang_bin, corals_pos=corals_pos,
                        Goal_pos=Goal_pos, df=df, eps=eps, Path=save_path,
                        free_turn_bins=free_turn_bins, max_bins_turn=max_bins_turn, seed=seed,
                        sim_time=sim_time, sim_num=sim_num,
                            plt=plt_flag, vid=vid_flag,agent_init_pos=agent_init_pos)
            model.Start()
            sim_num+=1
            toc()

            # run with default interceps:

            tic()
            model = Default_inter(m_tau=m_taus[0],steer_thres=steer_thres,telen_neurons=telen_neurons, ang_bin=ang_bin, corals_pos=corals_pos,
                        Goal_pos=Goal_pos, df=df, eps=eps, RateScale=RateScale[rep],
                        free_turn_bins=free_turn_bins, max_bins_turn=max_bins_turn, seed=seed,
                        sim_time=sim_time, sim_num=sim_num, Path=save_path,
                            plt=plt_flag, vid=vid_flag,agent_init_pos=agent_init_pos)
            model.Start()
            sim_num+=1
            toc()











            ### below codes are under construction ###

            # #run with memory
            # tic()
            # model=MEM(memorize_corals=4,m_tau=m_taus[0],steer_thres=steer_thres,
            #           telen_neurons=telen_neurons,ang_bin=ang_bin, corals_pos=corals_pos,
            #               Goal_pos=Goal_pos, df=df, eps=eps,
            #               free_turn_bins=free_turn_bins, max_bins_turn=max_bins_turn, seed=seed,
            #               sim_time=sim_time,sim_num=sim_num, Path=save_path,
            #               plt=plt_flag,vid=vid_flag,agent_init_pos=agent_init_pos)
            # model.Start()
            # sim_num += 1
            # toc()

            # #run with COMBO
            # tic()
            # model=COMBO(memorize_corals=4,m_tau=m_taus[0],steer_thres=steer_thres,
            #             telen_neurons=telen_neurons,ang_bin=ang_bin, corals_pos=corals_pos,
            #               Goal_pos=Goal_pos, df=df, eps=eps,
            #               free_turn_bins=free_turn_bins, max_bins_turn=max_bins_turn, seed=seed,
            #               sim_time=sim_time,sim_num=sim_num, Path=save_path,
            #               plt=plt_flag,vid=vid_flag,agent_init_pos=agent_init_pos)
            # model.Start()
            # sim_num += 1
            # toc()

            ###############################
