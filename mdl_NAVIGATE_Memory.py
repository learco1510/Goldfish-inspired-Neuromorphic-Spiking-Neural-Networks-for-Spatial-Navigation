import numpy as np
import nengo
# import nengo_spa as spa
import math, pickle, time
import cv2
# from numpy import ndarray
from scipy import stats
from utils import mypause
import matplotlib, matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.io import savemat
import nengo.spa as spa
import random
from ismember import ismember
import nengo_ocl
import os


class BVC_NAV:

    def __init__(self,Path,memorize_corals,m_tau,steer_thres,telen_neurons,ang_bin,corals_pos,Goal_pos,agent_init_pos,df,eps,free_turn_bins,max_bins_turn,seed,sim_time,sim_num,plt,vid):

        # force-steer
        # self.forceSteer = None
        # parameters to optimize
        self.memorize_each_time=memorize_corals
        self.m_tau = m_tau
        self.sig_thrsh = 10 ** steer_thres

        #model parameters
        self.goal_reached=0 #this will be 1 when simulation is over
        self.stop_sim = False  # this will be TRUE when simulation is over
        self.seed=None
        self.sim_time=sim_time #maximal simulation time in sec

        # default parameters
        self.telen_neurons = telen_neurons  # ~total units in the telencephalon
        self.init_pos=agent_init_pos #start maze here
        self.x = agent_init_pos[0]
        self.y = agent_init_pos[1]
        self.amp_gain = 1 / 100 #gain for 'simple' neurons (so FR is in range 1-4)
        self.ang_bin_size_deg = ang_bin #BVC directional resolution
        self.n_bins_max_turn = max_bins_turn  #maximal direction (in bins of ang_bin_size_deg each) to consider when turning
        self.n_bins_clear_path = free_turn_bins #how many bins to test if path is clear
        self.turn_angle_CCW=0 #initial turn angle
        self.smooth_coeff=.9 #for each turn, take into account 0.25 from previous turn
        self.steer_out=float(0) #by default, do not turn
        self.df=df #factor that modulates the minimal distance for slowing down from 20 cm
        self.eps=eps #distance from coral to fully stop
        #arena parameters
        self.arena_size=[4, 4] #arena size in [m]
        self.corals_pos = corals_pos
        self.n_corals=len(corals_pos)
        self.Goal_pos = Goal_pos #feeding position
        self.GoalDirection=np.arctan2( (Goal_pos[1]-self.init_pos[1]),
                                      (Goal_pos[0]-self.init_pos[0])  )/np.pi #initial goal-direction
        self.default_hd = self.GoalDirection #default default hd
        self.initial_default_hd = self.default_hd

        #corals to memorize- choose random 10
        self.n_corals_2_memorize=10
        self.Corals2Memorize = list(np.sort(np.random.randint(0,self.n_corals,self.n_corals_2_memorize)))

        #probe-plot parameters
        self.plot_config=plt
        self.vid_flag=vid
        # if self.plot_config:
        #     self.num_plots=0
        #     plt.ion()
        #     plt.show()
        self.probes={}

        #save data parameters
        folder_path = "Matlab Output/" + Path + "/"  # output directory
        os.makedirs(folder_path, exist_ok=True)  # create output directory if it doesn't exist already
        mat_folder_name = folder_path + "/" + Path + " mat"  # output directory for mat files
        os.makedirs(mat_folder_name, exist_ok=True)  # create output directory if it doesn't exist already
        self.mat_name = mat_folder_name + "/SIM_" + str("%.3d" % sim_num)
        self.pic_name = "Matlab Output/SIM_" + str("%.3d"%sim_num)
        self.input_params_dict = {"telen_neurons": telen_neurons, "sim_time": sim_time,
                                  "FOV_res":ang_bin,"seed":seed,"arena_size":self.arena_size,
                                  "Goal_pos":Goal_pos,"corals_pos":corals_pos,"DistSlow":df*20,
                                  "DistStop":eps,"FrontFOV":max_bins_turn*2*ang_bin,
                                  "FreePathFOV":ang_bin*(1+free_turn_bins),"SmoothTurnCoef":self.smooth_coeff}


        # construct neuromorphic model
        self.dt = 0.001 #simulation step size
        self.buildModel()


    def buildModel(self):

        # service function to add named probes
        def _add_probe(element, label):
            self.probes[label] = nengo.Probe(element, label=label, synapse=0.01)
            # if self.plot_config:
                # plt.figure(self.num_plots)
                # self.num_plots += 1

        # limit simulation time function
        def _max_stop_time(t):
            if t>self.sim_time: #maximal simulation time if goal wasn't reached
                self.stop_sim = True
            return self.stop_sim

        ################### Boundary encoding parameters ###########################
        ''' #optional- load exsiting distributions
        save_path = 'C:/Users\Dell/anaconda3\envs/nengo\Scripts\Lear/'
        BVC_intercepts = np.load(save_path + 'BVC_intercepts.npy')
        BVC_Rates = np.load(save_path + 'BVC_Rates.npy')
        '''
        ################## Fish BVC intercepts Distribution #####################
        fish_receptive_fields = \
            [0.303565699399023, 0.150651881887490, 0.612954089261763,
             1.32902305046175, 0.237358465697817, 0.572593575787876,
             0.804672645959743, 0.468032848207433, 0.465103040497480,
             0.538975230941517, 0.131637644359929, 1.79196669576041,
             0.702770519299926, 0.346483106391552, 0.415090154920481,
             1.42381073514439, 0.218941453593792, 1.28263930362589,
             0.319611957288709, 0.296690418408916, 0.127691451940062,
             0.821042813569916, 0.611300555983888, 0.0706456431783326,
             1.83412056590551, 1.78588767232961, 0.543818191887459,
             0.820062851642834, 1.00724116911013, 0.189768947502303,
             0.900946696238341, 0.228138317181491, 0.0195893010264571,
             1.86946161637729, 1.70685573025602]
        fish_receptive_fields_cm = np.array(fish_receptive_fields,
                                            dtype=float) * 0.7  # make it in cm
        # estimate distribution
        ae, loce, scalee = stats.skewnorm.fit(fish_receptive_fields_cm)
        ################### Fish BVC Rate Distribution #####################
        PeakRateBVC_Fish = \
            [1.23179012004551, 0.652634682539281, 4.09152522548310,
             1.59885112929847, 1.14081018374045, 6.54274691116549,
             2.21902173988490, 1.53838548171947, 10.2606854396450,
             0.326835293519020, 0.107522721333319, 2.38158063055244,
             0.199934324014590, 2.04147978313486, 2.09880178896422,
             1.99088166689020, 2.76220348861795, 5.51890082912580,
             4.23048845852580, 2.48385349846702, 6.14630410033045,
             0.392574798790659, 1.25628115518902, 1.05271181026214,
             0.529461993040073, 0.470916168665174, 3.25946160652742,
             0.557688406685955, 3.30565839449426, 1.51284640149477,
             2.06711978413453, 0.610938792413799, 0.441433752122931,
             2.15427182373612, 6.68880488440973]
        ae_rate, loce_rate, scalee_rate = stats.skewnorm.fit(PeakRateBVC_Fish)
        ################### Fish HP cells Rate Distribution #####################
        HP_rate = [0.2004, 0.2493, 0.2421, 0.4544, 0.2421, 0.4690,
                   0.4583, 0.5749, 2.2806, 0.9800]
        hp_rate, hploc_rate, hpscale_rate = stats.skewnorm.fit(HP_rate)
        '''
        ################### Fish velocity Rate Distribution #####################
        Vel_rate = [0.197338698747390, 0.380581776155509, 0.00502881088493350,
                    0.0364070833883619,
                    0.0274849640740761, 0.0399434968347826, 0.0959080169846669,
                    0.0626902630243608,
                    0.984263214195281, 0.141873793252625, 0.0179027099720430,
                    0.278798500697151,
                    0.0272255916653539, 4.40721626968568, 0.0839577370743340,
                    11.9883355299775,
                    0.248319274692277, 0.328447548766507, 0.230424234436725,
                    0.164223774383254,
                    0.866626664817651, 0.817386513407557, 1.41039241529147,
                    0.199830486833172]
        v_rate, vloc_rate, vscale_rate = stats.skewnorm.fit(Vel_rate)
        ################### Fish velocity receptive field Distribution #####################
        vel_rec = [8.01242153733998, 8.25676942317176, 5, 1, 5,
                   3.74763034523908,
                   0.559199852102832,
                   13.0357053721238, 0.196123594968107, 0.503285030846904, 5,
                   5.85266779897490,
                   0.829365741476824, 8.28690525620088, 0.0326276173643798,
                   22.9369932190041,
                   12.3153891189659, 10.1130683005553, 2.79621194110437,
                   2.43798519662587,
                   11.1893592815419, 8.91421085953175, 8.06612542053255,
                   0.281776502680117]
        vel_rec = np.array(vel_rec, dtype=float) * 0.01  # make it in cm
        v_rec, vloc_rec, vscale_rec = stats.skewnorm.fit(vel_rec)
        '''
        ##########################################################################

        # values to test
        telen_neurons = self.telen_neurons
        ang_bin_size_deg = self.ang_bin_size_deg
        n_bins_max_turn = self.n_bins_max_turn
        n_bins_clear_path=self.n_bins_clear_path
        corals_pos = self.corals_pos
        n_corals = len(corals_pos)  # total numer of corals
        Goal_pos = self.Goal_pos
        arena_size = self.arena_size # arena size in meter

        #constant parameters
        tau = 0.1 #synaptic time for path integration
        tau2=2*tau/10 #default synaptic time for other units
        attractor_scale = 1#0.1  # scale attracotr ensemble
        max_speed = 0.5  # up to 50cm/s
        speed_shift = -max_speed / 2  # shift for symmetricity purposes
        v0 = max_speed  # desired (initial) speed
        n_neuron_prop = np.round(np.array([20/132,
                    35/196])*telen_neurons) #proportional n: vel/hd/speed, BVC
        n_neurons=int(n_neuron_prop[0]) #for the speed/hd/velocity units
        self.max_angle = np.pi  # normalization factor for angles
        fish_size = 0.1  # fish radius to avoid collisions - 10cm
        self.coral_size = 0.15  # coral radius to avoid collisions - 15cm
        feed_visual_dist = 2 * self.coral_size  # distance in which food is visual for sure
            #BVC parameters:
        self.BVC_rec_field = 1.4 + self.coral_size # BVC max receptive field
        n_LIDAR_pts = int(360 / ang_bin_size_deg) #n bvc directions
        neurons_per_angle = int(n_neuron_prop[1]/n_LIDAR_pts) #neurons per dim
        angle_array = np.linspace(0,
                                  2 * self.max_angle - 2 * self.max_angle / n_LIDAR_pts,
                                  n_LIDAR_pts)  #LIDAR array 0 to 2pi
        white_noise = np.random.uniform(-0.003, 0.003,
                                        size=np.size(angle_array))  # add noise so no LIDAR values will be the same
        ang_bin_size = angle_array[1] - angle_array[0] #effectively ang_bin_size_deg in [rad]
            # Speed control parameters
        momentum_coeff = 0.1  # how fish shape distores its breaks ability
        eps = self.eps  # distance threshold for full-brakes - 1 cm
        deceleration_coeff = 3  # [tested in matlab]
        v_eps = deceleration_coeff * eps
        D = (momentum_coeff * max_speed ** 2 + max_speed / deceleration_coeff +
             eps + fish_size + self.coral_size) * self.df # *3 # Effective Distance (df*20cm) to start slow down:

        # define initial position
        def init_pos(t):
            if t < tau:
                return self.init_pos
            else:
                return [0, 0]

        # angle/distance functions
        def calc_dist(x):
            return (x[0] ** 2 + x[1] ** 2) ** 0.5
        def calc_angle(t, x):
            return np.arctan2(x[1], x[0]) / self.max_angle

        # speed and direction --> velocity
        def get_VxVy(t, x):
            return [x[1] * np.cos(x[0]), x[1] * np.sin(x[0])]

        #LIDAR read
        def read_LIDAR(t, x):
            X, Y, HD = x
            distArray = self.BVC_rec_field * np.ones(n_LIDAR_pts)

            #implement arena edges into array
            self.x_out=1 #variable that helps tune the ydirection
            m_eps = 3 #how many epsilons from the wall to allow freely
            n_bins_quartile = int(n_LIDAR_pts / 4)  # how many bins to block entire side
            bin_hd = int((np.abs(angle_array - (HD + 2 * np.pi) % (2 * np.pi) )).argmin())  # index of current HD
            if self.x + self.arena_size[0] / 2 < m_eps * self.eps:  # wall on the left
                self.x_out = 0.8
                Q = int(0.5 * n_LIDAR_pts)
                wall_inds = list( (np.arange(Q - n_bins_quartile - bin_hd + 1, Q + n_bins_quartile - bin_hd))%(n_LIDAR_pts) )
                distArray[wall_inds] = m_eps * self.eps
            if -self.x + self.arena_size[0] / 2 < m_eps * self.eps:  # wall on the right
                self.x_out = 1.2
                Q = int(0)
                wall_inds = list( (np.arange(Q - n_bins_quartile - bin_hd + 1, Q + n_bins_quartile - bin_hd))%(n_LIDAR_pts) )
                distArray[wall_inds] = m_eps * self.eps
            if self.y + self.arena_size[1] / 2 < m_eps * self.eps:  # wall on the bottom
                Q = int(0.75 * n_LIDAR_pts)
                wall_inds = list( (np.arange(Q - n_bins_quartile - bin_hd + 1, Q + n_bins_quartile - bin_hd))%(n_LIDAR_pts) )
                distArray[wall_inds] = m_eps * self.eps
            if -self.y + self.arena_size[1] / 2 < m_eps * self.eps:  # wall on the top
                Q = int(0.25 * n_LIDAR_pts)
                wall_inds = list( (np.arange(Q - n_bins_quartile - bin_hd + 1, Q + n_bins_quartile - bin_hd))%(n_LIDAR_pts) )
                distArray[wall_inds] = m_eps * self.eps

            # distArray+=white_noise
            angle_all= np.ones(n_corals) #allocate memory (default values)
            dist_all=self.BVC_rec_field * np.ones(n_corals) #allocate memory (default values)
            for i in range(0, n_corals):
                dy = corals_pos[i][1] - Y
                dx = corals_pos[i][0] - X
                angle = np.arctan2(dy, dx)-HD
                angle_all[i] = (angle + 2 * np.pi) % (
                        2 * np.pi)  # shift to [0,2pi] representation
                dist_all[i]=np.clip(calc_dist([dx, dy]), 0,self.BVC_rec_field) # fit to max receptive field

            #get coral index by sorted distance from far to near:
            Coral_sort_ind = np.argsort(dist_all)
            Coral_sort_ind=Coral_sort_ind[::-1] #sort descending order
            for i in range(0, n_corals): #now go coral by coral from further to nearer
                #read angle and distance from previous loop
                angle=angle_all[Coral_sort_ind[i]]
                dist=dist_all[Coral_sort_ind[i]]
                #take coral width into account for LIDAR read:
                coral_Aperture = 2*np.arctan(self.coral_size/dist) #portion of visual field for the coral
                coral_LIDAR_pts = np.round(coral_Aperture/ang_bin_size) #how many bins represented by this coral
                angle_array_idx = (np.abs(angle_array - angle)).argmin() #angle argument index in LIDAR array
                if coral_LIDAR_pts==2:
                    if angle_array[angle_array_idx]>angle:
                        angle_array_idx=np.arange(angle_array_idx-1,angle_array_idx+1,1) #add angle to the right
                    else:
                        angle_array_idx = np.arange(angle_array_idx, angle_array_idx+2, 1)  # add angle to the left
                elif coral_LIDAR_pts>2: #maximal 3 angles per coral, add 1 from each side
                    angle_array_idx = np.arange(angle_array_idx-1, angle_array_idx + 2, 1)
                if coral_LIDAR_pts>1:
                    angle_array_idx=list(np.mod(angle_array_idx,n_LIDAR_pts)) #fold to arrays range, make list

                distArray[angle_array_idx] = dist #implement to array

            #this defines priorities in cases there are many optional turning angles
            priority_noise=np.abs((np.round(n_LIDAR_pts/2)-np.arange(0,n_LIDAR_pts))/20000)
            self.Coral_sort_ind=Coral_sort_ind
            return distArray + priority_noise - self.BVC_rec_field #return array shifted to [-maxFiels, 0]

        # define turning function
        def Turn(V, angle):
            """
            Rotate a point counterclockwise by a given angle around [0,0].
            The angle should be given in radians.
            """
            vx, vy = V
            vx_rot = math.cos(angle) * (vx) - math.sin(angle) * (vy)
            vy_rot = math.sin(angle) * (vx) + math.cos(angle) * (vy)
            return vx_rot, vy_rot

        #efficient turn with BG model
        def Steer(t,x):
            def _set_steer(t,DA):
                # sig_thrsh = 0.1  # threshold for recognizing steering direction
                ind = np.argmax(x)  # select max. steer direction
                if DA[ind] < self.sig_thrsh:  # if under threshold, drive forward
                    steer = 0
                else:
                    steer = ang_bin_size*(ind-n_bins_max_turn)
                # self._ipc_state.steer = self.steer  # set steer command
                return steer
            def getBVC_net(x):
                with nengo.Network() as BVC_net:
                    BVC_net.input=nengo.Node(x)
                    BVC_net.output=nengo.Node(output=None, size_in=len(x))
                    nengo.Connection(BVC_net.input,BVC_net.output)
                return BVC_net

            with nengo.Network() as tmp_net:
                BVC_array=getBVC_net(x) #get BVC array as input node
                n_sample_filter = len(x)  # number of channels remaining
                bg = nengo.networks.BasalGanglia(
                    n_sample_filter)  # basal ganglia, use default number of neurons/ensemble
                # if self.steer_model:
                #     nengo.Connection(ext.output, bg.input, transform=0.75)
                # else:
                nengo.Connection(BVC_array.output, bg.input)#, transform=1)
                # _add_probe(bg.output, "bg_output")
                th = nengo.networks.Thalamus(n_sample_filter)  # thalamus, use default number of neurons/ensemble
                nengo.Connection(bg.output, th.input)
                # _add_probe(th.output, "th_output")
                steer_out = nengo.Node(output=_set_steer, size_in=n_sample_filter, size_out=1)  # connect to output
                nengo.Connection(th.output, steer_out)

                # Integrate steer over time. Using longer synapse to smooth-out noise
                steer_tau_i = self.m_tau*tau
                steer_sum_enc = nengo.Ensemble(n_neurons=n_neurons, dimensions=1, radius=ang_bin_size*n_bins_max_turn)
                nengo.Connection(steer_out, steer_sum_enc, transform=steer_tau_i)
                nengo.Connection(steer_sum_enc, steer_sum_enc, function=lambda x: x, synapse=steer_tau_i)
                steer_sum=nengo.Node(size_in=1)
                nengo.Connection(steer_sum_enc,steer_sum)
                self.steer_out=0 if steer_sum.output is None else steer_sum.output
                # nengo.Probe(steer_sum_enc,'Steer', synapse=0.01)
            # self.simSteer = nengo.Simulator(tmp_net, dt=0.01, seed=self.seed, optimize=True)

            return self.steer_out

        # conditioned speed function
        def _get_potential_velocity(t, bvns_data):
            # default outputs
            v = bvns_data[:2]  # inital velocity

            min_dist = self.eps  # minimal distance to enable speed
            dist_per_angle = bvns_data[
                             2:] + self.BVC_rec_field  # distance per egocentric angle [0,2pi]

            dist_FRONT = np.min(np.concatenate(
                (dist_per_angle[:n_bins_clear_path + 1],
                 dist_per_angle[-n_bins_clear_path:]))
            )  # front distance
            dist_0 = dist_per_angle[
                         0] - fish_size / 2 - self.coral_size  # avoid f2f collision
            d = dist_FRONT - fish_size / 2 - self.coral_size  # effective distance from boundary

            # do I need to slow down? get proportion:
            relative_D = (d - min_dist) / (
                    D - min_dist)  # relative distance
            # go axis by axis and slow/speed
            v1 = [0, 0]  # initialize
            for ax in [0, 1]:
                curr_speed = np.abs(v[ax])
                curr_dir = np.sign(v[ax])
                # if getting close
                if d < D:
                    speed = min((curr_speed + v_eps) * relative_D,
                                curr_speed)
                # if far enough, accelerate to max_speed
                else:
                    speed = min(v[ax] + deceleration_coeff * tau,
                                max_speed)
                # v1[ax] = curr_dir * np.abs(speed)
                v1[ax] = curr_dir * np.clip(np.abs(speed), 0, max_speed)
            v = v1  # set desired velocity

            # def setGoalDirection(GD,T0,T): #basically delay the turn times
            #     # assisting function to set goal direction to GD between To and T
            #     GD0=self.GoalDirection
            #     if T0 < t < T:
            #         self.GoalDirection=GD

            # #if force-steer is enabled, skip the rest of calculations
            # if self.forceSteer is not None and np.mod(t+.5,1)<self.dt*turn_time: #once in 1s during 100 ms
            #     v=Turn(v,self.forceSteer*self.max_angle/turn_time)
            #     self.GoalDirection=calc_angle(t,v)
            # elif dist_0 < min_dist and np.mod(t * 10,1) < self.dt * turn_time*0.67:  # once in 100ms during 67 ms
            #     # when too close, turn 2/3*180deg ccw
            #     v = Turn(v, np.pi / turn_time)
            #     self.GoalDirection = calc_angle(t, v)
            # else:
            if 1:  # try to go desired direction
                des_HD = np.mod(bvns_data[-1],
                                2 * np.pi)  # shift to [0,2pi] representation
                LIDAR_des_HD_ind = (np.abs(angle_array - des_HD)).argmin()
                all_des_HD_inds = list(np.mod(
                    range(LIDAR_des_HD_ind - n_bins_clear_path,
                          LIDAR_des_HD_ind + n_bins_clear_path + 1), n_LIDAR_pts
                ))

                # all_des_HD_inds = np.mod(all_des_HD_inds,
                #                          n_LIDAR_pts)  # shift list to 0,n_LIDAR_pts-1 range
                dist_des_HD = dist_per_angle[list(all_des_HD_inds)]
                if np.max(dist_des_HD)>D: #if path is clear
                    max_dir_ind=np.argmax(dist_des_HD) #choose clearest path
                    turn_angle_CCW=all_des_HD_inds[max_dir_ind]*ang_bin_size
                    # turn_angle_CCW=des_HD
                    # v=Turn(v,turn_angle_CCW)
                # if this fails, maximize distance in front:
                else:
                    # best turning in range [-self.max_angle_turn,self.max_angle_turn],
                    dist_array_wide_front = np.concatenate(
                        (dist_per_angle[-(n_bins_max_turn+1):],
                         dist_per_angle[:n_bins_max_turn+1+1]))
                    # # #average with neighbours
                    # dist_array_wide_front= (dist_array_wide_front[0:-2]+
                    #                             dist_array_wide_front[1:-1]+
                    #                             dist_array_wide_front[2:])/3

                    turn_angle_CCW=Steer(t, dist_array_wide_front) #assign steer_out to self  ###self.steer_out

                #perform turn
                # smooth with previous turn
                self.turn_angle_CCW = self.smooth_coeff*turn_angle_CCW+(1-self.smooth_coeff)*self.turn_angle_CCW
                # if np.mod(t, tau) < self.dt:
                v = Turn(v, self.turn_angle_CCW)
                # if steered, update desired direction backwards
                if np.mod(t, tau) < self.dt:
                    self.default_hd -= (self.turn_angle_CCW / self.max_angle)

            return np.array([v[0], v[1], dist_FRONT, self.turn_angle_CCW])

        #memory function to update
        def memorized_direction(t):
            D = 64  # spa dimensions  #### test this number!!
            t_start_memory=0.1 #time from simulation beginning to start using the memory
            memorize_each_time = self.memorize_each_time # 3 corals to memorize each time (test this number)- more associative than remembering all...
            if t<t_start_memory: #initial values until LIDAR read is implemented
                # self.idx_nearby_Corals = self.Corals2Memorize[int(np.random.randint(0, self.n_corals_2_memorize, memorize_each_time))]  # initial set of corals to memorize
                self.idx_nearby_Corals = self.Corals2Memorize[:memorize_each_time]  # initial set of corals to memorize
                query_index=0
                # query_angle2fish=0
            else:
                all_coral_members = ismember(self.Coral_sort_ind,self.Corals2Memorize) #find the nearest corals to remember
                coral_members_ind=[] #list to add true values
                for i in range (0, len(all_coral_members[0])):
                    if all_coral_members[0][i]:
                        coral_members_ind.append(i)
                query_index = self.Coral_sort_ind[coral_members_ind[0]]   #index of the nearest coral to query
                # query_angle2fish=self.Fish2CoralAngle[coral_members_ind[0]] #direction of nearest coral relative to fish
                if np.mod(t, 2):   # once in 2 sec, re-read array of nearest corals to memorize
                    self.idx_nearby_Corals= self.Coral_sort_ind[coral_members_ind[:memorize_each_time]]


            idx_nearby_Corals = self.idx_nearby_Corals  # index of corals in coral list that are nearby to query
            corals_pos = self.corals_pos
            Goal_pos = self.Goal_pos
            n_corals = len(corals_pos)
            max_angle = self.max_angle
            ang_res = self.ang_bin_size_deg  # given as ang_bin
            # self.default_hd = self.GoalDirection
            ang_bin_shift = 10  # avoid alphabetical bugs from using 0,1,2 and 10,11,12 in the same array

            # angle array creation:
            bin_size_ang_array = np.deg2rad(ang_res) / max_angle  # theta^o bins in [-1,1] range
            ang_array = np.arange(-1, 1, bin_size_ang_array)


            # function to create a spa vocabulary
            def CreateVocab(D, prefix, vals):
                vocab = spa.Vocabulary(dimensions=D, unitary=True)
                for i in range(0, len(vals)):
                    word = prefix + str(vals[i])
                    vocab.parse(word)
                return vocab
            # create coral_id vocabulary
            Corals_id_vocab = CreateVocab(D, prefix='C', vals=np.arange(0, n_corals, 1))

            # calculate angles and create angles dictionary:
            # calculate angle by atan2:
            def calc_angle(dy, dx):
                return np.arctan2(dy, dx) / max_angle
            # assiting function to find the bin in ang_array each coral is associated with
            def find_ang_bin(ang):
                bin = np.mod(np.digitize(ang, ang_array),ang_array.size) + ang_bin_shift
                return bin
            # calculate angles as bin numbers in ang_array
            Coral2Goal_bin = []
            for i in range(0, n_corals):
                temp_ang = calc_angle(Goal_pos[1] - corals_pos[i][1], Goal_pos[0] - corals_pos[i][0])
                Coral2Goal_bin += [find_ang_bin(temp_ang)]
                # add suffix for repeating values to enable dictionary
            Coral2Goal_bin_with_suffix = []
            for i in range(0, n_corals):
                temp_val = Coral2Goal_bin[i]
                previous_occurences_of_temp_val = str(Coral2Goal_bin[:i].count(temp_val))
                new_val_with_suffix = int(str(temp_val) + previous_occurences_of_temp_val)
                Coral2Goal_bin_with_suffix.append(new_val_with_suffix)
                # create coral_id vocabulary
            Ang2Goal_vocab = CreateVocab(D, prefix='D', vals=Coral2Goal_bin_with_suffix)

            #create index-pattern to memorize with pauses between instances
            pattern2memorize = np.insert(np.arange(0,len(idx_nearby_Corals)), #duplicate pattern 0,0,1,1,2,2,3,3
                                         obj=np.arange(0,len(idx_nearby_Corals)),
                                         values=np.arange(0,len(idx_nearby_Corals)),axis=0)
            pattern2memorize = np.insert(pattern2memorize,  # insert after every pair a flag for 'not memorize'
                                         obj=np.arange(0, len(idx_nearby_Corals),2),
                                         values=len(idx_nearby_Corals)+1, axis=0)
            pattern_length=len(pattern2memorize)
            # pattern_bin_duration=0.1  #each bin should be sent for 100 ms, so pattern is 200ms->memorize, 100ms->rest..
            # pattern_total_duration = pattern_length*pattern_bin_duration

            # simultanouesly send id and desired direction to memory
            def input_id(t):
                x = pattern2memorize[int(np.mod(int(t * 10), pattern_length))]  #vary every 100 ms
                if x >= len(idx_nearby_Corals) or t < 0.1:
                    return '0'
                else:
                    c_ind = x  # integer in range of #corals_nearby
                    return Corals_id_vocab.keys[idx_nearby_Corals[c_ind]]

            def input_dir(t):
                x = pattern2memorize[int(np.mod(int(t * 10), pattern_length))]  #vary every 100 ms
                if x >= len(idx_nearby_Corals) or t < 0.1:
                    return '0'
                else:
                    c_ind = x  # integer in range of #corals_nearby
                    return Ang2Goal_vocab.keys[idx_nearby_Corals[c_ind]]

            # define query by idx of nearest coral to query from those that are nearby
            # function to create distance string to query
            def query_input(t):
                return Corals_id_vocab.keys[query_index]

            # calculate similarity of query_answer to input_vocabulary to get angle
            def calc_similarity(t, x):
                similarity_vector = spa.similarity(x, Ang2Goal_vocab)
                if t<0.01:
                    self.max_sim= np.max([0])
                else:
                    self.max_sim = np.max(similarity_vector)
                if self.max_sim > 0.1:  # set threshold for recognition
                    ang_str = Ang2Goal_vocab.keys[np.argmax(similarity_vector)]
                    def_hd = ang_array[int(ang_str[1:3]) - ang_bin_shift]
                else:
                    def_hd = self.default_hd
                self.GoalDirection=def_hd
                return self.GoalDirection



            model = spa.SPA()
            with model:
                # ang_probe = nengo.Node(ground_truth)

                # define IDs nd goal_directions of Corals
                model.CoralID = spa.State(D)
                model.GoalDir = spa.State(D)
                # memorize them over time
                model.Coral_input = spa.Input(CoralID=input_id, GoalDir=input_dir)
                model.memory = spa.State(D, feedback=1)

                # query direction by current nearest Coral
                model.map_query = spa.State(D)
                model.pos_input = spa.Input(map_query=query_input)

                model.answer = spa.State(D)

                actions = spa.Actions(
                    "memory = GoalDir * CoralID ",
                    "answer = memory * ~map_query",
                )

                model.cortical = spa.Cortical(actions)

                output_node = nengo.Node(calc_similarity, size_in=D)
                nengo.Connection(model.answer.output, output_node, synapse=0.1)

                # _add_probe(output_node,'GoalDirection_function')

            return self.GoalDirection

        # food stimulus function
        # def feed_or_not(t, x):
        #     if calc_dist(x) < feed_visual_dist*attractor_scale:
        #         return 1
        #     else:
        #         return 0
        threshold_attract=0.1
        def Food_Stim(t, x):
                if x[0]>threshold_attract:
                    return x[0] * x[1], x[0] * x[2]
                else:
                    return 0* x[1],0* x[2]
        def getGoalDirection(t):
            return self.GoalDirection

        # stop simulation variable
        def goal_reached(t, x):
            if calc_dist(x) < fish_size and t>0.1: #delay initial condition
                self.stop_sim = True
                return 1
            else:
                return 0

        self.model = nengo.Network("BVC_NAV", seed=self.seed)
        with self.model:

            # time variable to flag 'simulation end'
            Time = nengo.Node(_max_stop_time,size_out=1)

            #probe steer
            SteerNode=nengo.Node(self.steer_out)
            _add_probe(SteerNode,'Steer')

            #probe direction tuning by memory
            direction_memory_node=nengo.Node(memorized_direction)
            _add_probe(direction_memory_node, 'GoalDirection_model')

            # speed
            Agent_speed = nengo.Node([v0])  # speed of agent
            sp_shift_norm = nengo.Node([speed_shift])
            n_speed = nengo.Ensemble(n_neurons,
                                     dimensions=1, radius=max_speed/2,
                                     neuron_type=nengo.LIF(amplitude=self.amp_gain))
            nengo.Connection(Agent_speed, n_speed)
            nengo.Connection(sp_shift_norm, n_speed)

            # head direction
            # GoalDirection=getGoalDirection #goal direction as function of time
            # Agent_HD = nengo.Node([GoalDirection])  # displacement direction
            Agent_HD = nengo.Node(getGoalDirection)
            n_HD = nengo.Ensemble(n_neurons, dimensions=1,
                                  neuron_type=nengo.LIF(amplitude=self.amp_gain))
            nengo.Connection(Agent_HD, n_HD,synapse=tau)
            _add_probe(n_HD,'HeadDirection')

            # Velocity
            Agent_vel = nengo.Node(get_VxVy,size_in=2)  # Agent's velocity node
            nengo.Connection(n_HD, Agent_vel[0],
                             transform=self.max_angle,synapse=tau2)  # get angle [-pi,pi]
            nengo.Connection(n_speed,
                             Agent_vel[1],synapse=tau2)  # get speed [-max_speed/2,max_speed/2]
            nengo.Connection(sp_shift_norm, Agent_vel[1],transform=-1)  # set speed to [0,.5]

            n_vel = nengo.Ensemble(n_neurons,
                                   dimensions=2,radius=max_speed*np.sqrt(2),
                                   neuron_type=nengo.LIF(amplitude=self.amp_gain))
            nengo.Connection(Agent_vel, n_vel[:2],synapse=tau2)

            # position X,Y integrator
            initial_pos=nengo.Node(init_pos,size_out=2)
            # define the position variable during roaming
            # HP_Rates = stats.skewnorm(hp_rate, hploc_rate,
            #                           hpscale_rate).rvs(n_neurons)
            # scale_rate = 400 / np.max(
            #     HP_Rates) - 1  # just so the neurons would work properly
            # npos = nengo.Ensemble(n_neurons=n_neurons,
            #                       dimensions=2, radius=arena_size[1] / (2 ** 0.5),
            #                       max_rates=HP_Rates * scale_rate,
            #                       neuron_type=nengo.SpikingRectifiedLinear(amplitude=2 / scale_rate)
            #                       )

            npos = nengo.Ensemble(n_neurons*2, dimensions=2,
                                  radius=arena_size[0] * .5 * np.sqrt(
                                      2))  # x,y position up2:2
            nengo.Connection(initial_pos,npos,synapse=tau) #connect initial position to pos
            _add_probe(npos,'position')
            # calculate current position using integrator
            nengo.Connection(npos[:2], npos[:2], synapse=tau)

            # visual cues
            # for i in range(0, len(corals_pos)):
            #     track_boundaries = nengo.Node(corals_pos[i])
            Goal = nengo.Node(Goal_pos)

            # BVC neurons definition:
            BVC_intercepts = stats.skewnorm(ae, loce, scalee).rvs(n_neurons)
            BVC_intercepts /= np.max(BVC_intercepts)*1.001
            BVC_Rates = stats.skewnorm(ae_rate, loce_rate, scalee_rate).rvs(
                n_neurons)
            scale_rate = 90  # just so the neurons would work properly
            # make the emsemble array for each angle in different index
            # EA.EnsembleArray
            ea_BVC = nengo.networks.EnsembleArray(neurons_per_angle,
                          n_LIDAR_pts,
                          radius=self.BVC_rec_field,
                          encoders=[-1] * neurons_per_angle,
                          intercepts=np.random.choice(
                              BVC_intercepts,size=neurons_per_angle),
                          max_rates=np.random.choice(
                              BVC_Rates,size=neurons_per_angle) * scale_rate,
                          neuron_type=
                                      # nengo.PoissonSpiking(
                              # nengo.RectifiedLinear(amplitude=1 / scale_rate)
                                nengo.SpikingRectifiedLinear(amplitude=1 / scale_rate)
                                      # )
                          )

            # connect position and HD and encode BVC(LIDAR) data
            ea_BVC_input = nengo.Node(read_LIDAR, size_in=3,
                                      size_out=n_LIDAR_pts)
            nengo.Connection(npos[:2], ea_BVC_input[:2],synapse=2*tau2)
            nengo.Connection(n_HD, ea_BVC_input[2], transform=self.max_angle,synapse=2*tau2)
            nengo.Connection(ea_BVC_input, ea_BVC.input, synapse=2*tau2)
                #probe lidar array:
            lidar_probe = nengo.Node(size_in=n_LIDAR_pts)
            nengo.Connection(ea_BVC_input,lidar_probe)
            _add_probe(lidar_probe, 'LIDAR_ARRAY')


            # Goal encoding: feed and fast!
            # check goal path is clear and set as attractor when visual
            ############### TBD using nengo SPA ##################################
            # food visibility node by distance
            # goal_to_agent_diff = nengo.Node(feed_or_not, size_in=2, size_out=1)
            def feed_or_not(x):
                att=np.array([0,1]) #optional outputs
                dx,dy=x
                self.goal_dist=calc_dist(np.array([dx,dy]))
                if  self.goal_dist < feed_visual_dist * attractor_scale:
                    attract = att[1]
                else:
                    attract = att[0]
                return attract
            goal_to_agent_diff = nengo.Ensemble(n_neurons=3000, dimensions=2,radius = self.arena_size[1]/2)
            nengo.Connection(npos[:2], goal_to_agent_diff, synapse=tau, transform=attractor_scale)
            nengo.Connection(Goal, goal_to_agent_diff, synapse=tau, transform=-attractor_scale)
            Food_stimulus = nengo.Node(Food_Stim, size_in=3, size_out=2)
            feed_bolean=nengo.Connection(goal_to_agent_diff, Food_stimulus[0], function=feed_or_not, synapse=tau)
            nengo.Connection(Goal, Food_stimulus[1:])
            nengo.Connection(npos, Food_stimulus[1:], transform=-1, synapse=tau)
            _add_probe(feed_bolean, 'Go2Goal')

            # above yields X_food-X, Yfood-Y when visual, 0,0 when not
            # therefore, adding it to npos will yield X+0,Y+0 when not visual,
            # and X+X_food-X, Y+Yfood-Y when visual (i.e. attractor):
            nengo.Connection(Food_stimulus, npos, synapse=tau)

            # speed control: if distance from boundar is getting smaller, decrease speed.
            potential_velocity = nengo.Node(_get_potential_velocity,
                                            size_in=2 + n_LIDAR_pts + 1,
                                            size_out=4)
            nengo.Connection(ea_BVC.output, potential_velocity[2:-1],
                             synapse=tau)
            nengo.Connection(Agent_vel, potential_velocity[:2],synapse=tau)

            # translate speed output into HD shift
            potential_HD = nengo.Node(calc_angle, size_in=2, size_out=1)
            nengo.Connection(potential_velocity[:2], potential_HD,synapse=2*tau)
            HD_error = nengo.Node(size_in=1)
            nengo.Connection(Agent_HD, HD_error, transform=-1,synapse=tau2)
            nengo.Connection(potential_HD, HD_error,synapse=tau2)
            nengo.Connection(HD_error, n_HD,synapse=tau)
            # send HD error to potential speed to check availability
            nengo.Connection(HD_error, potential_velocity[-1],
                             transform=-self.max_angle,synapse=tau)

            ################ probe pot_vel_output#############
            probe_v_pot=nengo.Node(size_in=2)
            nengo.Connection(potential_velocity[:2],probe_v_pot)
            # _add_probe(probe_v_pot,'V_potential')
            probe_dist_front=nengo.Node(size_in=1)
            nengo.Connection(potential_velocity[2],probe_dist_front)
            # _add_probe(probe_dist_front, 'Front_Dist')
            probe_turn=nengo.Node(size_in=1)
            nengo.Connection(potential_velocity[-1],probe_turn)
            # _add_probe(probe_turn, 'Turn_Angle')

            # apply velocity error to agent
            velocity_error = nengo.Ensemble(n_neurons=n_neurons,
                                           dimensions=2,
                                           radius=max_speed*np.sqrt(2),
                                           neuron_type=nengo.LIF(amplitude=self.amp_gain))

            # displacement_error=nengo.Node(size_in=2,size_out=2)
            nengo.Connection(n_vel[:2], velocity_error, transform=-1,synapse=tau2*5)
            nengo.Connection(potential_velocity[:2], velocity_error,synapse=tau2*5)

            # Actual velocity output from the motor system
            actual_velocity = nengo.Ensemble(n_neurons=n_neurons,
                                             dimensions=2,
                                             radius=max_speed*np.sqrt(2),
                                             neuron_type=nengo.LIF(amplitude=self.amp_gain))
            # _add_probe(actual_velocity, 'Vel_out')
            nengo.Connection(n_vel[:2], actual_velocity[:2],synapse=tau2)
            nengo.Connection(velocity_error, actual_velocity[:2],synapse=tau2)

            # connect the controlled speed to npos
            nengo.Connection(actual_velocity, npos[:2],
                             synapse=tau, transform=tau)

            # stop simulation when reaching goal
            self.goal_reached = nengo.Node(goal_reached, size_in=2, size_out=1)
            _add_probe(self.goal_reached, 'GOAL')
            nengo.Connection(Goal, self.goal_reached)
            nengo.Connection(npos, self.goal_reached, transform=-1)

        self.sim = nengo_ocl.Simulator(self.model, dt=self.dt,  #1ms step size
                                       seed=self.seed)


    def Start(self):
        # step + optional: live plot (currently doesn't work)
        t=0
        step_counter=int(1) #count steps of sim to plot every other step

        # create video
        FPS = 10
        vid_size = [300, 300]
        my_dpi = 100
        if self.vid_flag:
            vidOut=cv2.VideoWriter('{}_video.avi'.format(self.pic_name)
                                   ,cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (vid_size[0], vid_size[1]))

        #define stop_time 1 sec after stop_sim
        self.StopT = self.sim_time
        stopflag=False

        turn_t = self.dt  # initial time flag (dummy to create time phase between turns)

        # while not self.stop_sim:
        while t < self.StopT+0.1:
            t+=self.dt #dummy variable to get current time
            step_counter+=1 #count steps
            self.sim.run_steps(1, progress_bar=False)  # single simulation step

            if self.stop_sim and not stopflag:
                self.StopT = t + 1 #stop in 1 sec
                stopflag=True  #don't update this value anymore

            goal_reached=self.sim.data[self.probes['GOAL']][-1]


            #if out of arena change direction
            xy = self.sim.data[self.probes['position']][-1:]
            self.x=xy[:, 0]
            self.y=xy[:, 1]
            # if abs(x)>self.arena_size[0]*.95/2 or abs(y)>self.arena_size[1]*.95/2:
            #     # do this only once a second:
            #     if np.mod(t,1)<self.dt:
            #         self.forceSteer=.67
            # else:
            #     self.forceSteer=None
            # if abs(x)>self.arena_size[0]/2 or abs(y)>self.arena_size[1]/2:
            #     if t>turn_t-1: # do this only once a second
            #         turn_t=t
            #         self.GoalDirection = self.GoalDirection+1 if self.GoalDirection<self.eps else self.GoalDirection-1
            #     self.default_hd=self.GoalDirection
            # else:
            #     self.default_hd = self.initial_default_hd


            if goal_reached>.9 and t>0.1:  #if reached destination (skip initialization time), stop simulation
                self.stop_sim=True
            # Live plot
            if self.plot_config:
                if np.mod(step_counter,100)==0: #plot every other 100 steps (100ms):
                    # create plot
                    plt.ion()
                    plt.show()
                    fig = plt.figure(0, figsize=(vid_size[0] / my_dpi, vid_size[1] / my_dpi),
                                     dpi=my_dpi)  # set fig size to match video
                    # iterate over all the probes, and plot last few samples
                    pos = self.sim.data[self.probes['position']][-300:]
                    hd= self.sim.data[self.probes['HeadDirection']][-1]
                    hd_rad=hd*self.max_angle #range -pi,pi
                    bvc=self.sim.data[self.probes['LIDAR_ARRAY']][-1]
                    bvc=bvc+self.BVC_rec_field
                    plt.cla()
                    #recent trajectory
                    plt.plot(pos[:,0],pos[:,1],color='blue')
                    #fish pos
                    x,y=pos[-1,0],pos[-1,1]
                    fish = plt.Circle((x,y), 0.03, color='k')
                    plt.gca().add_patch(fish)
                    #corals
                    for r in range(0,self.n_corals):
                        Coral = plt.Circle((self.corals_pos[r]), self.coral_size, color='g')
                        plt.gca().add_patch(Coral)
                    Goal = plt.Circle((self.Goal_pos), 0.02, color='r')
                    plt.gca().add_patch(Goal)
                    #BVC:
                    ang_arr=np.arange(0,360,self.ang_bin_size_deg)
                    for ang in range(0,len(ang_arr)):
                        #define angle and end-point for bvc lines
                        th=hd_rad + np.deg2rad(ang_arr[ang])
                        x2=float(x+bvc[ang]*np.cos(th))
                        y2=float(y+bvc[ang]*np.sin(th))
                        #define linestyle and color
                        if ang == 0:
                            LS = '-'
                        else:
                            LS = '--'
                        if bvc[ang]>self.BVC_rec_field-.05:
                            LC='k'
                        else:
                            LC='r'
                        xlin=np.array([x,x2])
                        ylin=np.array([y,y2])
                        plt.plot(xlin,ylin,color=LC,linestyle=LS,linewidth=0.5)

                    plt.xlim(-self.arena_size[0]/2,self.arena_size[0]/2)
                    plt.ylim(-self.arena_size[0] / 2, self.arena_size[0] / 2)
                    plt.xticks([])
                    plt.yticks([])
                    # plt.title('BVC_Nav_by_Memory  t = ' + str(round(t,2)) + ' [s]')
                    plt.title('NavMem  Mem_thres = ' + str(round(self.max_sim, 2)) + ' [s]')
                    mypause(0.002)
                    if self.vid_flag:
                        # with vidOut.saving(fig, '{}_video.mp4'.format(self.mat_name), dpi=my_dpi):
                        # write frame into viedo
                        canvas = FigureCanvas(plt.gcf())
                        canvas.draw()
                        mat = np.array(canvas.renderer._renderer)
                        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
                        vidOut.write(mat)
                            # vidOut.grab_frame()



        # End of simulation plot and data dump
        if t >= self.StopT:
            print('simulation finished after ' + str(self.StopT) + ' sec')
            mat_dict = {'t': self.sim.trange()}
            # iterate over all the probes, and plot with different style
            for key in self.probes.keys():
                # plt.figure()
                arr = self.sim.data[self.probes[key]]
                # plots = plt.plot(arr)
                # for i, p in enumerate(plots):
                #     p.set_label(str(i))
                #     p.set_color(comb_styles[i%len(comb_styles)][0])
                #     p.set_linestyle(comb_styles[i%len(comb_styles)][1])
                # plt.title(key)
                # plt.legend()
                mat_dict[key] = arr
            # plt.show()

            # Save probe data
            savemat('{}_data.mat'.format(self.mat_name), mat_dict)  # matlab format
            with open('{}_data.pkl'.format(self.pic_name), 'wb') as f:  # pickle format
                pickle.dump(mat_dict, f)
            # save input params data
            savemat('{}_input_params.mat'.format(self.mat_name), self.input_params_dict)  # matlab format
            if self.vid_flag:
                cv2.destroyAllWindows()
                vidOut.release()