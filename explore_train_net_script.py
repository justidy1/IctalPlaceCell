
#%matplotlib widget
import nest
import numpy as np
import matplotlib.pyplot as plt

import place_field_gen as pfg

L_maze = 120
maze = pfg.LineMaze(360,L=L_maze)
t,x,y = pfg.gen_traj(maze,500,0.9)
r = pfg.firing_place_cell_linear(t,x,-30,L_maze,7)

plt.figure()
plt.plot(t,r)
plt.twinx()
plt.plot(t,x,'.r')
plt.show()

# Create CA3 and CA1 populations of parrot neurons
nest.ResetKernel()
nest.local_num_threads = 1
delta_t = 0.1
nest.SetKernelStatus({"resolution": delta_t})

# chop t and x into delta_t ms bins
t_res = np.arange(1000*t[0],1000*t[-1],delta_t)
x_res = np.interp(t_res,1000*t,x)

N_pyr = 1250
CA3_pyr = nest.Create("parrot_neuron", N_pyr)
CA1_pyr = nest.Create("parrot_neuron", N_pyr)
nest.SetStatus(CA3_pyr, {"tau_minus": 62.5})
nest.SetStatus(CA1_pyr, {"tau_minus": 40})

# make the clamped rates to assign to cells
N_generators = int(np.floor(N_pyr*0.1))
place_cell_spikes_CA3 = nest.Create("inhomogeneous_poisson_generator", N_generators)
place_cell_spikes_CA1 = nest.Create("inhomogeneous_poisson_generator", N_generators)
# for each generator, draw a random place field on the maze and assign the rate generated
# along a trajectory to the generator
place_fields_CA3 = []
pf = np.sort(np.random.uniform(-L_maze/2,L_maze/2,N_generators))
for i in range(N_generators):
    r = pfg.firing_place_cell_linear(t_res,x_res,pf[i],600,7)
    place_fields_CA3.append((pf[i],r))
    nest.SetStatus(place_cell_spikes_CA3[i], {"rate_times": t_res+delta_t, "rate_values": r})

place_fields_CA1 = []
pf = np.sort(np.random.uniform(-L_maze/2,L_maze/2,N_generators))
for i in range(N_generators):
    r = pfg.firing_place_cell_linear(t_res,x_res,pf[i],600,7)
    place_fields_CA1.append((pf[i],r))
    nest.SetStatus(place_cell_spikes_CA1[i], {"rate_times": t_res+delta_t, "rate_values": r})

# generate the rest of the generators with random rates
silent_cell_spikes = nest.Create("poisson_generator")
nest.SetStatus(silent_cell_spikes, {"rate": 0.1})

# connect the generators to the CA3 and CA1 populations
nest.Connect(place_cell_spikes_CA3, CA3_pyr[0:N_generators], syn_spec={'weight': 1.0}, conn_spec={'rule': 'one_to_one'})
nest.Connect(silent_cell_spikes, CA3_pyr[N_generators:], syn_spec={'weight': 1.0}, conn_spec={'rule': 'all_to_all'})
nest.Connect(place_cell_spikes_CA1, CA1_pyr[0:N_generators], syn_spec={'weight': 1.0}, conn_spec={'rule': 'one_to_one'})
nest.Connect(silent_cell_spikes, CA1_pyr[N_generators:], syn_spec={'weight': 1.0}, conn_spec={'rule': 'all_to_all'})

# define a STDP synapse
weight_recorder = nest.Create("weight_recorder")
nest.CopyModel("stdp_synapse", "CA3_to_CA3",{"alpha": -1.0, "lambda": 0.08, "tau_plus": 62.5, "weight": 0.3,"mu_plus":0,"mu_minus":0, "Wmax": 20.0,"weight_recorder": weight_recorder})
nest.CopyModel("stdp_synapse", "CA3_to_CA1",{"alpha": 0.4, "lambda": 0.04, "tau_plus": 20.0, "weight": 0.7,"mu_plus":0,"mu_minus":0, "Wmax": 20.0,"weight_recorder": weight_recorder})

# connect the CA3 and CA1 populations with the STDP synapse
nest.Connect(CA3_pyr, CA3_pyr, syn_spec={'synapse_model':"CA3_to_CA3",'receptor_type':1}, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.1})
nest.Connect(CA3_pyr, CA1_pyr, syn_spec={'synapse_model':"CA3_to_CA1",'receptor_type':1}, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.1})

# monitor the spiking activity of the CA3 and CA1 populations
spike_detector = nest.Create("spike_recorder")
nest.Connect(CA3_pyr, spike_detector)
nest.Connect(CA1_pyr, spike_detector)

print("Starting EXPLORATION simulation")
nest.Prepare()
nest.Run(6000)
print("Starting CHUNK 2 of EXPLORATION simulation")
nest.Run(6000)
nest.Cleanup()

from nest import raster_plot

raster_plot.from_device(spike_detector,hist=True, hist_binwidth=10.0)

conn_CA3_to_CA3 = nest.GetConnections(CA3_pyr,CA3_pyr).get(['source','target','weight'])
conn_CA3_to_CA1 = nest.GetConnections(CA3_pyr,CA1_pyr).get(['source','target','weight'])

# making a new newtork
nest.ResetKernel()
nest.SetKernelStatus({"resolution": 0.1})

# Create CA3 and CA1 populations of adaptive exponential integrate-and-fire neurons
N_pyr = 1250
pyr_spec = {'C_m':180,'g_L':4.3,'E_L':-75,'Delta_T':4.23,'V_th':-24,'V_peak':-3.25,'V_reset':-29.7,'t_ref':5.9,'tau_w':84.93,'a':-0.27,'b':206.84,'E_rev':[0,-70],'tau_rise':[1,0.3],'tau_decay':[9.5,3]}
CA3_pyr_aeif = nest.Create("aeif_cond_beta_multisynapse", N_pyr, pyr_spec)
CA1_pyr_aeif = nest.Create("aeif_cond_beta_multisynapse", N_pyr, pyr_spec)
# make interneurons in CA3 and CA1
N_int = 250
int_spec = {'C_m':118,'g_L':7.5,'E_L':-74,'Delta_T':4.6,'V_th':-57.7,'V_peak':-34.78,'V_reset':-65,'t_ref':1,'tau_w':178.58,'a':3.05,'b':0.91,'E_rev':[0,-70],'tau_rise':[1,0.3],'tau_decay':[9.5,3]}
CA3_int_aeif = nest.Create("aeif_cond_beta_multisynapse", N_int, int_spec)
CA1_int_aeif = nest.Create("aeif_cond_beta_multisynapse", N_int, int_spec)

# connect the CA3 and CA1 populations with static synapses with the weights learned from the previous simulation
# the order of nodes is presevered since the neurons are created in the same order
nest.Connect(conn_CA3_to_CA3['source'], conn_CA3_to_CA3['target'], 'one_to_one', syn_spec={'synapse_model':'static_synapse','weight': conn_CA3_to_CA3['weight'], 'receptor_type':1})
nest.Connect(conn_CA3_to_CA1['source'], conn_CA3_to_CA1['target'], 'one_to_one', syn_spec={'synapse_model':'static_synapse','weight': conn_CA3_to_CA1['weight'], 'receptor_type':1})

# now we wire pyr to int with a probability of 0.1
nest.Connect(CA3_pyr_aeif, CA3_int_aeif, syn_spec={'weight': 0.85, 'receptor_type':1}, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.1})
nest.Connect(CA1_pyr_aeif, CA1_int_aeif, syn_spec={'weight': 0.85, 'receptor_type':1}, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.1})

# connect the interneurons with each other with a probability of 0.25
nest.Connect(CA3_int_aeif, CA3_int_aeif, syn_spec={'weight': 5, 'receptor_type':2}, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.25})
nest.Connect(CA1_int_aeif, CA1_int_aeif, syn_spec={'weight': 5, 'receptor_type':2}, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.25})

# and the interneurons to the pyramidal cells with a probability of 0.25
nest.Connect(CA3_int_aeif, CA3_pyr_aeif, syn_spec={'weight': 0.65, 'receptor_type':2}, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.25})
nest.Connect(CA1_int_aeif, CA1_pyr_aeif, syn_spec={'weight': 0.65, 'receptor_type':2}, conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.25})

# create a stim to drive all the CA3 pyramidal cells
stim = nest.Create("poisson_generator",1,{'rate': 15.0})
nest.Connect(stim, CA3_pyr_aeif, syn_spec={'weight': 20.0, 'receptor_type':1}, conn_spec={'rule': 'all_to_all'})

# monitor the spiking activity of the CA3 and CA1 populations
spike_detector = nest.Create("spike_recorder")
nest.Connect(CA3_pyr_aeif, spike_detector)
nest.Connect(CA1_pyr_aeif, spike_detector)
nest.Connect(CA3_int_aeif, spike_detector)
nest.Connect(CA1_int_aeif, spike_detector)
print("Starting OFFLINE simulation")
nest.Simulate(1000)

raster_plot.from_device(spike_detector,hist=False)


