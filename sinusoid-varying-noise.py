import numpy as np
import gym
from ten import *
from main.dataset import *
from main.dataViz import *
import yaml


cfg_filename = 'configs/sinusoid-config.yml'
with open(cfg_filename,'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

noise1 = 0.1
noise2 = 0.3
noise3 = 0.5
# guarante to generate the same random values
dataset1 = SinusoidDataset(config, noise_var=noise1, rng=np.random.RandomState(1234)) 
dataset2 = SinusoidDataset(config, noise_var=noise2, rng=np.random.RandomState(1234))
dataset3 = SinusoidDataset(config, noise_var=noise3, rng=np.random.RandomState(1234))


agent1 = ALPaCA(config)
agent1.sigma_eps = noise1
agent1.train(dataset1,5000)



agent2 = ALPaCA(config)
agent2.sigma_eps = noise2
agent2.train(dataset2,5000)


agent3 = ALPaCA(config)
agent3.sigma_eps = noise3
agent3.train(dataset3,5000)

N_test = 50
test_horz = 30
dataset1 = SinusoidDataset(config, noise_var=noise1, rng=np.random.RandomState(4321))
dataset2 = SinusoidDataset(config, noise_var=noise2, rng=np.random.RandomState(4321))
dataset3 = SinusoidDataset(config, noise_var=noise3, rng=np.random.RandomState(4321))
X_test1, Y_test1, freq_list_test, amp_list_test, phase_list_test = dataset1.sample(N_test, test_horz, return_lists=True)
X_test2, Y_test2 = dataset2.sample(N_test, test_horz, return_lists=False)
X_test3, Y_test3 = dataset3.sample(N_test, test_horz, return_lists=False)

ind = 2
sample_size_list = [0,1,2,3,4,5,6,7,8,9,10]
plt.figure(figsize=(9,len(sample_size_list)*1))
for i,num_pts in enumerate(sample_size_list):
    X_update1 = X_test1[ind:(ind+1),:num_pts,:]
    Y_update1 = Y_test1[ind:(ind+1),:num_pts,:]
    
    X_update2 = X_test2[ind:(ind+1),:num_pts,:]
    Y_update2 = Y_test2[ind:(ind+1),:num_pts,:]
    
    X_update3 = X_test3[ind:(ind+1),:num_pts,:]
    Y_update3 = Y_test3[ind:(ind+1),:num_pts,:]
    
    title=None
    legend=False
    if i == 0:
        legend=True
        title=True
        
    ax1 = plt.subplot(len(sample_size_list),3,3*i+1)
    gen_sin_fig(agent1, X_update1, Y_update1, freq_list_test[ind], phase_list_test[ind], amp_list_test[ind], label=None)
    if i == 0:
        plt.title('ALPaCA Sigma_epsilon = 0.1')
    if i < len(sample_size_list) - 1:
        plt.setp(ax1.get_xticklabels(), visible=False)
    
    ax2 = plt.subplot(len(sample_size_list),3,3*i+2, sharey=ax1)
    gen_sin_fig(agent2, X_update2, Y_update2, freq_list_test[ind], phase_list_test[ind], amp_list_test[ind], label=None)
    plt.setp(ax2.get_yticklabels(), visible=False)
    if i == 0:
        plt.title('ALPaCA Sigma_epsilon = 0.3')
    if i < len(sample_size_list) - 1:
        plt.setp(ax2.get_xticklabels(), visible=False)
    
    
    ax3 = plt.subplot(len(sample_size_list),3,3*i+3, sharey=ax1)
    gen_sin_fig(agent3, X_update3, Y_update3, freq_list_test[ind], phase_list_test[ind], amp_list_test[ind], label=None)
    plt.setp(ax3.get_yticklabels(), visible=False)
    if i == 0:
        plt.title('ALPaCA Sigma_epsilon = 0.5')
    if i < len(sample_size_list) - 1:
        plt.setp(ax3.get_xticklabels(), visible=False)

plt.tight_layout(w_pad=0.0,h_pad=0.2)
plt.savefig('figures/sinusoid_varying_noise.pdf')
plt.show()