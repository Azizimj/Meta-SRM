import numpy as np
import bandits, envs, exp, mpc
import meta_learners, meta_learner_general
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


T = 400
H=10
K=20
train_lens=range(25,201,25)
print(train_lens)
R=100

def get_pointwise_best(n,alg):
    means = np.zeros((len(train_lens), n))
    stds = np.zeros((len(train_lens), n))
    for i in range(len(train_lens)):
        try:
            tmp = np.loadtxt(results_dir+'%s_%d_rewards.out' % (alg,train_lens[i]))
            v = np.cumsum(tmp,axis=1)
            means[i,:] = np.mean(v,axis=0)
            stds[i,:] = np.std(v,axis=0)
        except:
            print("Skipping %d" % (train_lens[i]), flush=True)
            continue
    idx = np.argmax(means,axis=0)
    return(np.array([means[idx[i], i] for i in range(n)]), np.array([stds[idx[i], i] for i in range(n)]))

results_dir='./results/discrete_T=%d_H=%d/' % (T,H)


oracle_ts_rew = np.loadtxt(results_dir+"oracle-ts_0_rewards.out")
oracle_ts_act = np.loadtxt(results_dir+"oracle-ts_0_actions.out")
v = np.cumsum(oracle_ts_rew,axis=1)
oracle_ts_mean = np.mean(v, axis=0)
oracle_ts_std = np.std(v, axis=0)

mis_ts_rew = np.loadtxt(results_dir+"mis-ts_0_rewards.out")
mis_ts_act = np.loadtxt(results_dir+"mis-ts_0_actions.out")
v = np.cumsum(mis_ts_rew,axis=1)
mis_ts_mean = np.mean(v, axis=0)
mis_ts_std = np.std(v, axis=0)

# mts_rew = np.loadtxt(results_dir+"mts_%d_rewards.out" % (train_len))
mts_act = np.loadtxt(results_dir+"mts_%d_actions.out" % (100))
(mts_mean,mts_std) = get_pointwise_best(len(mis_ts_mean),'mts')

oracle_kg_rew = np.loadtxt(results_dir+"oracle-kg_0_rewards.out")
oracle_kg_act = np.loadtxt(results_dir+"oracle-kg_0_actions.out")
v = np.cumsum(oracle_kg_rew,axis=1)
oracle_kg_mean = np.mean(v, axis=0)
oracle_kg_std = np.std(v, axis=0)

mis_kg_rew = np.loadtxt(results_dir+"mis-kg_0_rewards.out")
mis_kg_act = np.loadtxt(results_dir+"mis-kg_0_actions.out")
v = np.cumsum(mis_kg_rew,axis=1)
mis_kg_mean = np.mean(v, axis=0)
mis_kg_std = np.std(v, axis=0)

# mkg_rew = np.loadtxt(results_dir+"mkg_%d_rewards.out" % (train_len))
mkg_act = np.loadtxt(results_dir+"mkg_%d_actions.out" % (100))
(mkg_mean,mkg_std) = get_pointwise_best(len(mis_ts_mean),'mkg')

# rewards = [oracle_ts_rew, mis_ts_rew, mts_rew, oracle_kg_rew, mis_kg_rew, mkg_rew]
rewards = [
    (oracle_kg_mean,oracle_kg_std),
    (mkg_mean,mkg_std),
    (mis_kg_mean,mis_kg_std),
    (oracle_ts_mean,oracle_ts_std), 
    (mts_mean,mts_std), 
    (mis_ts_mean,mis_ts_std),
    ]
actions = [
    oracle_kg_act, 
    mkg_act,
    mis_kg_act,
    oracle_ts_act, 
    mts_act,
    mis_ts_act
    ]
order = [
    'OracleKG', 
    'MetaKG',
    'MisKG', 
    'OracleTS', 
    'MetaTS',
    'MisTS'
]
colors= ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink']
symbols = ['o', 's', 'D', 'v', 'P', 'X']

import copy, DiscreteEnv


x = np.arange(1,T+1)[25:]
for i in range(len(rewards)):
    m = rewards[i][0]
    s = rewards[i][1]
# for (m,s) in rewards:
#     arr = np.cumsum(v,axis=1)/np.arange(1,T+1)
#     m = np.mean(arr,axis=0)[1:]
#     std = np.std(arr,axis=0)[1:]
    m = m[25:]/x
    s = s[25:]/x
    line = plt.plot(x, m,color=colors[i],marker=symbols[i],markevery=50)
#    line = plt.plot(x[25::50],m[25::50],color=colors[i],marker=symbols[i])
    _ = plt.fill_between(x,m-np.sqrt(2./R)*s, m+np.sqrt(2./R)*s,alpha=0.4)
#     _ = colors.append(line[0].get_color())
plt.legend(order)
plt.ylabel('Cumulative average reward')
plt.xlabel('Episodes')
plt.xlim([25,400])
plt.title('Algorithm performance')
plt.subplots_adjust(left=0.03, right=0.99, bottom=0.3)
plt.savefig('kg_learning_curve.pdf', format='pdf', dpi=100, bbox_inches='tight')
plt.close()

def process_act_mat(M):
    print(M.shape)
    return ([np.count_nonzero(M == x) for x in range(K)])
hists = [process_act_mat(M) for M in actions]
M = np.zeros((len(hists), K))
for i in range(len(hists)):
    M[i,:] = np.array(hists[i])/(T*R)

# cmap = mpl.colors.ListedColormap(['white', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue'])
cmap = mpl.colors.ListedColormap(['white', 'lightsteelblue', 'cornflowerblue', 'mediumblue', 'mediumpurple', 'indigo'])
cmap.set_over('white')
cmap.set_under('indigo')
bounds = [0, 0.01, 0.05, 0.1, 0.2, 0.8, 1.0]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
print(M)

im = plt.imshow(M, cmap=cmap,norm=norm)
ax = plt.gca()
plt.title('Distribution of the first arm pulled by each algorithm')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)


ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.set_xticks(range(K))
lst = ['A%d' % (int(x)) for x in range(1,K+1)]
# lst = [r'\textbf{%d}' % (d) if d % 5 == 0 else r'%d' % (d) for d in lst]
labels = ax.set_xticklabels(lst,rotation=90)
for i in range(4):
    labels[5*i].set_weight('bold')
plt.subplots_adjust(left=0.03, right=0.99, bottom=0.3)
plt.savefig('kg_actions.pdf', format='pdf', dpi=100, bbox_inches='tight')
plt.close()

### Second version
norm = mpl.colors.BoundaryNorm(bounds,ncolors=256)
im = plt.imshow(M, cmap='Blues',norm=norm)
ax = plt.gca()
plt.title('Distribution of the first arm pulled by each algorithm')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)


ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.set_xticks(range(K))
lst = ['A%d' % (int(x)) for x in range(1,K+1)]
# lst = [r'\textbf{%d}' % (d) if d % 5 == 0 else r'%d' % (d) for d in lst]
labels = ax.set_xticklabels(lst,rotation=90)
for i in range(4):
    labels[5*i].set_weight('bold')
plt.subplots_adjust(left=0.03, right=0.99, bottom=0.3)
plt.savefig('kg_actions_blues.pdf', format='pdf', dpi=100, bbox_inches='tight')
plt.close()


import copy, DiscreteEnv
env = DiscreteEnv.DiscreteEnv(10)
v = np.zeros((len(env.param),1))
v[:,0] = env.param


# f = plt.figure(figsize=(10,8))
img = plt.imshow(copy.deepcopy(DiscreteEnv.DISTS), cmap='Blues', interpolation=None)
ax = plt.gca()
ax.set_xticks(range(20))
ax.set_xticklabels(['A%d' % (i) for i in range(1,21)], rotation=90)
ax.get_xticklabels()[0].set_fontweight('bold')
ax.get_xticklabels()[5].set_fontweight('bold')
ax.get_xticklabels()[10].set_fontweight('bold')
ax.get_xticklabels()[15].set_fontweight('bold')

print(env.param)
ax.set_yticks(range(16))
labels = ['9/40','9/40','9/40','9/40',
          '1/120','1/120','1/120','1/120',
          '1/120','1/120','1/120','1/120',
          '1/120','1/120','1/120','1/120',
          '1/120','1/120','1/120','1/120']
# _ = ax.set_yticklabels(['T%d (%s)' % (x,labels[x-1]) for x in range(1,17)])
_ = ax.set_yticklabels(['T%d (%0.2f)' % (x,v[x-1,0]) for x in range(1,17)])
# ax.get_yticklabels()[0].set_text('T1 (9/40)')
# ax.get_yticklabels()[1].set_text('T2 (9/40)')
# ax.get_yticklabels()[2].set_text('T3 (9/40)')
# ax.get_yticklabels()[3].set_text('T4 (9/40)')

# ax2 = ax.twinx()
# ax2.yaxis.tick_right()
# # _ = plt.imshow(copy.deepcopy(DiscreteEnv.DISTS), cmap='Blues')
# ax2.set_yticks(range(16))
# ax2.set_yticklabels(['p=%0.1f' % v[-i,0] for i in range(16)])
ax.set_xlabel('Arms')
ax.set_ylabel('Tasks with prior probabilities')
# ax2.set_ylabel('Prior probabilities')
plt.title('Arm rewards in each task')

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(img,cax=cax)

# cax = plt.axes([0.78, 0.14, 0.1, 0.72])
# v = np.zeros((len(env.param),1))
# v[:,0] = env.param
# plt.imshow(v, cmap='Blues')
# ax3 = plt.gca()
# # plt.colorbar(img)
# ax3.set_yticks([])
# ax3.set_xticks([])
# ax3.yaxis.tick_right()
# ax3.set_ylabel('Prior probabilities')
# ax3.yaxis.set_label_position('right')

plt.savefig('kg_instance.pdf', format='pdf', dpi=100, bbox_inches='tight')
