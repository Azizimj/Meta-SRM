import numpy as np
import bandits, envs, exp, mpc
import meta_learner_general
import matplotlib.pyplot as plt

T = 10000
H=10
H=20
K=6
R=100
iters = 5
step = 10
out_type = "rewards"
# out_type = "simple_regs"

results_dir='./results/standard_block_T=%d_H=%d_K=%d-i=%d/' % (T,H,K, iters)

train_lengths = range(200,5001,200) # [200,400,600,800,1000,1200,1400,1600,1800,2000]
train_lengths = range(500, 5001, 500)

def plot_box():
    print("Preprocessing Data")
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    # T = 10000
    # H=10
    # K=6
    # R=100
    # results_dir='./results/standard_block_T=%d_H=%d_K=%d/' % (T,H,K)

    # train_lengths = range(200,5001,200) # [200,400,600,800,1000,1200,1400,1600,1800,2000]

    # process oracle
    oracle_rew = np.loadtxt(results_dir+f'oracle_0_0_{out_type}.out')
    print('Oracle: %0.3f' % (np.mean(oracle_rew)))
    # process mts-10
    mts_10_rew = np.loadtxt(results_dir+f'mts_{train_lengths[-1]}_10_{out_type}.out')
    print('MTS-10: %0.3f' % (np.mean(mts_10_rew)))
    # process misspecified
    misspecified_rew = np.loadtxt(results_dir+f'misspecified_0_0_{out_type}.out')
    print('Misspec: %0.3f' % (np.mean(misspecified_rew)))
    # process no-cov
    no_cov_rew = np.loadtxt(results_dir+f'mts-no-cov_{train_lengths[-1]}_0_{out_type}.out')
    print('No-cov: %0.3f' % (np.mean(no_cov_rew)))

    vecs = (oracle_rew, mts_10_rew, misspecified_rew, no_cov_rew)
    arr = []
    for v in vecs:
        arr.append([(T*v[i,-1] - T*v[i,int(T/20)]/2)*2/T for i in range(v.shape[0])])
    alg = ['Oracle', 'Misspecified', 'No-Cov', 'TS-2', 'TS-5', 'TS-10']
          
#     f = plt.figure(figsize=(5,4),dpi=100)
    bplot = plt.boxplot(arr, patch_artist=True,widths=0.5,positions=[1,1.6,2.2,2.8])
    # fill with colors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    for patch,color in zip(bplot['medians'], colors):
        patch.set_color(color)
    patch.set_linewidth(2)
        
    ax = plt.gca()
    ax.set_xlim([0.7,3.1])
    ax.set_xticks([1,1.6,2.2,2.8])
    ax.set_xticklabels(['OracleTS', 'MetaTS: full', 'MisTS', 'MetaTS: no-cov'])
    plt.ylabel('Per-episode reward after meta-training')
    plt.title(f'Gaussian MAB, A={K}, H={H}')

def plot_mab_curve():
    print("Preprocessing Data")
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    symbols = ['o', 's', 'v', 'D']
    # T = 10000
    # H=10
    # K=6
    # R=100
    # results_dir='./results/standard_block_T=%d_H=%d_K=%d/' % (T,H,K)

    # train_lengths = range(200,5001,200) # [200,400,600,800,1000,1200,1400,1600,1800,2000]

    def get_pointwise_best(n,num_explore):
        means = np.zeros((len(train_lengths), n))
        stds = np.zeros((len(train_lengths), n))
        for i in range(len(train_lengths)):
            try:
                tmp = np.loadtxt(results_dir+f'mts_{train_lengths[i]}_{num_explore}_{out_type}.out')
                means[i,:] = np.mean(tmp,axis=0)
                stds[i,:] = np.std(tmp,axis=0)
            except:
                continue        
        idx = np.argmax(means, axis=0)
        return(np.array([means[idx[i], i] for i in range(n)]), np.array([stds[idx[i], i] for i in range(n)]))

    def get_pointwise_best_no_cov(n):
        means = np.zeros((len(train_lengths), n))
        stds = np.zeros((len(train_lengths), n))
        for i in range(len(train_lengths)):
            try:
                tmp = np.loadtxt(results_dir+f'mts-no-cov_{train_lengths[i]}_0_{out_type}.out')
                means[i,:] = np.mean(tmp,axis=0)
                stds[i,:] = np.std(tmp,axis=0)
            except:
                continue        
        idx = np.argmax(means, axis=0)
        return(np.array([means[idx[i], i] for i in range(n)]), np.array([stds[idx[i], i] for i in range(n)]))

    # x = np.arange(10,T+1,10)
    x = np.arange(1,T+1)

    vecs = []
    oracle_rew = np.loadtxt(results_dir+f'oracle_0_0_{out_type}.out')#[:, ::10]
    vecs.append((np.mean(oracle_rew,axis=0), np.std(oracle_rew,axis=0)))
    vecs.append(get_pointwise_best(len(x), 10))
    vecs.append(get_pointwise_best_no_cov(len(x)))
    misspecified_rew = np.loadtxt(results_dir+f'misspecified_0_0_{out_type}.out')#[:, ::10]
    vecs.append((np.mean(misspecified_rew,axis=0), np.std(misspecified_rew,axis=0)))
    # x=x[20:]
    x=x[min(train_lengths):][::step]
    i = 0
    ls = []
    for (m,s) in vecs:
        m = m[min(train_lengths):][::step]
        s = s[min(train_lengths):][::step]
        ls.append(plt.plot(x,m,color=colors[i],marker=symbols[i],markevery=100))
#        plt.plot(x[100::100], m[100::100],color=colors[i], marker=symbols[i])
        plt.fill_between(x,m-2/np.sqrt(R)*s, m+2/np.sqrt(R)*s,alpha=0.4, color=colors[i])
        i += 1
        plt.legend(['OracleTS', 'MetaTS: full', 'MetaTS: no-cov', 'MisTS'])
#         plt.legend(ls,['OracleTS', 'MetaTS: full', 'MetaTS: no-cov', 'MisTS'])
    plt.ylabel('Cumulative average reward')
    plt.xlabel('Episodes')
    plt.xlim([min(train_lengths),T])
    plt.title(f'Gaussian MAB, A={K}, H={H}')


if __name__=='__main__':
    
    plot_box()
    plt.subplots_adjust(left=0.03, right=0.99, bottom=0.3)
    plt.savefig(f'./{results_dir}/mab_{out_type}_test_error.pdf', format='pdf', dpi=100, bbox_inches='tight')
    plt.close()
    
    plot_mab_curve()
    plt.subplots_adjust(left=0.03, right=0.99, bottom=0.3)
    plt.savefig(f'./{results_dir}/mab_{out_type}_learning_curve.pdf', format='pdf', dpi=100, bbox_inches='tight')
    plt.close()

