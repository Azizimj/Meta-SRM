import numpy as np
import bandits, envs, exp, mpc
import meta_learner_general
import matplotlib.pyplot as plt

out_type = "rewards"
out_type = "simple_regs"
train_lengths = [10,20,30,40,50,60,70,80,90,100]
train_lengths = [100,200,300,400,500,600,700,800,900,1000]
T = 2000
H=20
# H=100
K=6
d=6
R=100
mts_rew_mid = 600
i = 10
i = 100
results_dir='./results/linearBAI_T=%d_d=%d_H=%d_K=%d/' % (T,d,H,K)
results_dir='./results/linear_T=%d_d=%d_H=%d_K=%d-528/' % (T,d,H,K)
results_dir='./results/linear_T=%d_d=%d_H=%d_K=%d-i=%d/' % (T,d,H,K, i)




def plot_box():
    print("Preprocessing Data")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    # T = 200
    # H=20
    # d=6
    # K=6
    # R=100
    # results_dir='./results/linear_T=%d_d=%d_H=%d_K=%d/' % (T,d,H,K)


    # train_lengths = [10,20,30,40,50,60,70,80,90,100]

    # process oracle
    oracle_rew = np.loadtxt(results_dir+f'oracle_0_{out_type}.out')
    print('Oracle: %0.3f' % (np.mean(oracle_rew)))
    # process mts-10
    mts_rew = np.loadtxt(results_dir+f'mts_{mts_rew_mid}_{out_type}.out')
    print('MTS: %0.3f' % (np.mean(mts_rew)))
    # process misspecified
    misspecified_rew = np.loadtxt(results_dir+f'misspecified_0_{out_type}.out')
    print('Misspec: %0.3f' % (np.mean(misspecified_rew)))

    vecs = (oracle_rew, mts_rew, misspecified_rew)
    arr = []
    for v in vecs:
        arr.append([np.mean(v[i,int(T/2):]) for i in range(v.shape[0])])
#        arr.append([(T*v[i,-1] - T*v[i,int(T/2)]/2)*2/T for i in range(v.shape[0])])
    alg = ['Oracle', 'MTS', 'Misspecified']
          
#     f = plt.figure(figsize=(4,4),dpi=100)
    bplot = plt.boxplot(arr, patch_artist=True,widths=0.5,positions=[1,1.6,2.2])
#     bplot = plt.violinplot(arr,positions=[1,2,3],widths=0.75,showmeans=True)
    # fill with colors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    for patch,color in zip(bplot['medians'], colors):
        patch.set_color(color)
    patch.set_linewidth(2)
        
    ax = plt.gca()
    ax.set_xlim([0.7,2.5])
    ax.set_xticks([1,1.6,2.2])
#     ax.set_xticklabels(['Oracle', 'MTS', 'Misspecified'],rotation=30)
    ax.set_xticklabels(['OracleTS', 'MetaTS: full', 'MisTS'])
    plt.ylabel('Per-episode reward after meta-training')
    plt.title('Gaussian Linear CB, A=6, d=6, H=20')


print("Preprocessing Data")
# T = 200
# H=20
# K=6
# d=6
# R=100
# results_dir='./results/linearBAI_T=%d_d=%d_H=%d_K=%d/' % (T,d,H,K)

# train_lengths = [10,20,30,40,50,60,70,80,90,100]

# process oracle
oracle_rew = np.loadtxt(results_dir+f'oracle_0_{out_type}.out')
# process misspecified
misspecified_rew = np.loadtxt(results_dir+f'misspecified_0_{out_type}.out')


x = np.arange(1,T+1)
def get_pointwise_best(n):  # finds the best result between iterations
    means = np.zeros((len(train_lengths), n))
    stds = np.zeros((len(train_lengths), n))
    for i in range(len(train_lengths)):
        try:
            tmp = np.loadtxt(results_dir+f'mts_{train_lengths[i]}_{out_type}.out')
            tmp = np.cumsum(tmp,axis=1)/np.arange(1,n+1)
            # tmp = tmp/np.arange(1,n+1)
            means[i,:] = np.mean(tmp,axis=0)
            stds[i,:] = np.std(tmp,axis=0)
        except:
            continue
    idx = np.argmax(means, axis=0)
    return(np.array([means[idx[i], i] for i in range(n)]), np.array([stds[idx[i], i] for i in range(n)]))

oracle_rew = np.cumsum(oracle_rew,axis=1)/np.arange(1,len(x)+1)
misspecified_rew = np.cumsum(misspecified_rew,axis=1)/np.arange(1,len(x)+1)
vecs = [(np.mean(oracle_rew,axis=0), np.std(oracle_rew,axis=0))]  # mean and std over iterations
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
symbols = ['o', 's', 'D', 'v']

vecs.append(get_pointwise_best(len(x)))
vecs.append((np.mean(misspecified_rew,axis=0), np.std(misspecified_rew,axis=0)))

dis_idx = int(min(train_lengths))
x=x[dis_idx:]  # dismiss the results of the training (exploration)
for i in range(len(vecs)):
    m = vecs[i][0]  # mean reward of algo over iters
    s = vecs[i][1]  # std of rewards of algo over iters
    m = m[dis_idx:]
    s = s[dis_idx:]
    plt.plot(x,m,color=colors[i],marker=symbols[i],markevery=100)
#     plt.plot(x[100::100],m[100::100],color=colors[i],marker=symbols[i])
    plt.fill_between(x,m-2/np.sqrt(R)*s, m+2/np.sqrt(R)*s,alpha=0.4)
plt.legend(['OracleTS', 'MetaTS: full', 'MisTS'])
plt.ylabel('Cumulative average reward')
plt.xlabel('Episodes')
plt.xlim([dis_idx,T])
plt.title('Linear CB, d=6, A=6, H=20')
plt.subplots_adjust(left=0.03, right=0.99, bottom=0.3)
plt.savefig('%s/linear_%s_learning_curve.pdf' %(results_dir, out_type), format='pdf', dpi=100, bbox_inches='tight')
plt.close()

plot_box()
plt.subplots_adjust(left=0.03, right=0.99, bottom=0.3)
plt.savefig('%s/linear_%s_test_error.pdf' %(results_dir, out_type), format='pdf', dpi=100, bbox_inches='tight')
plt.close()

