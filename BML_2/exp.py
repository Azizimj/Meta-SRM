import numpy as np
import BML_2.envs as envs
import BML_2.meta_learner_general as meta_learner_general
import BML_2.bandits as bandits

import multiprocessing as mp

BAI = True  # indicating that it is BAI
BAI = False

np.random.seed(110)

def test_():
    print(np.random.randn())

def make_spiked_instance(K):
    mean = np.zeros(K)
    mean[3] = 0.5
    cov = 0.1*np.eye(K)
    return(mean,cov)

def make_near_block(n,k):
    M = np.matrix(np.zeros((n,n)))
    for i in range(int(n/k)):
        M[k*i:k*(i+1), k*i:k*(i+1)] = 0.9
    for i in range(n):
        M[i,i] = 1
    return (M)

def make_block_instance(K):
    mean = np.zeros(K)
    mean[0] = 0.5
    mean[int(K/2)] = 0.1
    cov = make_near_block(K,int(K/2))
    return (mean,cov)

INSTANCES = {
    'spiked': make_spiked_instance,
    'block': make_block_instance
}

def experiment(alg, env, repsT):
    
    ep_rewards = []
    ep_rounds_rewards = []  # episode rounds rewards
    ep_sreg = []
    ep_rounds_sreg = []  # episode rounds simple regs
    cov_errs = []
    mean_errs = []
    for t in range(repsT):
        env.start()
        alg.reset()
        ep_rew = 0
        while not env.done:
            action = alg.select()
            (r,done) = env.step(action)
            alg.update(action,r)
            ep_rew += r
            ep_rounds_rewards.append(r)
            # BAI
            idx_mf = np.argmax(env.Ns)
            _sreg = env.param[env.idx_best] - env.param[idx_mf]
            assert _sreg>=0, 'negative simple regret!'
            ep_rounds_sreg.append(_sreg)

        ep_rewards.append(ep_rew)

        # BAI
        idx_mf = np.argmax(env.Ns)
        ep_sreg.append(env.param[env.idx_best] - env.param[idx_mf])

        if 'cov' in alg.__dict__.keys():
            cov_errs.append(np.linalg.norm(alg.cov - env.cov))
        if 'mean' in alg.__dict__.keys():
            mean_errs.append(np.linalg.norm(alg.mean - env.mean))

    return (np.cumsum(ep_rewards) / np.arange(1, repsT + 1), mean_errs,
            cov_errs, np.cumsum(ep_sreg) / np.arange(1, repsT + 1),
            ep_rounds_rewards, ep_rounds_sreg)

def explore_then_commit_experiment(alg,env,train_steps,reps):
    ep_rewards = []
    cov_errs = []
    mean_errs = []
    for t in range(reps):
        env.start()
        alg.reset()
        ep_rew = 0
        greedy=False
        while not env.done:
            if env.h == train_steps:
                greedy = True
                ep_rew = 0
            action = alg.select(greedy=greedy)
            (r,done) = env.step(action)
            ep_rew += r
            if not greedy:
                alg.update(action,r)
        ep_rewards.append(ep_rew/(env.horizon-train_steps))
        if 'cov' in alg.__dict__.keys():
            cov_errs.append(np.linalg.norm(alg.cov - env.cov))
        if 'mean' in alg.__dict__.keys():
            mean_errs.append(np.linalg.norm(alg.mean - env.mean))
    return (ep_rewards, mean_errs, cov_errs)

def one_iter_standard(Args, mean, cov, iter_cnt=0, prior_dist=None, reward_dist=None):
    ###
    # env = envs.GaussianEnv(Args.H, Args.K, mean, cov)
    # if Args.alg == 'mts-no-cov':
    #     Alg = meta_learner_general.MetaThompsonGeneral(Args.K, train=Args.train, num_random=1, fit_cov=False)
    # elif Args.alg == 'mts':
    #     Alg = meta_learner_general.MetaThompsonGeneral(Args.K, train=Args.train, num_random=Args.num_random, fit_cov=True)
    # elif Args.alg == 'oracle':
    #     Alg = bandits.Thompson(Args.K, env.mean, env.cov)
    # else:
    #     Alg = bandits.Thompson(Args.K, np.zeros(Args.K), np.eye(Args.K))
    #
    # (vec, me, ce, svec) = experiment(Alg, env, Args.T)

    env = envs.GaussianEnv(Args['H'], Args['K'], mean, cov, prior_dist=prior_dist, reward_dist=reward_dist)
    if Args['alg'] == 'mts-no-cov':
        Alg = meta_learner_general.MetaThompsonGeneral(Args['K'], train=Args['train'], num_random=1, fit_cov=False)
    elif Args['alg'] == 'mts':
        Alg = meta_learner_general.MetaThompsonGeneral(Args['K'], train=Args['train'], num_random=Args['num_random'],
                                                       fit_cov=True)
    elif Args['alg'] == 'oracle':
        Alg = bandits.Thompson(Args['K'], env.mean, env.cov)
    else:
        Alg = bandits.Thompson(Args['K'], np.zeros(Args['K']), np.eye(Args['K']))

    (rew_vec, sreg_vec, mean_, cov_est_, ep_rounds_rewards, ep_rounds_sreg) = experiment(Alg, env, Args['T'])

    # return (iter_cnt, vec, svec, me, ce)
    return (iter_cnt, rew_vec, sreg_vec, mean_, cov_est_, Args, ep_rounds_rewards, ep_rounds_sreg)

def collect_result(res):
    (i, vec, svec, me, ce, Args,  _, _) = res
    global rewards, simple_regs, mean_est, cov_est
    rewards.append(vec)
    simple_regs.append(svec)
    mean_est.append(me)
    cov_est.append(ce)

    print("Iter: %d, test_reward: %0.3f" % (i, np.mean(vec[int(Args['T']/2):])), flush=True)

    return


if __name__=='__main__':
    import envs, bandits, meta_learner_general, mpc, kg
    import argparse, sys, os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', action='store', default='standard', type=str)
    parser.add_argument('--instance', action='store', default='block', type=str)
    parser.add_argument('--T', action='store', default=400, type=int)
    parser.add_argument('--iters', action='store', default=5, type=int)
    parser.add_argument('--H', action='store', default=10, type=int)
    parser.add_argument('--K', action='store', default=5, type=int)
    parser.add_argument('--alg', action='store', choices=['mts-no-cov', 'mts', 'oracle', 'misspecified', 'mpc', 'kg'],
                        default='mts')
    parser.add_argument('--train', action='store', default=0, type=int)
    parser.add_argument('--num_random', action='store', default=2, type=int)
    parser.add_argument('--train_len', action='store', default=2, type=int)
    parser.add_argument('--outdir_pref', action='store', default='./', type=str)

    Args = parser.parse_args(sys.argv[1:])
    print(Args, flush=True)

    Args_d = vars(Args)

    ## outdir = "/mnt/scratch/....." for philly jobs
    outdir = "%sresults/%s_%s_T=%d_H=%d_K=%d-i=%d/" % (Args.outdir_pref, Args.type, Args.instance, Args.T, Args.H,
                                                       Args.K, Args.iters)
    num_cpu = min(8, mp.cpu_count())
    # num_cpu = mp.cpu_count()
    parr = 0  # mp off
    parr = 1  # mp on

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if Args.type == 'standard':

        (mean,cov) = INSTANCES[Args.instance](Args.K)

        rewards = []
        simple_regs = []
        mean_est = []
        cov_est = []
        if parr:
            pool = mp.Pool(num_cpu)
            # poolobjs = [pool.apply_async(one_iter_standard, args=[Args, mean, cov, i], callback=collect_result)
            #             for i in range(Args.iters)]
            poolobjs = [pool.apply_async(one_iter_standard, args=[Args_d, mean, cov, i], callback=collect_result)
                        for i in range(Args.iters)]
            pool.close()
            pool.join()
            # for f in poolobjs:
            #     print(f.get())  # print the errors
        else:
            for i in range(Args.iters):
                # env = envs.GaussianEnv(Args.H,Args.K,mean,cov)
                # if Args.alg == 'mts-no-cov':
                #     Alg = meta_learner_general.MetaThompsonGeneral(Args.K,train=Args.train,num_random=1,fit_cov=False)
                # elif Args.alg == 'mts':
                #     Alg = meta_learner_general.MetaThompsonGeneral(Args.K,train=Args.train,num_random=Args.num_random,fit_cov=True)
                # elif Args.alg == 'oracle':
                #     Alg = bandits.Thompson(Args.K, env.mean, env.cov)
                # else:
                #     Alg = bandits.Thompson(Args.K, np.zeros(Args.K), np.eye(Args.K))
                #
                # (vec,me,ce, svec) = experiment(Alg, env, Args.T)
                (_, vec, me, ce, svec, _, _) = one_iter_standard(Args, mean, cov, i)

                rewards.append(vec[::10])
                simple_regs.append(svec[::10])
                mean_est.append(me[::10])
                cov_est.append(ce[::10])
                print("Iter: %d, test_reward: %0.3f" % (i, np.mean(vec[int(Args.T/2):])), flush=True)

        np.savetxt(outdir+"%s_%d_%d_rewards.out" % (Args.alg, Args.train, Args.num_random), rewards)
        np.savetxt(outdir+"%s_%d_%d_simple_regs.out" % (Args.alg, Args.train, Args.num_random), simple_regs)
        np.savetxt(outdir+"%s_%d_%d_means.out" % (Args.alg, Args.train, Args.num_random), mean_est)
        np.savetxt(outdir+"%s_%d_%d_covs.out" % (Args.alg, Args.train, Args.num_random), cov_est)

    elif Args.type == 'etc':
        mean = np.zeros(Args.K)
        cov = 1*np.eye(Args.K)
 
        rewards = []
        mean_est = []
        cov_est = []
        
        for i in range(Args.iters):
            env = envs.GaussianEnv(Args.H,Args.K,mean,cov)
            if Args.alg == 'mpc':
                Alg = mpc.MPCPolicy(Args.K,env.mean,env.cov)
            elif Args.alg == 'kg':
                Alg = kg.KGPolicy(Args.K,env.mean,env.cov)
            elif Args.alg == 'oracle':
                Alg = bandits.Thompson(Args.K,env.mean,env.cov)
            (vec,me,ce) = explore_then_commit_experiment(Alg,env,Args.train_len,Args.T)
            rewards.append(vec)
            mean_est.append(me)
            cov_est.append(ce)
            print("Iter: %d, commit_reward: %0.3f" % (i, np.mean(vec)), flush=True)

        np.savetxt(outdir+"%s_%d_rewards.out" % (Args.alg, Args.train_len), rewards)
        np.savetxt(outdir+"%s_%d_means.out" % (Args.alg, Args.train_len), mean_est)
        np.savetxt(outdir+"%s_%d_covs.out" % (Args.alg, Args.train_len), cov_est)

