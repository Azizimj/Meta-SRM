import numpy as np
import copy

import multiprocessing as mp

BAI = True  # indicating that it is BAI
BAI = False

class LinearBandit():

    def __init__(self, d, K, H, prior=None, noise=False, seed=None):
        self.d = d
        self.K = K
        self.H = H
        self.noise = noise
        self.seed = seed

        self.Ns = np.zeros(self.K)  # number of plays

        if seed is not None:
            np.random.seed(574)

        self.ep = 0

        self.prior = [np.zeros(d), np.eye(d)]
        if prior is not None:
            self.prior = prior

    def start(self):
        self.ep += 1
        self.h = 0
        self.Ns = np.zeros(self.K)
        self.done = False
        self.curr_x = None
        self.features = None
        self.curr_r = None

        self.weights = np.random.multivariate_normal(self.prior[0], self.prior[1])

        return (self.get_new_context())

    def get_new_features_means(self):
        self.features = np.matrix(np.random.normal(0, 1, [self.K, self.d]))
        # self.features[0,:] = 0.05*self.features[0,:] + np.matrix(self.weights.T)
        self.features = np.diag(1. / np.sqrt(np.diag(self.features * self.features.T))) * self.features  # ???
        self.curr_means = np.array((self.features * np.matrix(self.weights).T).T)[0]

        self.idx_best = np.argmax(self.curr_means)

        return self.features, self.curr_means  # .038311

    def get_new_context(self):
        ## Generate random feature matrix and normalize.
        if self.seed is not None:
            np.random.seed((self.h+self.ep*self.H+17)*(self.seed+1) + 37)

        if not BAI or self.h==0: # under BAI features should not change
            self.features, self.curr_means = self.get_new_features_means()

        if self.noise and type(self.noise) == float:
            self.noise_term = np.random.normal(0,self.noise)
            self.curr_r = np.array(self.curr_means+self.noise_term)
        elif self.noise:
            self.noise_term = np.random.normal(0, 0.1)
            self.curr_r = np.array(self.curr_means+self.noise_term)
        else:
            self.curr_r = np.array(self.curr_means)

        return (self.features)

    def get_best_reward(self):
        idx = np.argsort(self.curr_means)
        return np.sum(self.curr_r[idx[-1]])

    def step(self, a):
        assert self.done is False, "Environment must be reset"
        self.h += 1
        self.Ns[a] += 1

        if self.h == self.H:
            self.done = True
        r = self.curr_r[a]  # cum reward

        # if BAI:# simple reward
        #     idx_most_freq = np.argmax(self.Ns)
        #     sr = self.curr_means[self.idx_best] - self.curr_means[idx_most_freq]
        
        return (r, self.get_new_context(), self.done)
        

class LinearTS():
    def __init__(self, d, K, prior=None):
        self.d = d
        self.K = K
        self.prior = [np.zeros(d), np.eye(d)]
        if prior is not None:
            self.prior = prior
        self.post = copy.deepcopy(self.prior)
        self.data_norm = np.matrix(np.zeros((self.d,self.d)))
        self.b_vec = np.matrix(np.zeros((self.d,1)))

    def select(self,features):
        sample = np.matrix(np.random.multivariate_normal(self.post[0], self.post[1])).T
        scores = np.array((features*sample).T)[0]
        idx = np.argmax(scores)
        return idx

    def update(self,features,action,reward):
        x = np.matrix(features[action,:])
        self.post[1] += x.T*x     ## posterior covariance = \Lambda_0 + \sum_t x_t x_t^\top 
        self.data_norm += x.T*x   ## data_norm = \sum_t x_t x_t^\top
        self.b_vec += x.T*reward    ## b_vec = \sum_t x_t r_t
        ols_param = np.linalg.pinv(self.data_norm)*self.b_vec  ## (\sum_t x_t x_t^\top)^\dagger \sum_t x_t y_t
        self.post[0] = np.linalg.pinv(self.post[1])*(self.data_norm*ols_param + self.prior[1]*np.matrix(self.prior[0]).T)
        self.post[0] = np.array(self.post[0].T)[0]

    def reset(self):
        self.post=copy.deepcopy(self.prior)
        self.data_norm = np.matrix(np.zeros((self.d,self.d)))
        self.b_vec = np.matrix(np.zeros((self.d,1)))


class LinearExplore():
    def __init__(self,d,K):
        self.d = d
        self.K = K
        ## Meta learner state
        self.prior = [np.zeros(d), np.eye(d)]
        self.estimated_prior = [np.zeros(d), np.eye(d)]
        self.t = 1

        ## Per episode state
        self.post = copy.deepcopy(self.prior)
        self.data_norm = np.matrix(np.zeros((self.d,self.d)))
        self.b_vec = np.matrix(np.zeros((self.d,1)))

    def select(self,features):
        a = np.random.choice(self.K)
        return(a)

    def update(self,features,action,reward):
        x = np.matrix(features[action,:])
        self.post[1] += x.T*x
        self.data_norm += x.T*x
        self.b_vec += x.T*reward
        ols_param = np.linalg.pinv(self.data_norm)*self.b_vec  ## (\sum_t x_t x_t^\top)^\dagger \sum_t x_t y_t
        self.post[0] = np.linalg.pinv(self.post[1])*(self.data_norm*ols_param + self.prior[1]*np.matrix(self.prior[0]).T)  # ???
        self.post[0] = np.array(self.post[0].T)[0]

    def reset(self):

        ## Update meta learner state
        self.t+=1
        ols_pred = np.linalg.pinv(self.data_norm)*self.b_vec
        
        ## First moment estimator is just ols_pred
        self.estimated_prior[0] = ((self.t-1)*self.estimated_prior[0] + np.array(ols_pred.T)[0])/self.t

        ## Second moment estimator needs to debias
        ## ols_pred = true_beta + N(0, data_norm^{-1}) 
        second_moment = ols_pred*ols_pred.T - np.linalg.pinv(self.data_norm)
        self.estimated_prior[1] = ((self.t-1)*self.estimated_prior[1] + second_moment)/self.t
        
        ## Reset per-ep state
        self.post = copy.deepcopy(self.prior)
        self.data_norm = np.matrix(np.zeros((self.d,self.d)))
        self.b_vec = np.matrix(np.zeros((self.d,1)))


class LinearMetaTS():
    def __init__(self,d,K,train_len):
        self.d = d
        self.K = K
        self.train_len = train_len
        self.prior = [np.zeros(d),np.eye(d)]
        self.base = LinearExplore(self.d,self.K)
        self.episodes = 0


        self.estimated_prior = self.base.estimated_prior

    def select(self,features):
        return(self.base.select(features))

    def update(self,features,action,reward):
        self.base.update(features,action,reward)

    def reset(self):
        self.base.reset()
        self.episodes += 1
        if self.episodes <= self.train_len:
            self.estimated_prior = self.base.estimated_prior            
        if self.episodes == self.train_len:
            mean = self.base.estimated_prior[0]
            cov = self.base.estimated_prior[1] - np.matrix(mean).T*np.matrix(mean)
            self.prior = [mean, cov]
            self.base = LinearTS(self.d,self.K,prior=self.prior)


def experiment(alg, env, repsT):
    ep_rewards = []  # episode rewards
    ep_sreg = []
    mean_errs = []
    cov_errs = []
    for t in range(repsT):
        ctx = env.start()
        alg.reset()
        ep_rew = 0
        while not env.done:
            action = alg.select(ctx)
            (r, new_ctx, done) = env.step(action)
            alg.update(ctx, action, r)
            ep_rew += r
            ctx = new_ctx
        ep_rewards.append(ep_rew)

        # BAI
        idx_mf = np.argmax(env.Ns)
        ep_sreg.append(env.curr_means[env.idx_best] - env.curr_means[idx_mf])

        if 'estimated_prior' in alg.__dict__.keys():
            mean = alg.estimated_prior[0]
            mean_errs.append(np.linalg.norm(mean - env.prior[0]))
            cov = alg.estimated_prior[1] - np.matrix(mean).T * np.matrix(mean)
            cov_errs.append(np.linalg.norm(cov - env.prior[1]))

    return (ep_rewards, ep_sreg, mean_errs, cov_errs)


def one_iter(Args, mean, cov, iter_cnt=0):

    env = LinearBandit(Args.d, Args.K, Args.H, prior=[mean, cov], noise=1.0, seed=59 + iter_cnt)
    if Args.alg == 'mts':
        Alg = LinearMetaTS(Args.d, Args.K, Args.train)
    elif Args.alg == 'oracle':
        Alg = LinearTS(Args.d, Args.K, prior=[mean, cov])
    else:
        Alg = LinearTS(Args.d, Args.K, prior=None)

    (vec, svec, me, ce) = experiment(Alg, env, Args.T)

    return (iter_cnt, vec, svec, me, ce)


def collect_result(res):
    (i, vec, svec, me, ce) = res
    global rewards, simple_regs, mean_est, cov_est
    rewards.append(vec)
    simple_regs.append(svec)
    mean_est.append(me)
    cov_est.append(ce)

    print("Iter: %d, test_reward: %0.3f" % (i, np.mean(vec[int(Args.T/2):])), flush=True)

    return


if __name__=='__main__':
    import argparse, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store', default=400, type=int)
    parser.add_argument('--iters', action='store', default=4, type=int)
    parser.add_argument('--H', action='store', default=10, type=int)
    parser.add_argument('--d', action='store', default=10, type=int)
    parser.add_argument('--K', action='store', default=5, type=int)
    parser.add_argument('--alg', action='store', choices=['mts', 'oracle', 'misspecified'], default='oracle')
    parser.add_argument('--train', action='store', default=0, type=int)
    parser.add_argument('--outdir_pref', action='store', default='./', type=str)

    Args = parser.parse_args(sys.argv[1:])
    print(Args, flush=True)

    num_cpu = min(8, mp.cpu_count())
    # num_cpu = mp.cpu_count()
    parr = 0  # mp off
    parr = 1  # mp on

    ## outdir = "/mnt/scratch/....." for philly jobs
    outdir = "%sresults/%s_T=%d_d=%d_H=%d_K=%d-i=%d/" % (Args.outdir_pref, 'linear' if not BAI else 'linearBAI',
                                                    Args.T, Args.d, Args.H, Args.K, Args.iters)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    from exp import INSTANCES

#     (mean,cov) = INSTANCES['block'](Args.d)

    mean = np.ones(Args.d)
    (_,cov) = INSTANCES['block'](Args.d)
    cov = 0.1*cov

    rewards = []
    simple_regs = []
    mean_est = []
    cov_est = []

    if parr:
        pool = mp.Pool(num_cpu)
        poolobjs = [pool.apply_async(one_iter, args=[Args, mean, cov, i], callback=collect_result)
                    for i in range(Args.iters)]
        pool.close()
        pool.join()
        # for f in poolobjs:
        #     print(f.get())  # print the errors
    else:
        for i in range(Args.iters):

            # env = LinearBandit(Args.d,Args.K,Args.H,prior=[mean,cov],noise=1.0,seed=59+i)
            # if Args.alg == 'mts':
            #     Alg = LinearMetaTS(Args.d,Args.K,Args.train)
            # elif Args.alg == 'oracle':
            #     Alg = LinearTS(Args.d,Args.K,prior=[mean,cov])
            # else:
            #     Alg = LinearTS(Args.d,Args.K,prior=None)
            #
            # (vec, svec, me,ce) = experiment(Alg,env,Args.T)

            (_, vec, svec, me, ce) = one_iter(i)

            rewards.append(vec)
            simple_regs.append(svec)
            mean_est.append(me)
            cov_est.append(ce)
            print("Iter: %d, test_reward: %0.3f" % (i, np.mean(vec[int(Args.T/2):])), flush=True)

    np.savetxt(outdir+"%s_%d_rewards.out" % (Args.alg,Args.train), rewards)
    np.savetxt(outdir+"%s_%d_simple_regs.out" % (Args.alg,Args.train), simple_regs)
    np.savetxt(outdir+"%s_%d_means.out" % (Args.alg,Args.train), mean_est)
    np.savetxt(outdir+"%s_%d_covs.out" % (Args.alg,Args.train), cov_est)
