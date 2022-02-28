import numpy as np
import BML_2.bandits as bandits
import matplotlib.pyplot as plt


class MetaThompsonGeneral(object):
    
    def __init__(self,k,train=None,num_random = None, fit_cov=True,params={}):
        self.k = k
        self.params = params
        self.fit_cov=fit_cov

        self.mean = np.zeros(self.k)
        self.running_mean = np.zeros(self.k)

        self.explore_action = 0

        if 'mean' in params.keys():
            self.mean = params['mean']
            self.running_mean = params['mean']

        self.cov = np.eye(self.k)
        self.second_moment = np.eye(self.k)
        self.cov_unprojected = np.eye(self.k)

        self.running_cov = np.matrix(np.zeros((self.k,self.k)))
        if 'cov' in params.keys():
            self.cov = params['cov']
            self.running_cov = params['cov']
            self.second_moment = params['cov']

        self.t = 0
        if train is None:
            self.train = -1
        else:
            self.train = train

        
        if num_random is None:
            self.num_random = 1e9
        else:
            self.num_random = num_random

        self.explore_stages = 0
        self.explore_rounds = 0
        self.cov_log = []

    def explore_decision(self):
        if self.h < self.num_random and (self.train < 0 or self.t <= self.train):
            return  True
        else:
            return  False

    def select(self):

        self.explore_decision()
        if self.explore_decision() is True:
            action = np.random.choice(self.k)
        else:
            action = self.base_learner.select()
            self.isExplore = False
#             print('not  random?',self.h,self.t,self.train)
        return (action)

    def update(self,action,reward):
        self.explore_decision()
        self.base_learner.update(action,reward)
    
        if self.explore_decision() is True:
            self.explore_stages += 1
            self.ep_running_mean[action] += reward
            self.ep_running_sq[action] += reward*reward

            if self.h == 0:
                self.explore_rounds += 1
        self.h += 1

    def update_mean_and_cov(self):
        k = self.k + 0.0
        n = self.explore_stages + 0.0
        t =  self.explore_rounds + 0.0

        in_ep_mean = (k/n) * self.ep_running_mean
        old_mean = (t-1) * self.mean/t
        self.mean = old_mean + (in_ep_mean/t)

        if self.fit_cov is False:
            return 

        if n < 2:
            raise Exception('updating covariance with too few samples')

        # constructing unbiased estimator of mu_t mu_t^T

        M = np.matrix(in_ep_mean).T *  np.matrix(in_ep_mean)
        # Exp[M[a][a]] | round t
        #                = (k/n)^2 * (n/k * sigma_reward^2)
        #                   + (k/n)^2  (n/k mu_t(a)^2 + n(n-1)/k^2 mu_t(a)^2)
        #                =  k/n * sigma_reward^2 + mu_t(a)^2 ( (k+n-1)/n)
        # 
        # Exp[ M[a][b]]  = (k/n)^2 * (n*(n-1))/k^2 mu_t(a) mu_t(b)
        #                = (n-1)/n * mu_t(a)mu_t(b)
        #renormalize off diagonals
        cov_t = n/(n-1) * M
        #if t == 100:
        #    print(M)
        #    print(cov_t)
            
        for a in range(self.k):
            # correct for noise
            cov_t[a,a] = (k/n * self.ep_running_sq[a]) - 1

          
            
        self.second_moment = ((t-1) * self.second_moment/t) + (cov_t/t)
        self.cov_unprojected = self.second_moment  - np.matrix(self.mean).T * np.matrix(self.mean)
        
        self.cov = MetaThompsonGeneral.psd_proj(self.cov_unprojected)

    def psd_proj(M):
        eigval, eigvec = np.linalg.eig(M)
        Q = np.matrix(eigvec)
        d = np.matrix(np.diag(np.maximum(eigval,0)))
        return(Q*d*Q.T)

    def reset(self):
        if self.explore_rounds >= 1 and self.explore_stages > 0:
            self.update_mean_and_cov()
            self.cov_log.append(self.second_moment[0,1])


        self.t += 1
        self.base_learner = bandits.Thompson(self.k,self.mean,self.cov)
        self.h=0
        self.ep_running_mean = np.zeros(self.k)
        self.ep_running_sq = np.zeros(self.k)

        self.explore_stages = 0


if __name__=='__main__':
    import envs
    k = 2
    H = 10
    n = 1
    T = 1000

    env = envs.GaussianEnv(H,k)
    #alg = MetaThompsonGeneral(k,train = 100)
    #print(env.mean,env.cov)
    #print('erp')
    cov_run = np.zeros((k,k))
    for i in range(n):
        print(i)
        alg = MetaThompsonGeneral(k,train = T)
        for r in range(T):
            env.start()
            alg.reset()
            for t in range(H):
                a = alg.select()
                (r,done) = env.step(a)
                alg.update(a,r)
                #print(alg.cov)
        cov_run += alg.cov_unprojected
        cov_log = alg.cov_log
    
    print(env.cov)
    print(alg.cov)
    print(alg.cov_unprojected)

    print(env.mean)
    print(alg.mean)
