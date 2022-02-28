import numpy as np
import BML_2.bandits as bandits

class MetaThompson(object):
    
    def __init__(self,k,train=None,fit_cov=True,params={}):
        self.k = k
        self.params = params
        self.fit_cov=fit_cov

        self.mean = np.zeros(self.k)
        self.running_mean = np.zeros(self.k)

        if 'mean' in params.keys():
            self.mean = params['mean']
            self.running_mean = params['mean']
        self.cov = np.eye(self.k)
        self.running_cov = np.matrix(np.zeros((self.k,self.k)))
        if 'cov' in params.keys():
            self.cov = params['cov']
            self.running_cov = params['cov']

        self.t = 0
        if train is None:
            self.train = -1
        else:
            self.train = train

        self.num_random=2
        if self.fit_cov is False:
            self.num_random=1
        

    def select(self):
        if self.h < self.num_random and (self.train < 0 or self.t <= self.train):
            action = np.random.choice(self.k)
        else:
            action = self.base_learner.select()
        return(action)

    def update(self,action,reward):
        self.base_learner.update(action,reward)
        
        if self.h < self.num_random and (self.train < 0 or self.t <= self.train):
            vec = np.zeros(self.k)
            vec[action] = self.k*reward  ### importance weighting
            self.tmp_vec += vec

            if self.h == self.num_random-1:
                self.running_mean += self.tmp_vec/self.num_random
                self.mean = self.running_mean/(self.t+1)
                if self.fit_cov:
                    ### Covariance estimation, I think correct now
                    self.running_cov += np.matrix(self.tmp_vec).T*np.matrix(self.tmp_vec)/self.num_random
                    self.cov = self.get_cov(self.running_mean,self.running_cov,self.t)
        self.h += 1

    def get_cov(self,mean,c,t):
        k = mean.shape[0]
        m = mean/t
        cov = c/t
        M = cov - np.matrix(m).T*np.matrix(m)
        M -= k*np.diag(np.diag(np.matrix(m).T*np.matrix(m)))
        M -= k*np.eye(k)
        for a in range(k):
            M[a,a] = M[a,a]/(k+1)

        ## make sure matrix is psd (might be too slow now...)
        eigval,eigvec = np.linalg.eig(M)
        Q = np.matrix(eigvec)
        d = np.matrix(np.diag(np.maximum(eigval,0)))
        return(Q*d*Q.T)

    def reset(self):
        self.t += 1
        self.base_learner = bandits.Thompson(self.k,self.mean,self.cov)
        self.tmp_vec = np.zeros(self.k)
        self.h=0


if __name__=='__main__':
    import envs
    k = 2
    H = 10

    env = envs.GaussianEnv(H,k)
    alg = MetaThompson(k)
    
    for r in range(1000):
        env.start()
        alg.reset()
        for t in range(H):
            a = alg.select()
            (r,done) = env.step(a)
            alg.update(a,r)
        
    
    print(env.cov)
    print(alg.cov)
    print(env.mean)
    print(alg.mean)
