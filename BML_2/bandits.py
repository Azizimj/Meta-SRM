import numpy as np

class UCB(object):
    
    def __init__(self,k,conf):
        self.conf = conf
        self.k = k
        self.rew = [0 for i in range(self.k)]
        self.counts = [0 for i in range(self.k)]
        

    def select(self):
        if np.min(self.counts) == 0:
            action = np.argmin(self.counts)
            return action
        
        scores = [self.rew[i]/self.counts[i] + self.conf*np.sqrt(1.0/self.counts[i]) for i in range(self.k)]
        action = np.argmax(scores)
        return action

    def update(self,action,reward):
        self.rew[action] += reward
        self.counts[action] += 1

    def reset(self):
        self.rew = [0 for i in range(self.k)]
        self.counts = [0 for i in range(self.k)]
        

class Thompson(object):
    
    def __init__(self,k,mean,cov):
        self.k = k
        self.mean = np.matrix(mean)
        self.cov = cov
        self.prec = np.linalg.pinv(self.cov)
        
        self.rew = np.matrix(np.zeros((1,self.k)))
        self.counts = [0 for i in range(self.k)]

    def select(self,greedy=False):
        ### Compute posterior (I think correct?)
        
        posterior_cov = np.linalg.pinv(self.prec + np.diag(self.counts))
        posterior_mean = (self.mean*self.prec + self.rew)*posterior_cov

        sample = np.random.multivariate_normal(np.array(posterior_mean)[0,:],posterior_cov)
        if greedy:
            sample = posterior_mean
        action = np.argmax(sample)
        return action

    def update(self,action,reward):
        self.rew[0, action] += reward
        self.counts[action] += 1

    def reset(self):
        self.rew = np.matrix(np.zeros((1,self.k)))
        self.counts = [0 for i in range(self.k)]
        

if __name__=='__main__':
    import envs
    k = 10
    H = 1000
    env = envs.GaussianEnv(H,k)
    alg = Thompson(k,env.mean,env.cov)
    env.start()
    for t in range(H):
        a = alg.select()
        (r,done) = env.step(a)
        alg.update(a,r)
    print(env.param)
    print(alg.rew)
    print(alg.counts)
