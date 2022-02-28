import numpy as np

class Environment(object):
    
    def __init__(self,horizon,k):
        self.horizon = horizon
        self.k = k
        self.h = 0
        self.done = True
        
    def start(self):
        self.h = 0
        self.done = False
        return None

    def step(self,action):
        assert self.done is False, "Environment must be reset"
        self.h += 1 
        if self.h == self.horizon:
            self.done = True
            
        return (0, self.done)
        
    

class GaussianEnv(Environment):

    def __init__(self, horizon, k, mean = None, cov=None,seed=None, prior_dist=None, reward_dist=None):
        self.horizon = horizon
        self.k = k
        self.Ns = np.zeros(self.k)
        self.mean = mean
        self.prior_dist = prior_dist
        self.reward_dist = reward_dist
        if mean is None:
            self.mean = np.random.normal(0,1,self.k)
        assert np.shape(self.mean) == (self.k,), "Mean is the wrong shape"
        
        self.cov = cov
        if self.cov is None:
            self.cov = np.matrix(np.eye(self.k))
        assert np.shape(self.cov) == (self.k,self.k), "Mean is the wrong shape"

        self.param = None
        self.h = 0
        self.done=True
        if seed == None:
            self.seed = np.random.randint(0,100)
        else:
            self.seed = seed
        np.random.seed(self.seed)

    def start(self):
        self.h = 0
        self.Ns = np.zeros(self.k)
        self.done=False
        if self.prior_dist == 'AdaTScode':
            self.param = self.mean + np.diagonal(self.cov) * np.random.randn(self.k)
        else:
            self.param = np.random.multivariate_normal(self.mean,self.cov)

        self.idx_best = np.argmax(self.param)

    def step(self, action):
        assert self.done is False, "Environment must be reset"
        self.h += 1
        self.Ns[action] += 1
        if self.h == self.horizon:
            self.done = True

        if self.reward_dist=='GausAdaTScode':
            _sigma = 1.0
            r = self.param + _sigma * np.random.randn(self.k)  #TODO: sigma assumed always 1
            r = r[action]
        else:
            r = np.random.normal(self.param[action], 1)
            
        return (r, self.done)
        
if __name__=='__main__':
    env = GaussianEnv(300, 3)
    env.start()
    rewards = [0,0,0]
    counts = [0,0,0]
    for t in range(300):
        a = np.random.choice(3)
        (r,done) = env.step(a)
        rewards[a] += r
        counts[a] += 1

    print(env.param)
    print([rewards[i]/counts[i] for i in range(3)])
