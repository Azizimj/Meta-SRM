import numpy as np
import copy

class MPCPolicy(object):
    
    def __init__(self,k,mean,cov,lookahead=1,ntrials=100):
        self.k =k
        self.mean = np.matrix(mean)
        self.cov = cov
        self.lookahead=lookahead
        self.ntrials=ntrials

        self.prec = np.linalg.pinv(self.cov)
        self.rew = np.matrix(np.zeros((1,self.k)))
        self.counts = [0 for i in range(self.k)]


    def select(self,greedy=False):
        pc = np.linalg.pinv(self.prec + np.diag(self.counts))
        pm = (self.mean*self.prec + self.rew)*pc

        if greedy:
            action = np.argmax(pm)
            return action

#         print("---- Selection ----",flush=True)
#         print(pc)
#         print(pm)
        scores = []
        for a in range(self.k):
            v = 0
            for t in range(self.ntrials):
                v += self.get_lookahead_value(a,pm,pc)
            scores.append(v/self.ntrials)
#         print(scores)
        return np.argmax(scores)

    def get_lookahead_value(self,action,pm,pc):
        ### First compute posterior mean and covariance (these are passed in to amortize)
        ### Sample rewards from posterior and select arm a
        sample_mu = np.random.multivariate_normal(np.array(pm)[0,:],pc)
        ra = np.random.normal(sample_mu[action], 1)

        ### Update posterior mean and take the max
        ### These are the simulated statistics
        counts = copy.deepcopy(self.counts)
        counts[action] += 1
        rew = copy.deepcopy(self.rew)
        rew[0,action] += ra

        posterior_cov = np.linalg.pinv(self.prec + np.diag(counts))
        posterior_mean = (self.mean*self.prec + rew)*posterior_cov
        ### Two options here
        ### Knowledge gradient
        return np.max(posterior_mean)
        #### MPC
        # return ra + np.max(posterior_mean)


    def update(self,action,reward):
        self.rew[0,action] += reward
        self.counts[action] += 1

    def reset(self):
        self.rew = np.matrix(np.zeros((1,self.k)))
        self.counts = [0 for i in range(self.k)]
        

if __name__=='__main__':
    import envs
    k = 5
    H = 50
    env = envs.GaussianEnv(H,k,mean=np.array(np.zeros(k)))
    alg = MPCPolicy(k,env.mean,env.cov)
    for r in range(10):
        env.start()
        alg.reset()
        for t in range(H):
            a = alg.select()
            (r,done) = env.step(a)
            alg.update(a,r)
        # print("Environment mean: " + str(env.mean))
        print("Environment param: " + str(env.param))
        # print("Alg rewards: " + str(alg.rew))
        print("Alg counts: " + str(alg.counts))

    
