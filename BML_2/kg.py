import numpy as np
from scipy.stats import norm
import copy

def kg_f(z):
    return z * norm.cdf(z) + norm.pdf(z)

def kg_sigma(precision):
    return 1. / np.sqrt(precision*(precision+1.))

def kg_zeta(mean, precision):
    max_index = np.argmax(mean)
    max_val = mean[max_index]
    mean[max_index] = np.nan
    second_max_val = np.nanmax(mean)
    mean[max_index] = max_val
    return -np.abs([m - second_max_val if m == max_val else m - max_val for m in mean]) / kg_sigma(precision)

def kg_select(mean, precision):
    return np.argmax(kg_sigma(precision) * kg_f(kg_zeta(mean, precision)))

class KGPolicy(object):
    
    def __init__(self,k,mean,cov):
        self.k = k
        self.mean = mean

        self.prec = 1. / np.diag(cov)
        self.rew = np.zeros(self.k)
        self.counts = np.zeros(self.k)

    def select(self,greedy=False):
        pc = self.prec + self.counts
        pm = (self.mean*self.prec + self.rew)*pc
        return kg_select(pm, pc)

    def update(self,action,reward):
        self.rew[action] += reward
        self.counts[action] += 1

    def reset(self):
        self.rew = np.zeros(self.k)
        self.counts = np.zeros(self.k)
        
if __name__=='__main__':
    import envs
    k = 5
    H = 50
    env = envs.GaussianEnv(H,k,mean=np.array(np.zeros(k)))
    alg = KGPolicy(k,env.mean,env.cov)
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

    
