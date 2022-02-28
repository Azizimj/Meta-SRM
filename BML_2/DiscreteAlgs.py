import numpy as np
from DiscreteEnv import DISTS
import copy

def argmax_random(vec):
    return np.random.choice(np.flatnonzero(vec == np.max(vec)))

def discrete_update(dist,act,rew):
    for i in range(len(dist)):
        if DISTS[i][act] != rew:
            dist[i] = 0
    dist = dist/np.sum(dist)
    return(dist)
    

class DiscreteTS():
    
    def __init__(self,k,prior):
        self.k=k
        self.prior=prior
        self.post=copy.deepcopy(prior)

    def select(self):
        sample = np.random.choice(range(len(self.post)), p=self.post)
#         a = np.argmax(DISTS[sample])
        a = argmax_random(DISTS[sample])
        return(a)

    def update(self,action,reward):
        self.post = discrete_update(copy.deepcopy(self.post),action,reward)
        
    def reset(self):
        self.post=copy.deepcopy(self.prior)

class DiscreteKG():

    def __init__(self,k,prior,ntrials=10):
        self.k=k
        self.prior=prior
        self.post=copy.deepcopy(prior)
        self.ntrials=ntrials
        self.dists = np.matrix(np.zeros((len(DISTS), self.k)))
        for i in range(len(DISTS)):
            self.dists[i,:] = np.array(DISTS[i])

    def select(self):
        scores = []
        tiebreaker = []
        for a in range(self.k):
            v = 0
            r = 0
            for t in range(self.ntrials):
                vp, rp = self.get_lookahead_value(a)
                v += vp; r += rp
            scores.append(v/self.ntrials)
            tiebreaker.append( (v+r)/self.ntrials)
        if len(np.flatnonzero(scores == np.max(scores))) == 1:
            return np.argmax(scores)
        else:
            inds = np.flatnonzero(scores == np.max(scores))
            tiebreaker = np.array(tiebreaker)
            ind = argmax_random(tiebreaker[inds])
            return inds[ind]

    def get_lookahead_value(self,action):
        sample_idx = np.random.choice(range(len(self.post)), p=self.post)
        ra = DISTS[sample_idx][action]
        
        ## Recompute posterior after the fake sample
        tmp_post = discrete_update(copy.deepcopy(self.post), action, ra)
        
#         ## compute posterior mean
#         mean = np.matrix(tmp_post)*self.dists
#         return (np.max(mean), ra)
        ## Resample from posterior
#         print("MC update")
#         print(tmp_post)
        sample_idx = np.random.choice(range(len(self.post)), p=tmp_post,size=self.ntrials)
        emp_post = np.zeros(len(tmp_post))
        for i in sample_idx:
            emp_post[i] += 1
        emp_post = emp_post/self.ntrials
        mean = np.matrix(emp_post)*self.dists
#         print(DISTS[sample_idx[0]])
#         print(DISTS[sample_idx[1]])
#         print(mean, flush=True)
        return (np.max(mean), ra)

    def update(self,action,reward):
        self.post = discrete_update(copy.deepcopy(self.post), action, reward)

    def reset(self):
        self.post=copy.deepcopy(self.prior)

class DiscreteExplore():
    def __init__(self,k):
        self.k = k
        self.prior = np.ones(len(DISTS))/len(DISTS)
        self.post = copy.deepcopy(self.prior)
        self.collapsed = len(np.flatnonzero(self.post)) == 1
        
        self.counts = np.ones(len(DISTS))
        self.total = np.sum(self.counts)

    def select(self):
        if self.collapsed:
            # print("Collapsed posterior", flush=True)
            idx = np.argmax(self.post)
            a = np.argmax(DISTS[idx])
        else:
            a = np.random.choice(self.k)
            # print("Selecting random action: %d" % (a), flush=True)
        return(a)

    def update(self,action,reward):
        self.post = discrete_update(copy.deepcopy(self.post), action, reward)
        self.collapsed = len(np.flatnonzero(self.post)) == 1

    def reset(self):
        if self.collapsed:
            idx = np.argmax(self.post)
            self.counts[idx] += 1
            self.total += 1
        self.post = copy.deepcopy(self.prior)    
        self.collapsed = len(np.flatnonzero(self.post)) == 1

class ExploreKG():
    def __init__(self,k,train_len):
        self.k=k
        self.train_len = train_len
        self.base = DiscreteExplore(self.k)
        self.episodes = 0

    def select(self):
        return(self.base.select())

    def update(self,action,reward):
        self.base.update(action,reward)

    def reset(self):
        self.base.reset()
        self.episodes += 1
        if self.episodes == self.train_len:
            self.prior = self.base.counts/self.base.total
            self.base = DiscreteKG(self.k,self.prior)

class ExploreTS():
    def __init__(self,k,train_len):
        self.k=k
        self.train_len = train_len
        self.base = DiscreteExplore(self.k)
        self.episodes = 0

    def select(self):
        return(self.base.select())

    def update(self,action,reward):
        self.base.update(action,reward)

    def reset(self):
        self.base.reset()
        self.episodes += 1
        if self.episodes == self.train_len:
            self.prior = self.base.counts/self.base.total
            self.base = DiscreteTS(self.k,self.prior)

if __name__=='__main__':
    import DiscreteEnv, copy
    env = DiscreteEnv.DiscreteEnv(5)
    for i in range(1):
        # Alg = DiscreteTS(env.k, copy.deepcopy(env.param))
        Alg = DiscreteKG(env.k, copy.deepcopy(env.param))
        env.start()
        print("Starting")
        for j in range(5):
            a = Alg.select()
            (r,done) = env.step(a)
            Alg.update(a,r)
            print("idx: %d, act: %d, rew: %0.1f" % (env.idx, a, r))
            print(Alg.post)

    arr = np.ones(int(len(DISTS)/2))/(len(DISTS)/2)
    arr = np.append(arr,np.zeros(int(len(DISTS)/2)))
    print(arr)
    env.param = arr
    Alg = DiscreteExplore(env.k)
    for i in range(10000):
        env.start()
        Alg.reset()
        for j in range(5):
            a = Alg.select()
            (r,done) = env.step(a)
            Alg.update(a,r)
    Alg.reset()
    print(Alg.counts/Alg.total)
