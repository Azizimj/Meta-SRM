import numpy as np
import envs

DISTS = [[0.1, 1, 0, 0, 0, 0, 0, 0],
         [0.15, 0, 1, 0, 0, 0, 0, 0],
         [0.2, 0, 0, 1, 0, 0, 0, 0],
         [0.25, 0, 0, 0, 1, 0, 0, 0],
         [0.3, 0, 0, 0, 0, 1, 0, 0],
         [0.35, 0, 0, 0, 0, 0, 1, 0],
         [0.4, 0, 0, 0, 0, 0, 0, 1]
         ]

DISTS = [[0.1, 1, 0, 0, 0, 0, 0, 0],
         [0.2, 0, 1, 0, 0, 0, 0, 0],
         [0.3, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0.1, 1, 0, 0],
         [0, 0, 0, 0, 0.2, 0, 1, 0],
         [0, 0, 0, 0, 0.3, 0, 0, 1]
         ]

def make_dist(actions,blocks):
    assert actions/blocks == int(actions/blocks), "actions/blocks should be an integer"
    step = int(actions/blocks)
    
    DISTS = []
    for b in range(blocks):
        for i in range(1,int(step)):
            d = [0 for i in range(actions)]
            d[b*step] = i/10.
            d[b*step+i] = 1
            DISTS.append(d)
    return(DISTS)

K = 20
BLOCKS = 4
DISTS = make_dist(K,BLOCKS)

class DiscreteEnv(envs.Environment):
    
    def __init__(self, horizon):
        self.horizon = horizon
        self.k = len(DISTS[0])
        self.dists = DISTS
#         self.param = np.random.dirichlet(np.ones(len(DISTS)),1)
#         self.param = self.param[0]
#         self.param = np.ones(len(DISTS))/len(DISTS)
#         idx = np.random.choice(BLOCKS)
        arr = np.ones(int(len(DISTS)/BLOCKS))/(len(DISTS)/BLOCKS)
        self.param = 0.9*arr
        for i in range(1,BLOCKS):
            self.param = np.append(self.param,0.1/(BLOCKS-1)*arr)
#         print("[DiscreteEnv] Environment initialized, param: " + str(self.param), flush=True)

        
    def start(self):
        self.h = 0
        self.done = False
        self.idx = np.random.choice(range(len(DISTS)), p=self.param)

    def step(self,action):
        assert self.done is False, "Environment must be reset"
        self.h += 1 
        if self.h == self.horizon:
            self.done = True
        r = self.dists[self.idx][action]
        return (r, self.done)


if __name__=='__main__':
    env = DiscreteEnv(5)
    for i in range(10):
        env.start()
        for j in range(5):
            a = np.random.choice(8)
            (r,done) = env.step(a)
            print("idx: %d, act: %d, rew: %0.1f" % (env.idx, a, r))
