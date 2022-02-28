import numpy as np
import DiscreteEnv, DiscreteAlgs

def experiment(alg,env,reps):
    
    ep_rewards = []
    ep_actions = []
    for t in range(reps):
        # print("New episode", flush=True)
        env.start()
        alg.reset()
        ep_rew = 0
        ep_act = []
        while not env.done:
            action = alg.select()
            (r,done) = env.step(action)
            alg.update(action,r)
            ep_rew += r
            ep_act.append(action)
        # print("%d" % (ep_act[0]), flush=True)
        ep_rewards.append(ep_rew)
        ep_actions.append(ep_act[0])
    return(ep_rewards, ep_actions)

if __name__=='__main__':
    import argparse, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store', default=1000, type=int)
    parser.add_argument('--iters', action='store', default=1, type=int)
    parser.add_argument('--H', action='store', default=10, type=int)
    parser.add_argument('--alg', action='store', choices=['mts', 'mkg', 'mis-kg', 'mis-ts', 'oracle-kg', 'oracle-ts'])
    parser.add_argument('--train', action='store', default=0, type=int)
    parser.add_argument('--outdir_pref', action='store', default='./', type=str)

    Args = parser.parse_args(sys.argv[1:])
    print(Args, flush=True)

    outdir = "%sresults/discrete_T=%d_H=%d/" % (Args.outdir_pref, Args.T, Args.H)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    rewards = []
    actions = []

    for i in range(Args.iters):
        env = DiscreteEnv.DiscreteEnv(Args.H)
        if Args.alg == 'mts':
            Alg = DiscreteAlgs.ExploreTS(env.k,Args.train)
        if Args.alg == 'mkg':
            Alg = DiscreteAlgs.ExploreKG(env.k,Args.train)
        if Args.alg == 'mis-ts':
            Alg = DiscreteAlgs.DiscreteTS(env.k,np.ones(len(DiscreteEnv.DISTS))/len(DiscreteEnv.DISTS))
        if Args.alg == 'mis-kg':
            Alg = DiscreteAlgs.DiscreteKG(env.k,np.ones(len(DiscreteEnv.DISTS))/len(DiscreteEnv.DISTS))
        if Args.alg == 'oracle-ts':
            Alg = DiscreteAlgs.DiscreteTS(env.k,env.param)
        if Args.alg == 'oracle-kg':
            Alg = DiscreteAlgs.DiscreteKG(env.k,env.param)

        (rew,act) = experiment(Alg,env,Args.T)
        rewards.append(rew)
        actions.append(act)
        print("Iter: %d, test_reward: %0.3f" % (i, np.mean(rew[int(Args.T/2):])), flush=True)
        
    np.savetxt(outdir+"%s_%d_rewards.out" % (Args.alg, Args.train), rewards)
    np.savetxt(outdir+"%s_%d_actions.out" % (Args.alg, Args.train), actions)
