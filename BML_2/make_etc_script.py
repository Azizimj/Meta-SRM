
prefix = "python3 exp.py --type etc --T 200 --H 20 --K 2"
for alg in ["oracle", "mpc"]:
    for t in range(10):
        print(prefix + " --alg %s --train_len %d" % (alg, t))
