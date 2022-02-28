

train_rounds = [200,400,600,800,1000,1500,2000]

prefix = "python3 exp.py --type standard --instance block --T 5000 --iters 20 --H 50 --K 50 "

print(prefix + " --alg oracle --train 0 --num_random 0")
print(prefix + " --alg misspecified --train 0 --num_random 0")
for t in train_rounds:
    print(prefix + " --alg mts-no-cov --train %d --num_random 0" % (t))
    print(prefix + " --alg mts --train %d --num_random 2" % (t))
    print(prefix + " --alg mts --train %d --num_random 10" % (t))
    print(prefix + " --alg mts --train %d --num_random 50" % (t))
