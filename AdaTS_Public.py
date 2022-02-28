#@title Imports and defaults
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import loggamma
from scipy.stats import beta
import time, os

import multiprocessing as mp

num_cpu = min(10, mp.cpu_count())
# num_cpu = 2
parr = 0  # mp off, also BML not included
parr = 1  # mp on, #TODO has some issues with OracleTS, TS that always gives zero sreg

from utils import save_res, linestyle2dashes, alg_labels, get_pst_time
from BML_2.exp import one_iter_standard

time_stamp = get_pst_time()

# np.random.seed(110)

BAI_flg = 1  # BAI behavior on and off
BML_flag = True
BML_algs = ['mts', 'mts-no-cov']


results_dir = "./res"
output_dir = "./out"

if not os.path.isdir(output_dir):
  os.makedirs(output_dir)
if not os.path.isdir(results_dir):
  os.makedirs(results_dir)

mpl.style.use("classic")
mpl.rcParams["figure.figsize"] = [5, 3]
mpl.rcParams["figure.figsize"] = [5, 6]

mpl.rcParams["axes.linewidth"] = 0.75
mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams["grid.linewidth"] = 0.75
mpl.rcParams["lines.linewidth"] = 0.75
mpl.rcParams["patch.linewidth"] = 0.75
mpl.rcParams["xtick.major.size"] = 3
mpl.rcParams["ytick.major.size"] = 3

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 9
mpl.rcParams["axes.titlesize"] = "medium"
mpl.rcParams["legend.fontsize"] = "medium"

import platform
# print("python %s" % platform.python_version())
# print("matplotlib %s" % mpl.__version__)


#########--------------------------------

#@title Bandit simulator and environments
class BerBandit(object):
  """Bernoulli bandit."""

  def __init__(self, mu):
    self.mu = np.copy(mu)
    self.K = self.mu.size

    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = (np.random.rand() < self.mu).astype(float)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Bernoulli bandit with arms (%s)" % \
      ", ".join("%.3f" % s for s in self.mu)


class BetaBandit(object):
  """Beta bandit."""

  def __init__(self, mu, a_plus_b=4):
    self.mu = np.copy(mu)
    self.K = self.mu.size
    self.a_plus_b = a_plus_b

    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = \
      np.random.beta(self.a_plus_b * self.mu, self.a_plus_b * (1 - self.mu))

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Beta bandit with arms (%s)" % \
      ", ".join("%.3f" % s for s in self.mu)


class GaussBandit(object):
  """Gaussian bandit."""

  def __init__(self, mu, sigma=0.5):
    self.mu = np.copy(mu)
    self.K = self.mu.size
    self.sigma = sigma

    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = self.mu + self.sigma * np.random.randn(self.K)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Gaussian bandit with arms (%s)" % \
      ", ".join("%.3f" % s for s in self.mu)


class LinBandit(object):
  """Linear bandit."""

  def __init__(self, X, theta, noise="normal", sigma=0.5):
    self.X = np.copy(X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.theta = np.copy(theta)
    self.noise = noise
    if self.noise == "normal":
      self.sigma = sigma

    self.mu = self.X.dot(self.theta)
    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    if self.noise == "normal":
      self.rt = self.mu + self.sigma * np.random.randn(self.K)
    elif self.noise == "bernoulli":
      self.rt = (np.random.rand(self.K) < self.mu).astype(float)
    elif self.noise == "beta":
      self.rt = np.random.beta(4 * self.mu, 4 * (1 - self.mu))

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    if self.noise == "normal":
      return "Linear bandit: %d dimensions, %d arms" % \
        (self.d, self.K)
    elif self.noise == "bernoulli":
      return "Bernoulli linear bandit: %d dimensions, %d arms" % \
        (self.d, self.K)
    elif self.noise == "beta":
      return "Beta linear bandit: %d dimensions, %d arms" % \
        (self.d, self.K)


class LogBandit(object):
  """Logistic bandit."""

  def __init__(self, X, theta):
    self.X = np.copy(X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.theta = np.copy(theta)

    self.mu = 1 / (1 + np.exp(- self.X.dot(self.theta)))
    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = (np.random.rand(self.K) < self.mu).astype(float)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Logistic bandit: %d dimensions, %d arms" % (self.d, self.K)

  @staticmethod
  def ball_env(d=3, K=10, num_env=100):
    """Arm features and theta are generated randomly in a ball."""

    env = []
    for env_id in range(num_env):
      # standard d-dimensional basis (with a bias term)
      basis = np.eye(d)
      basis[:, -1] = 1

      # arm features in a unit (d - 2)-sphere
      X = np.random.randn(K, d - 1)
      X /= np.sqrt(np.square(X).sum(axis=1))[:, np.newaxis]
      X = np.hstack((X, np.ones((K, 1))))  # bias term
      X[: basis.shape[0], :] = basis

      # parameter vector in a (d - 2)-sphere with radius 1.5
      theta = np.random.randn(d - 1)
      theta *= 1.5 / np.sqrt(np.square(theta).sum())
      theta = np.append(theta, [0])

      # create environment
      env.append(LogBandit(X, theta))
      print("%3d: %.2f %.2f | " % (env[-1].best_arm,
        env[-1].mu.min(), env[-1].mu.max()), end="")
      if (env_id + 1) % 10 == 0:
        print()

    return env


class CoBandit(object):
  """Contextual bandit with linear generalization."""

  def __init__(self, X, Theta, sigma=0.5):
    self.X = np.copy(X)  # [number of contexs] x d feature matrix
    self.Theta = np.copy(Theta)  # d x [number of arms] parameter matrix
    self.K = self.Theta.shape[1]
    self.d = self.X.shape[1]
    self.num_contexts = self.X.shape[0]
    self.sigma = sigma

    self.mu = self.X.dot(self.Theta)
    self.best_arm = np.argmax(self.mu, axis=1)

    self.randomize()

  def randomize(self):
    # choose context
    self.ct = np.random.randint(self.num_contexts)
    self.mut = self.mu[self.ct, :]

    # generate stochastic rewards
    self.rt = self.mut + self.sigma * np.random.randn(self.K)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm[self.ct]] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mut[self.best_arm[self.ct]] - self.mut[arm]

  def print(self):
    return "Contextual bandit: %d dimensions, %d arms" % (self.d, self.K)


#########3-----------------------------------

#@title Classical algorithms for multi-armed, linear, and GLM bandits
class UCB1:
  def __init__(self, env, n, params):
    self.K = env.K
    self.crs = 1.0  # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = 1e-6 * np.ones(self.K)  # number of pulls
    self.reward = 1e-6 * np.random.rand(self.K)  # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(self.K)  # tie breaking

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r

  def get_arm(self, t):
    # UCBs
    t += 1  # time starts at one
    ciw = self.crs * np.sqrt(2 * np.log(t))
    self.ucb = self.reward / self.pulls + \
      ciw * np.sqrt(1 / self.pulls) + self.tiebreak

    arm = np.argmax(self.ucb)
    return arm

  @staticmethod
  def print():
    return "UCB1"


class UCBV:
  def __init__(self, env, n, params):
    self.K = env.K
    self.n = n

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = np.zeros(self.K)  # number of pulls
    self.reward = np.zeros(self.K)  # cumulative reward
    self.reward2 = np.zeros(self.K)  # cumulative squared reward
    self.tiebreak = 1e-6 * np.random.rand(self.K)  # tie breaking

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r
    self.reward2[arm] += r * r

  def get_arm(self, t):
    if t < self.K:
      # pull each arm once in the first K rounds
      self.ucb = np.zeros(self.K)
      self.ucb[t] = 1
    else:
      # UCBs
      t += 1  # time starts at one

      # from \sum_{t = 1}^n \sum_{s = 1}^t (1 / n^2) <= 1
      delta = 1.0 / np.power(self.n, 2)
      # # from \sum_{t = 1}^n \sum_{s = 1}^t (1 / t^3) <= \pi^2 / 6
      # delta = 1.0 / np.power(t, 3)

      muhat = self.reward / self.pulls
      varhat = (self.reward2 - self.pulls * np.square(muhat)) / self.pulls
      varhat = np.maximum(varhat, 0)
      self.ucb = muhat + \
        np.sqrt(2 * varhat * np.log(3 / delta) / self.pulls) + \
        3 * np.log(3 / delta) / self.pulls + \
        self.tiebreak

    arm = np.argmax(self.ucb)
    return arm

  @staticmethod
  def print():
    return "UCB-V"


class KLUCB:
  def __init__(self, env, n, params):
    self.K = env.K

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = 1e-6 * np.ones(self.K)  # number of pulls
    self.reward = 1e-6 * np.random.rand(self.K)  # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(self.K)  # tie breaking

  def UCB(self, p, N, t):
    C = (np.log(t) + 3 * np.log(np.log(t) + 1e-6)) / N

    qmin = np.minimum(np.maximum(p, 1e-6), 1 - 1e-6)
    qmax = (1 - 1e-6) * np.ones(p.size)
    for i in range(16):
      q = (qmax + qmin) / 2
      ndx = (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))) < C
      qmin[ndx] = q[ndx]
      qmax[~ndx] = q[~ndx]

    return q

  def update(self, t, arm, r):
    if (r > 0) and (r < 1):
      r = (np.random.rand() < r).astype(float)
    self.pulls[arm] += 1
    self.reward[arm] += r

  def get_arm(self, t):
    # UCBs
    t += 1  # time starts at one
    self.ucb = \
      self.UCB(self.reward / self.pulls, self.pulls, t) + self.tiebreak

    arm = np.argmax(self.ucb)
    return arm

  @staticmethod
  def print():
    return "KL-UCB"


class TS:
  def __init__(self, env, n, params):
    self.K = env.K
    self.crs = 1.0  # confidence region scaling

    self.alpha = np.ones(self.K)  # positive observations
    self.beta = np.ones(self.K)  # negative observations

    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

  def update(self, t, arm, r):
    if (r > 0) and (r < 1):
      r = (np.random.rand() < r).astype(float)
    self.alpha[arm] += r
    self.beta[arm] += 1 - r

  def get_arm(self, t):
    if t < self.K:
      # each arm is initially pulled once
      self.mu = np.zeros(self.K)
      self.mu[t] = 1
    else:
      # posterior sampling
      crs2 = np.square(self.crs)
      self.mu = np.random.beta(self.alpha / crs2, self.beta / crs2)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "TS"


class GaussTS:
  def __init__(self, env, n, params):
    self.K = env.K
    self.sigma = 0.5

    self.mu0 = np.zeros(self.K)
    self.sigma0 = 0.5 * np.ones(self.K)

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = np.zeros(self.K)  # number of pulls
    self.reward = np.zeros(self.K)  # cumulative reward

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r

  def get_arm(self, t):
    if t < self.K:
      # each arm is initially pulled once
      self.mu = np.zeros(self.K)
      self.mu[t] = 1
    else:
      # posterior distribution
      sigma2 = np.square(self.sigma)
      sigma02 = np.square(self.sigma0)
      post_var = 1.0 / (1.0 / sigma02 + self.pulls / sigma2)
      post_mean = post_var * (self.mu0 / sigma02 + self.reward / sigma2)

      # posterior sampling
      self.mu = post_mean + np.sqrt(post_var) * np.random.randn(self.K)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Gaussian TS"


class EpsilonGreedy:
  def __init__(self, env, n, params):
    self.env = env
    self.K = env.K
    self.epsilon = self.K / np.sqrt(n)
    self.crs = 1.0  # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = 1e-6 * np.ones(self.K)  # number of pulls
    self.reward = 1e-6 * np.ones(self.K)  # cumulative reward
    # self.tiebreak = 1e-6 * np.random.rand(self.K)  # tie breaking

    self.grad = np.zeros(n)
    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r

    best_r = self.env.rt[self.env.best_arm]
    if self.is_baseline:
      # baseline action and update
      base_arm = self.base_alg.get_arm(t)
      base_r = self.env.reward(base_arm)
      self.base_alg.update(t, base_arm, base_r)

      self.metrics[t, :] = np.asarray([r, r - best_r, r - base_r])
    else:
      self.metrics[t, :] = np.asarray([r, r - best_r, 0])

  def get_arm(self, t):
    # decision statistics
    muhat = self.reward / self.pulls
    best_arm = np.argmax(muhat)

    # probabilities of pulling arms
    eps = self.crs * self.epsilon
    p = (1 - eps) * (np.arange(self.K) == best_arm) + eps / self.K

    # pull the arm
    arm = best_arm
    if np.random.rand() < eps:
      arm = np.random.randint(self.K)

    # derivative of the probability of the pulled arm
    self.grad[t] = self.epsilon * (1 / self.K - (arm == best_arm)) / p[arm]

    return arm

  @staticmethod
  def print():
    return "e-greedy"


class Exp3:
  def __init__(self, env, n, params):
    self.env = env
    self.K = env.K
    self.crs = min(1, np.sqrt(self.K * np.log(self.K) / ((np.e - 1) * n)))

    for attr, val in params.items():
      setattr(self, attr, val)

    self.eta = self.crs / self.K
    self.reward = np.zeros(self.K)  # cumulative reward

    self.grad = np.zeros(n)
    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  def update(self, t, arm, r):
    self.reward[arm] += r / self.phat[arm]

    best_r = self.env.rt[self.env.best_arm]
    if self.is_baseline:
      # baseline action and update
      base_arm = self.base_alg.get_arm(t)
      base_r = self.env.reward(base_arm)
      self.base_alg.update(t, base_arm, base_r)

      self.metrics[t, :] = np.asarray([r, r - best_r, r - base_r])
    else:
      self.metrics[t, :] = np.asarray([r, r - best_r, 0])

  def get_arm(self, t):
    # probabilities of pulling arms
    scaled_reward = self.reward - self.reward.max()
    p = np.exp(self.eta * scaled_reward)
    p /= p.sum()
    self.phat = (1 - self.crs) * p + self.eta

    # pull the arm
    q = np.cumsum(self.phat)
    arm = np.flatnonzero(np.random.rand() * q[-1] < q)[0]

    # derivative of the probability of the pulled arm
    self.grad[t] = (1 / self.phat[arm]) * \
      ((1 - self.crs) * (p[arm] / self.K) *
      (scaled_reward[arm] - p.dot(scaled_reward)) - p[arm] + 1 / self.K)

    return arm

  @staticmethod
  def print():
    return "Exp3"


class FPL:
  def __init__(self, env, n, params):
    self.K = env.K
    self.eta = np.sqrt((np.log(self.K) + 1) / (self.K * n))

    for attr, val in params.items():
      setattr(self, attr, val)

    self.loss = np.zeros(self.K) # cumulative loss

  def update(self, t, arm, r):
    # estimate the probability of pulling the arm
    wait_time = 0
    while True:
      wait_time += 1
      ploss = self.loss + np.random.exponential(1 / self.eta, self.K)
      if np.argmin(ploss) == arm:
        break;

    self.loss[arm] += (1 - r) * wait_time

  def get_arm(self, t):
    # perturb cumulative loss
    ploss = self.loss + np.random.exponential(1 / self.eta, self.K)

    arm = np.argmin(ploss)
    return arm

  @staticmethod
  def print():
    return "FPL"


class LinBanditAlg:
  def __init__(self, env, n, params):
    self.env = env
    self.X = np.copy(env.X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.n = n
    self.theta0 = np.zeros(self.d)
    self.sigma0 = 1.0
    self.sigma = 0.5
    self.crs = 1.0 # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    if not hasattr(self, "Sigma0"):
      self.Sigma0 = np.square(self.sigma0) * np.eye(self.d)

    # sufficient statistics
    self.Gram = np.linalg.inv(self.Sigma0)
    self.B = self.Gram.dot(self.theta0)

  def update(self, t, arm, r):
    x = self.X[arm, :]
    self.Gram += np.outer(x, x) / np.square(self.sigma)
    self.B += x * r / np.square(self.sigma)


class LinUCB(LinBanditAlg):
  def __init__(self, env, n, params):
    LinBanditAlg.__init__(self, env, n, params)

    self.cew = self.crs * self.confidence_ellipsoid_width(n)

  def confidence_ellipsoid_width(self, t):
    # Theorem 2 in Abassi-Yadkori (2011)
    # Improved Algorithms for Linear Stochastic Bandits
    delta = 1 / self.n
    L = np.amax(np.linalg.norm(self.X, axis=1))
    Lambda = 1 / np.square(self.sigma0)
    R = self.sigma
    S = np.sqrt(self.d)
    width = np.sqrt(Lambda) * S + \
      R * np.sqrt(self.d * np.log((1 + t * np.square(L) / Lambda) / delta))
    return width

  def get_arm(self, t):
    Gram_inv = np.linalg.inv(self.Gram)
    theta = Gram_inv.dot(self.B)

    # UCBs
    Gram_inv /= np.square(self.sigma)
    self.mu = self.X.dot(theta) + self.cew * \
      np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis=1))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinUCB"


class LinGreedy(LinBanditAlg):
  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if np.random.rand() < 0.05 * np.sqrt(self.n / (t + 1)) / 2:
      self.mu[np.random.randint(self.K)] = np.Inf
    else:
      theta = np.linalg.solve(self.Gram, self.B)
      self.mu = self.X.dot(theta)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Lin e-greedy"


class LinTS(LinBanditAlg):
  def update(self, t, arm, r):
    x = self.env.X[arm, :]
    self.Gram += np.outer(x, x) / np.square(self.sigma)
    self.B += x * r / np.square(self.sigma)

  def get_arm(self, t):
    Gram_inv = np.linalg.inv(self.Gram)
    thetabar = Gram_inv.dot(self.B)

    # posterior sampling
    thetatilde = np.random.multivariate_normal(thetabar,
      np.square(self.crs) * Gram_inv)
    self.mu = self.env.X.dot(thetatilde)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinTS"


class CoLinTS(LinBanditAlg):
  def update(self, t, arm, r):
    x = self.env.X[arm, :]
    self.Gram += np.outer(x, x) / np.square(self.sigma)
    self.B += x * r / np.square(self.sigma)

  def get_arm(self, t):
    Gram_inv = np.linalg.inv(self.Gram)
    thetabar = Gram_inv.dot(self.B)

    # posterior sampling
    thetatilde = np.random.multivariate_normal(thetabar,
      np.square(self.crs) * Gram_inv)
    self.mu = self.env.X.dot(thetatilde)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "CoLinTS"


class LogBanditAlg:
  def __init__(self, env, n, params):
    self.env = env
    self.X = np.copy(env.X)
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.n = n
    self.sigma0 = 1.0
    self.crs = 1.0 # confidence region scaling
    self.crs_is_width = False

    self.irls_theta = np.zeros(self.d)
    self.irls_error = 1e-3
    self.irls_num_iter = 30

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    self.pos = np.zeros(self.K, dtype=int) # number of positive observations
    self.neg = np.zeros(self.K, dtype=int) # number of negative observations
    self.X2 = np.zeros((self.K, self.d, self.d)) # outer products of arm features
    for k in range(self.K):
      self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])

  def update(self, t, arm, r):
    self.pos[arm] += r
    self.neg[arm] += 1 - r

  def sigmoid(self, x):
    y = 1 / (1 + np.exp(- x))
    return y

  def solve(self):
    # iterative reweighted least squares for Bayesian logistic regression
    # Sections 4.3.3 and 4.5.1 in Bishop (2006)
    # Pattern Recognition and Machine Learning
    theta = np.copy(self.irls_theta)

    num_iter = 0
    while num_iter < self.irls_num_iter:
      theta_old = np.copy(theta)

      Xtheta = self.X.dot(theta)
      R = self.sigmoid(Xtheta) * (1 - self.sigmoid(Xtheta))
      pulls = self.pos + self.neg
      Gram = np.tensordot(R * pulls, self.X2, axes=([0], [0])) + \
        np.eye(self.d) / np.square(self.sigma0)
      Rz = R * pulls * Xtheta - \
        self.pos * (self.sigmoid(Xtheta) - 1) - \
        self.neg * (self.sigmoid(Xtheta) - 0)
      theta = np.linalg.solve(Gram, self.X.T.dot(Rz))

      if np.linalg.norm(theta - theta_old) < self.irls_error:
        break;
      num_iter += 1

    if num_iter == self.irls_num_iter:
      self.irls_theta = np.zeros(self.d)
    else:
      self.irls_theta = np.copy(theta)

    return theta, Gram


class LogUCB(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)

    if not self.crs_is_width:
      self.cew = self.crs * self.confidence_ellipsoid_width(n)
    else:
      self.cew = self.crs

  def confidence_ellipsoid_width(self, t):
    # Section 4.1 in Filippi (2010)
    # Parametric Bandits: The Generalized Linear Case
    delta = 1 / self.n
    c_m = np.amax(np.linalg.norm(self.X, axis=1))
    c_mu = 0.25 # minimum derivative of the mean function
    k_mu = 0.25
    kappa = np.sqrt(3 + 2 * np.log(1 + 2 * np.square(c_m / self.sigma0)))
    R_max = 1.0
    width = (2 * k_mu * kappa * R_max / c_mu) * \
      np.sqrt(2 * self.d * np.log(t) * np.log(2 * self.d * self.n / delta))
    return width

  def get_arm(self, t):
    pulls = self.pos + self.neg
    Gram = np.tensordot(pulls, self.X2, axes=([0], [0])) + \
      np.eye(self.d) / np.square(self.sigma0)
    Gram_inv = np.linalg.inv(Gram)
    theta, _ = self.solve()

    # UCBs
    self.mu = self.sigmoid(self.X.dot(theta)) + self.cew * \
      np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis=1))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "GLM-UCB (log)"


class UCBLog(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)

    if not self.crs_is_width:
      self.cew = self.crs * self.confidence_ellipsoid_width(n)
    else:
      self.cew = self.crs

  def confidence_ellipsoid_width(self, t):
    # Theorem 2 in Li (2017)
    # Provably Optimal Algorithms for Generalized Linear Contextual Bandits
    delta = 1 / self.n
    sigma = 0.5
    kappa = 0.25 # minimum derivative of a constrained mean function
    width = (sigma / kappa) * \
      np.sqrt((self.d / 2) * np.log(1 + 2 * self.n / self.d) + \
      np.log(1 / delta))
    return width

  def get_arm(self, t):
    pulls = self.pos + self.neg
    Gram = np.tensordot(pulls, self.X2, axes=([0], [0])) + \
      np.eye(self.d) / np.square(self.sigma0)
    Gram_inv = np.linalg.inv(Gram)
    theta, _ = self.solve()

    # UCBs
    self.mu = self.X.dot(theta) + self.cew * \
      np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis=1))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "UCB-GLM (log)"


class LogGreedy(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)

    self.epsilon = 0.05

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if np.random.rand() < self.epsilon * np.sqrt(self.n / (t + 1)) / 2:
      self.mu[np.random.randint(self.K)] = np.Inf
    else:
      theta, _ = self.solve()
      self.mu = self.sigmoid(self.X.dot(theta))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Log e-greedy"


class LogTS(LogBanditAlg):
  def get_arm(self, t):
    thetabar, Gram = self.solve()
    Gram_inv = np.linalg.inv(Gram)

    # posterior sampling
    thetatilde = np.random.multivariate_normal(thetabar,
      np.square(self.crs) * Gram_inv)
    self.mu = self.X.dot(thetatilde)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "GLM-TSL (log)"


class LogFPL(LogBanditAlg):
  def __init__(self, env, n, params):
    self.a = 1.0

    LogBanditAlg.__init__(self, env, n, params)

  def solve(self):
    # normal noise perturbation
    pulls = self.pos + self.neg
    z = self.a * np.sqrt(pulls) * \
      np.minimum(np.maximum(np.random.randn(self.K), -6), 6)

    # iterative reweighted least squares for Bayesian logistic regression
    # Sections 4.3.3 and 4.5.1 in Bishop (2006)
    # Pattern Recognition and Machine Learning
    theta = np.copy(self.irls_theta)

    num_iter = 0
    while num_iter < self.irls_num_iter:
      theta_old = np.copy(theta)

      Xtheta = self.X.dot(theta)
      R = self.sigmoid(Xtheta) * (1 - self.sigmoid(Xtheta))
      Gram = np.tensordot(R * pulls, self.X2, axes=([0], [0])) + \
        np.eye(self.d) / np.square(self.sigma0)
      Rz = R * pulls * Xtheta - \
        (pulls * self.sigmoid(Xtheta) - (self.pos + z))
      theta = np.linalg.solve(Gram, self.X.T.dot(Rz))

      if np.linalg.norm(theta - theta_old) < self.irls_error:
        break;
      num_iter += 1

    if num_iter == self.irls_num_iter:
      self.irls_theta = np.zeros(self.d)
    else:
      self.irls_theta = np.copy(theta)

    return theta, Gram

  def get_arm(self, t):
    self.mu = np.zeros(self.K)
    if t < self.d:
      self.mu[t] = np.Inf
    else:
      # history perturbation
      theta, _ = self.solve()
      self.mu = self.sigmoid(self.X.dot(theta))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "GLM-FPL (log)"


def get_label(alg):
  if alg[1][0] != 1:
    return ""
  if alg[-1] not in BML_algs:
    return alg[-1]
  elif alg[-1]=='mts':
    return "fMetaTS"
  elif alg[-1]=='mts-mean':
    return "mfMetaTS"
  else:
    raise ValueError


def evaluate_one(Alg, params, env, n, period_size=1):
  """One run of a bandit algorithm."""
  alg = Alg(env, n, params)

  regret = np.zeros(n // period_size)
  sregret = np.zeros(n // period_size)
  for t in range(n):
    # generate state
    env.randomize()

    # take action
    arm = alg.get_arm(t)

    # update model and regret
    alg.update(t, arm, env.reward(arm))
    regret_at_t = env.regret(arm)
    regret[t // period_size] += regret_at_t

    # BAI
    idx_mf = np.argmax(alg.pulls)
    sregret[t // period_size] += env.pregret(idx_mf)

  return regret, alg, sregret


def evaluate(Alg, params, env, n=1000, period_size=1, printout=True):
  """Multiple runs of a bandit algorithm."""
  if printout:
    print("Evaluating %s" % Alg.print(), end="")
  start = time.time()

  num_exps = len(env)
  regret = np.zeros((n // period_size, num_exps))
  sregret = np.zeros((n // period_size, num_exps))
  alg_ = num_exps * [None]

  dots = np.linspace(0, num_exps - 1, 100).astype(int)
  for ex in range(num_exps):
    output = evaluate_one(Alg, params, env[ex], n, period_size)
    regret[:, ex] = output[0]
    alg_[ex] = output[1]
    sregret[:, ex] = output[2]

    if ex in dots:
      if printout:
        print(".", end="")
  if printout:
    print(" %.1f seconds" % (time.time() - start))

  if printout:
    total_regret = regret.sum(axis=0)
    print("Regret: %.2f +/- %.2f (median: %.2f, max: %.2f, min: %.2f)" %
      (total_regret.mean(), total_regret.std() / np.sqrt(num_exps),
      np.median(total_regret), total_regret.max(), total_regret.min()))

  return regret, alg_, sregret


def one_run_mab(run, alg, params):

  algs, num_runs, num_tasks = params['algs'], params['num_runs'], params['num_tasks']
  n, K = params['n'], params['K']
  mu_q, sigma_q, sigma_0 = params['mu_q'], params['sigma_q'], params['sigma_0']
  alg_num, sigma = params['alg_num'], params['sigma']
  alp_fmts = params['alp_fmts']

  num_tasks_n = num_tasks * n
  run_regret = np.zeros((num_tasks_n, 1))
  run_sregret = np.zeros((num_tasks_n, 1))


  print(f"alg {alg[-1]}, run {run}")
  # true prior
  mu_star = mu_q + sigma_q * np.random.randn(K)

  # potential meta-prior misspecification
  sigma_q_alg = alg[1][0] * sigma_q
  if alg[1][1]!=0:
    mu_q_alg = mu_q + (np.random.rand(1)*np.ones(K)*2*alg[1][1]-alg[1][1])  # move mu_q to a uniform random number in radius alg[1][1]
  else:
    mu_q_alg = 1*mu_q

  # incrementally updated statistics
  mu_inc = mu_q_alg / np.square(sigma_q_alg)
  lambda_inc = 1.0 / np.square(sigma_q_alg)

  # initial meta-posterior
  mu_hat = mu_inc / lambda_inc
  sigma_hat = 1.0 / np.sqrt(lambda_inc)

  if BML_flag and str(alg[-1]).startswith('mts'):
    if alg[-1] == 'mts-mean':
      alp_fmts = 1
    elif alg[-1] in ['mts-min']:
      alp_fmts = 0
    Args_d = {'K': K, 'H': n, 'T': num_tasks, 'alg': alg[-1]}
    # Args_d['train'] = min(100, int(num_tasks / 2))  # TODO
    train_lens = list(np.arange(int(num_tasks / 10), int(num_tasks / 2), int(num_tasks / 10)))
    rew_vecs, sreg_vecs = np.zeros((len(train_lens), num_tasks_n)), np.zeros((len(train_lens), num_tasks_n))
    Args_d['num_random'] = 2 * K  # TODO
    for cc, train_len in enumerate(train_lens):
      Args_d['train'] = train_len
      (_, _, _, _, _, _, rew_vec, sreg_vec) = one_iter_standard(Args_d, mu_star, np.diag(np.square(sigma_0)),
                                                                iter_cnt=run, prior_dist='AdaTScode',
                                                                reward_dist='GausAdaTScode')
      rew_vecs[cc, :], sreg_vecs[cc, :] = np.array(rew_vec).reshape(num_tasks_n, ), np.array(sreg_vec).reshape(num_tasks_n, )
    # sreg_vecs = np.min(sreg_vecs, axis=0)  # pointwise best
    # sreg_vec = np.mean(sreg_vecs, axis=0)  # mean, more fair
    sreg_vecs = (alp_fmts * np.mean(sreg_vecs, axis=0) + (1 - alp_fmts) * np.min(sreg_vecs, axis=0))  # mean, more fair
    # regret[:, run] += np.array(rew_vec).reshape(num_tasks_n, 1)  # rewards returned by BML #TODO
    run_sregret = np.array(sreg_vecs).reshape(num_tasks_n, 1)
  else:
    for task in range(num_tasks):
      # sample problem instance from N(\mu_*, \sigma_0^2 I_K)
      mu = mu_star + sigma_0 * np.random.randn(K)
      env = GaussBandit(mu, sigma=sigma)

      # task prior
      if alg_num == 0:
        # OracleTS
        mu_task = np.copy(mu_star)
        sigma_task = np.copy(sigma_0)
      elif alg_num == 1:
        # TS
        mu_task = np.copy(mu_q_alg)
        sigma_task = np.sqrt(np.square(sigma_0) + np.square(sigma_q))
      elif alg_num >= 2:
        if alg[-1].startswith("Meta"):
          # MetaTS
          mu_tilde = mu_hat + sigma_hat * np.random.randn(K)
          mu_task = np.copy(mu_tilde)
          sigma_task = np.copy(sigma_0)
        else:
          # AdaTS
          mu_task = np.copy(mu_hat)
          sigma_task = np.sqrt(np.square(sigma_0) + np.square(sigma_hat))

      # evaluate on a sampled problem instance
      alg_class = globals()[alg[0]]
      alg_params = {
        "mu0": mu_task,
        "sigma0": sigma_task,
        "sigma": sigma}
      task_regret, logs, task_sregret = evaluate(alg_class, alg_params, [env], n, printout=False)
      run_regret[task * n: (task + 1) * n, 0] += task_regret.flatten()
      run_sregret[task * n: (task + 1) * n, 0] += task_sregret.flatten()

      # meta-posterior update
      if alg_num >= 2:
          sigma2 = np.square(sigma)
          sigma_02 = np.square(sigma_0)

          # incremental update
          mu_inc += logs[0].reward / (logs[0].pulls * sigma_02 + sigma2)
          lambda_inc += logs[0].pulls / (logs[0].pulls * sigma_02 + sigma2)

          # updated meta-posterior
          mu_hat = mu_inc / lambda_inc
          sigma_hat = 1.0 / np.sqrt(lambda_inc)

  return run_regret, run_sregret, run


def collect_result(res):
  run_regret, run_sregret, run = res
  global regret, sregret

  regret[:, run] += run_regret.flatten()
  sregret[:, run] += run_sregret.flatten()

  return


#######-----------------------------------

if __name__ == '__main__':

  params = {}
  # Gaussian bandit
  algs = [
    ("GaussTS", 1, "cyan", "-", "OracleTS"),
    ("GaussTS", 1, "blue", "-", "TS"),
    ("GaussTS", 1, "red", "-", "AdaTS"),
    ("GaussTS", 3, "red", "--", "AdaTSx"),  # AdaTS with 3x wider meta-prior
    ("GaussTS", 0.333, "red", ":", "AdaTSd"),  # AdaTS with 3x narrower meta-prior
    ("GaussTS", 1, "gray", "-", "MetaTS"),
    ("GaussTS", 1, "green", "-", "mts")] # mts from BML_2
    # ,("GaussTS", 1, "green", "--", "mts-no-cov")]  # mts no cov from BML_2 (supposed to be the same as MetaTS)

  algs = [
    ("GaussTS", [1,0], "OracleTS"),  # (1,0) is 1 for variance of meta-prior misspec, 0 for mean of meta-prior misspec
    ("GaussTS", [1,0], "TS"),
    ("GaussTS", [1,0], "AdaTS"),#]
    ("GaussTS", [1,50], "MisAdaTS")#] # misspecified AdaTS with a random meta-prior mean (one more level)
    ,("GaussTS", [1,0], "mts")]
    # ("GaussTS", 1, "green", "-.", "mts-min"), # mts from BML_2
    # ("GaussTS", 1, "pink", "-.-", "mts-mean")] # mts from BML_2
    # ,("GaussTS", 1, "green", "--", "mts-no-cov")]  # mts no cov from BML_2 (supposed to be the same as MetaTS)

  params['algs'] = algs

  num_runs = 100
  # num_runs = 10
  # num_runs = 3
  params['num_runs'] = num_runs

  # num_tasks = 20
  # num_tasks = 1000
  # num_tasks = 500
  num_tasks = 200
  # num_tasks = 20
  params['num_tasks'] = num_tasks

  # n = 200 # Horizon
  n = 20
  params['n'] = n

  Ks = [8, 16]
  # Ks = [16, 32]
  # Ks = [4]
  Ks = [2, 6, 8]
  # Ks = [4]
  params['Ks'] = Ks

  sigma_q_scales = [0.5, 1, 2]
  sigma_q_scales = [1]
  params['sigma_q_scales'] = sigma_q_scales

  step = np.arange(1, n * num_tasks + 1) / n
  sube = (step.size // 10) * np.arange(1, 11) - 1

  # reward noise
  sigma = 1.0
  params['sigma'] = sigma

  alp_fmts = 1 # hyperparameter of fMetaTS, mean coefficient
  # alp_fmts = 0
  # alp_fmts = .7
  params['alp_fmts'] = alp_fmts

  print(f"Ks {Ks}, num_runs {num_runs}, num_tasks {num_tasks}, n {n}, sigma_q_scales {sigma_q_scales},"
        f" sigma {sigma}, algs {algs}.")

  for K in Ks:
    params['K'] = K
    for sigma_q_scale in sigma_q_scales:
      params['sigma_q_scale'] = sigma_q_scale
      # meta-prior parameters
      mu_q = np.zeros(K)
      params['mu_q'] = mu_q
      sigma_q = sigma_q_scale * np.ones(K)
      params['sigma_q'] = sigma_q
      # prior parameters
      sigma_0 = 0.1 * np.ones(K)
      params['sigma_0'] = sigma_0

      plt.figure(figsize=(4, 2.5))

      alg_num = 0
      params['alg_num'] = alg_num
      num_tasks_n = num_tasks * n
      for alg in algs:
        np.random.seed(110)
        print(alg)
        regret = np.zeros((num_tasks_n, num_runs))
        sregret = np.zeros((num_tasks_n, num_runs))

        if parr:
          pool = mp.Pool(num_cpu)
          poolobjs = [pool.apply_async(one_run_mab, args=[run_, alg, params], callback=collect_result)
                      for run_ in range(num_runs)]
          pool.close()
          pool.join()
          for f in poolobjs:
              print(f.get())  # print the errors

          # args_ = [[run_, alg, params] for run_ in range(num_runs)]
          # with mp.Pool(num_cpu) as pool:
          #   poolobjs = pool.starmap(one_run_mab, args_)
          # for run_ in range(num_runs):
          #   run_regret, run_sregret, _ = poolobjs[run_]
          #   # regret[:, run_] += run_regret.flatten()
          #   sregret[:, run_] += run_sregret.flatten()

          # for run_ in range(num_runs):
          #   run_regret, run_sregret, _ = one_run_mab(run_, alg, params)
          #   # regret[:, run_] += run_regret.flatten()
          #   sregret[:, run_] += run_sregret.flatten()
          #   pass
        else:
          for run in range(num_runs):
            print(f"alg {alg[-1]}, run {run}")
            # true prior
            mu_star = mu_q + sigma_q * np.random.randn(K)

            # potential meta-prior misspecification
            sigma_q_alg = alg[1][0] * sigma_q
            if alg[1][1] != 0:
              mu_q_alg = mu_q + (np.random.rand(1) * np.ones(K) * 2 * alg[1][1] - alg[1][1])  # move mu_q to a uniform random number in radius alg[1][1]
            else:
              mu_q_alg = 1*mu_q


            # incrementally updated statistics
            mu_inc = mu_q_alg / np.square(sigma_q_alg)
            lambda_inc = 1.0 / np.square(sigma_q_alg)

            # initial meta-posterior
            mu_hat = mu_inc / lambda_inc
            sigma_hat = 1.0 / np.sqrt(lambda_inc)

            if BML_flag and str(alg[-1]).startswith('mts'):
              if alg[-1] == 'mts-mean':
                alp_fmts = 1
              elif alg[-1] in ['mts-min']:
                alp_fmts = 0
              Args_d = {'K': K, 'H': n, 'T': num_tasks, 'alg': alg[-1]}
              # Args_d['train'] = min(100, int(num_tasks / 2))  # TODO
              train_lens = list(np.arange(int(num_tasks / 10), int(num_tasks / 2), int(num_tasks / 10)))
              rew_vecs, sreg_vecs = np.zeros((len(train_lens), num_tasks_n)), np.zeros((len(train_lens), num_tasks_n))
              # Args_d['num_random'] = 2 * K  # TODO
              Args_d['num_random'] = 10  # TODO
              for cc, train_len in enumerate(train_lens):
                Args_d['train'] = train_len
                (_, _, _, _, _, _, rew_vec, sreg_vec) = one_iter_standard(Args_d, mu_star, np.diag(np.square(sigma_0)),
                                                                          iter_cnt=run, prior_dist='AdaTScode',
                                                                          reward_dist='GausAdaTScode')
                rew_vecs[cc, :], sreg_vecs[cc, :] = \
                  np.array(rew_vec).reshape(num_tasks_n, ), np.array(sreg_vec).reshape(num_tasks_n, )
              # sreg_vec = np.min(sreg_vecs, axis=0)  # pointwise best
              # sreg_vec = np.mean(sreg_vecs, axis=0)  # mean, more fair
              sreg_vecs = (alp_fmts * np.mean(sreg_vecs, axis=0) + (1 - alp_fmts) * np.min(sreg_vecs, axis=0))  # mean, more fair
              # regret[:, run] += np.array(rew_vec).reshape(num_tasks_n, 1)  # rewards returned by BML #TODO
              sregret[:, run] = (sregret[:, run].reshape(num_tasks_n, 1) +
                                 np.array(sreg_vecs).reshape(num_tasks_n, 1)).reshape(num_tasks_n,)
            else:
              for task in range(num_tasks):
                # sample problem instance from N(\mu_*, \sigma_0^2 I_K)
                mu = mu_star + sigma_0 * np.random.randn(K)
                env = GaussBandit(mu, sigma=sigma)

                # task prior
                if alg_num == 0:
                  # OracleTS
                  mu_task = np.copy(mu_star)
                  sigma_task = np.copy(sigma_0)
                elif alg_num == 1:
                  # TS
                  mu_task = np.copy(mu_q_alg)
                  sigma_task = np.sqrt(np.square(sigma_0) + np.square(sigma_q))
                elif alg_num >= 2:
                  if alg[-1].startswith("Meta"):
                    # MetaTS
                    mu_tilde = mu_hat + sigma_hat * np.random.randn(K)
                    mu_task = np.copy(mu_tilde)
                    sigma_task = np.copy(sigma_0)
                  else:
                    # AdaTS
                    mu_task = np.copy(mu_hat)
                    sigma_task = np.sqrt(np.square(sigma_0) + np.square(sigma_hat))

                # evaluate on a sampled problem instance
                alg_class = globals()[alg[0]]
                alg_params = {
                  "mu0": mu_task,
                  "sigma0": sigma_task,
                  "sigma": sigma}
                task_regret, logs, task_sregret = evaluate(alg_class, alg_params, [env], n, printout=False)
                regret[task * n : (task + 1) * n, run] += task_regret.flatten()
                sregret[task * n : (task + 1) * n, run] += task_sregret.flatten()

                # meta-posterior update
                if alg_num >= 2:
                  sigma2 = np.square(sigma)
                  sigma_02 = np.square(sigma_0)

                  # incremental update
                  mu_inc += logs[0].reward / (logs[0].pulls * sigma_02 + sigma2)
                  lambda_inc += logs[0].pulls / (logs[0].pulls * sigma_02 + sigma2)

                  # updated meta-posterior
                  mu_hat = mu_inc / lambda_inc
                  sigma_hat = 1.0 / np.sqrt(lambda_inc)

        fname = f"Gaus-{alg[-1]}-K={K}-sigq={sigma_q_scale}-nr{num_runs}-" \
                f"nt{num_tasks}-n{n}-{'SR' if BAI_flg else 'R'}-{time_stamp}"
        save_res(f'{output_dir}/{fname}.out', regret, sregret)

        if BAI_flg:
          cum_regret = sregret.cumsum(axis=0)
        else:
          cum_regret = regret.cumsum(axis=0)  # cumulative regret
        plt.plot(step, cum_regret.mean(axis=1), color=alg_labels[alg[-1]][1],
                 dashes=linestyle2dashes(alg_labels[alg[-1]][2]), label=alg_labels[alg[-1]][0])
                 # label=get_label(alg))
          # label=alg[-1] if alg[1][0] == 1 else "")
        plt.errorbar(step[sube], cum_regret[sube, :].mean(axis=1),
                     cum_regret[sube, :].std(axis=1) / np.sqrt(cum_regret.shape[1]),
                     fmt="none", ecolor=alg_labels[alg[-1]][1])

        print("%s: %.1f +/- %.1f" % (alg[-1],
          cum_regret[-1, :].mean(),
          cum_regret[-1, :].mean() / np.sqrt(cum_regret.shape[1])))

        alg_num += 1
        params['alg_num'] = alg_num

      titl = r"Gaussian (K = %d, n = %d, $\sigma_q$ = %.3f)" % (K, n, sigma_q_scale)
      plt.title(titl)
      plt.xlabel("Num of Tasks")
      plt.ylabel("Regret" if not BAI_flg else "Simple Regret")
      plt.ylim(bottom=0)
      # plt.legend(loc="upper left", frameon=False)
      plt.legend(loc="best", frameon=False)

      plt.tight_layout()
      # plt.show()
      fname = f"Gaus-K={K}-sigq={sigma_q_scale}-nr{num_runs}-nt{num_tasks}-n{n}-{'SR' if BAI_flg else 'R'}-alp{alp_fmts}" \
              f"-{time_stamp}"
      plt.savefig(f'{results_dir}/{fname}.pdf', format='pdf', dpi=100, bbox_inches='tight')
      # plt.close()