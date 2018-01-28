#!/usr/bin/python
# -*- coding: utf-8 -*-
from ESUtil import *


class XNESAdaptive:

    '''XNES interpreation for multinormal distribution where co-variance matrix is diagonal matrix'''

    def __init__(  # number of model parameters
        self,
        num_params,
        sigma_init=0.10,
        sigma_alpha=0.20,
        sigma_decay=0.999,
        sigma_limit=0.01,
        learning_rate=0.01,
        learning_rate_decay=0.9999,
        learning_rate_limit=0.001,
        popsize=255,
        done_threshold=1e-6,
        average_baseline=True,
        weight_decay=0.01,
        rank_fitness=True,
        forget_best=True,
        diversity_base=0.10,
        ):

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_alpha = sigma_alpha
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.average_baseline = average_baseline
        if self.average_baseline:
            assert self.popsize % 2 == 0, 'Population size must be even'
            self.batch_size = int(self.popsize / 2)
        else:
            assert self.popsize & 1, 'Population size must be odd'
            self.batch_size = int((self.popsize - 1) / 2)
        self.forget_best = forget_best
        self.batch_reward = np.zeros(self.batch_size * 2)
        self.mu = np.zeros(self.num_params)
        self.sigma = np.ones(self.num_params) * self.sigma_init
        self.curr_best_mu = np.zeros(self.num_params)
        self.best_mu = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_interation = True
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        self.done_threshold = done_threshold

        self.rewardWindow = SlideWindow(20)  # adaptive diversity function
        self.entropyWindow = SlideWindow(20)  # adaptive diversity function
        self.diversityWindow = SlideWindow(4)  # recording the history diversity

        self.diversity_base = diversity_base

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''
        # antithetic sampling
        self.epsilon = np.random.randn(self.batch_size,
                self.num_params) * self.sigma.reshape(1,
                self.num_params)
        self.epsilon_full = np.concatenate([self.epsilon,
                -self.epsilon])
        if self.average_baseline:
            epsilon = self.epsilon_full
        else:

      # first population is mu, then positive epsilon, then negative epsilon

            epsilon = np.concatenate([np.zeros((1, self.num_params)),
                    self.epsilon_full])
        solutions = self.mu.reshape(1, self.num_params) + epsilon
        self.solutions = solutions
        return solutions

    def tell(self, reward_table_result):

    # input must be a numpy float array
    # reward evaluation: scalar values

        assert len(reward_table_result) == self.popsize, \
            'Inconsistent reward_table size reported.'

        reward_table = np.array(reward_table_result)

        if self.rank_fitness:
            reward_table = compute_centered_ranks(reward_table)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay,
                    self.solutions)
            reward_table += l2_decay

        reward_offset = 1
        if self.average_baseline:
            b = np.mean(reward_table)
            reward_offset = 0
        else:
            b = reward_table[0]  # baseline

        reward = reward_table[reward_offset:]
        idx = np.argsort(reward)[::-1]

    # Gradients control

        self.rewardWindow.update(np.array(reward).mean())
        self.entropyWindow.update(calEntropy(self.sigma))  # thanks

        best_reward = reward[idx[0]]
        if best_reward > b or self.average_baseline:
            best_mu = self.mu + self.epsilon_full[idx[0]]
            best_reward = reward[idx[0]]
        else:
            best_mu = self.mu
            best_reward = b

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or self.curr_best_reward \
                > self.best_reward:
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

    # using xNES idea to learn next step

        epsilon = self.epsilon_full
        change_mu = self.learning_rate * np.dot(reward, epsilon)  # done
        self.mu += change_mu

    # #using the batch size is needed or not

        sigma = self.sigma

    # I =  np.eye(self.num_params)*self.sigma

        rS = (epsilon * epsilon - (sigma * sigma).reshape(1,
              self.num_params)) / sigma.reshape(1, self.num_params)
        covGradient = np.dot(reward, rS)
        dM = 0.5 * self.learning_rate * covGradient  # cov-variance difference

    # simple obseravation based on the entropy

        self.diversity_best = self.diversity_base
        if self.rewardWindow.evident():
            diversity_bound = self.rewardWindow.lastDiff() \
                / self.entropyWindow.lastDiff()  # be positive
            self.diversity_best = min(self.diversity_best,
                    diversity_bound)
            self.diversity_best = max(0, self.diversity_best)
            if self.rewardWindow.lastDiff() < -self.rewardWindow.std() \
                or self.entropyWindow.lastDiff() < 0:
                self.diversity_best = 0
        e_sigma = self.learning_rate * self.sigma \
                * self.diversity_base
        self.sigma = self.sigma * np.exp(dM) + e_sigma  # based-on my understanding

        self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay
        if self.learning_rate > self.learning_rate_limit:
            self.learning_rate *= self.learning_rate_decay

    def done(self):
        return self.rms_stdev() < self.done_threshold

    def current_param(self):
        return self.curr_best_mu

    def best_param(self):
        return self.best_mu

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_mu, self.best_reward, self.curr_best_reward,
                self.sigma, self.diversity_best)
