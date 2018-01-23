## BART models

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

I_MAX = 50 # maximum number of pumps ever tried by a player

def cara_utility(rewards, gamma):
    return(rewards**gamma)


class Balloon(object):
    def __init__(self, p_pop, rewards):
        self.popped = False
        self.banked = False
        self.pumps = 0
        self.rewards = rewards
        self.p_pop = p_pop
        self.unconditional_p_pop = np.zeros(np.size(self.p_pop))
        p_survive = 1
        for i, p in enumerate(self.p_pop):
            self.unconditional_p_pop[i] = p_survive*(p)
            p_survive *= (1-p)

    def pump(self):
        if self.popped:
            raise RuntimeError('Attempting pump in a popped balloon.')
        if (self.p_pop[self.pumps] > np.random.uniform()):
            self.popped = True
        self.pumps += 1
        return(self.popped)

    def bank(self):
        if self.popped:
            return(0)
        else:
            self.banked = True
            return(self.rewards[self.pumps])

    def plot_conditional_p_pop(self, show = True):
        plt.stem(self.p_pop)
        plt.title('Conditional p_pop')
        if show:
            plt.show()

    def plot_unconditional_p_pop(self, show = True):
        plt.stem(self.unconditional_p_pop)
        plt.title('Unconditional p_pop')
        if show:
            plt.show()

    def get_state(self):
        return(dict(pumps = self.pumps, popped = self.popped, banked = self.banked))


class Experiment:
    def __init__(self, p_pop, rewards, n = 0, player = None):
        # p_pop: n_balloons * max_pump size matrix, p_pop[i,j] = probability of balloon i popping at trial j 
        #        (conditioned on not popping until then)
        # rewards: rewards[i,j] = reward of balloon i after j pumps
        assert p_pop.shape == rewards.shape
        self.p_pop = p_pop
        self.rewards = rewards
        self.player = player
        self.n_balloons = n
        if self.p_pop.ndim > 1:
            self.n_balloons = self.p_pop.shape[0]
        self.setup()

    def setup(self):
        if self.p_pop.ndim > 1:
            self.balloons = [Balloon(p_pop = self.p_pop[i,:], rewards = self.rewards[i,:]) for i in range(np.size(self.p_pop,0))]
        
        self.balloons = [Balloon(p_pop = self.p_pop, rewards = self.rewards) for i in range(self.n_balloons)]
        self.i_current_balloon = 0
        self.wallet = 0
        self.finished = False

    def get_balloon_state(self):
        return(self.current_balloon().get_state())

    def check_finished(self):
        if (self.i_current_balloon == len(self.balloons)):
            self.finished = True

    def next_balloon(self):
        self.i_current_balloon += 1
        self.check_finished()

    def current_balloon(self):
        return(self.balloons[self.i_current_balloon])

    def pump(self):
        if not self.finished:
            popped = self.current_balloon().pump()
            if(popped):
                b = self.current_balloon().get_state()
                print('Balloon popped at: ', b['pumps'])
                self.player.observe(n_pump = b['pumps'], n_pop = 1)
                self.next_balloon()
            return(popped)
        else:
            raise RuntimeError('Experiment already finished, no further pumping allowed.')

    def bank(self):
        if not self.finished:
            self.wallet += self.current_balloon().bank()
            b = self.current_balloon().get_state()
            self.player.observe(n_pump = b['pumps'], n_pop = 0)
            print('Balloon banked at: ', b['pumps'])
            self.next_balloon()
        else:
            raise RuntimeError('Experiment already finished, no further pumping allowed.')
    
    def get_state(self):
        return(dict(finished = self.finished, wallet = self.wallet, balloons = self.balloons))

    def get_data(self):
        data = []
        for b in self.balloons:
            data.append(b.get_state())
        return(data)

    def run_artificial(self, verbose = False):
        if self.player == None:
            raise RuntimeError('No artificial player given.')
        
        self.setup()
        while not self.finished:
            decision = self.player.decision(self.rewards)
            if verbose:
                print('Decision: ', decision, '   | player state: a=',self.player.a, '  m=',self.player.m)
            i = 1
            popped = False
            while (i <= decision) & (not popped):
                if (np.random.uniform() < 1/(1+np.exp(self.player.beta*(i-decision)))):
                    popped = self.pump()
                    i += 1
                else:
                    break
                
            if not popped:
                self.bank()

        if verbose:
            print('Experiment finished.')
            print('Total money earned: ', self.wallet)
        return(self.wallet)


class PlayerModel(object):
    def __init__(self):
        pass

    def get_q(self):
        q_uncond = self.get_unconditional_q()
        q_pop = np.zeros(np.size(q_uncond))
        q_sum = 0
        for i, q in enumerate(q_uncond):
            if (i == 0):
                q_pop[i] = q
            else:
                q_pop[i] = q/q_sum
            q_sum += q
        return(q_pop)

    def get_unconditional_q(self):
        q_pop = self.get_q()
        q_uncond = np.zeros(np.size(q_pop))
        q_product = 1
        for i, q in enumerate(q_pop):
            q_pop[i] = q*q_product
            q_product *= q

    def get_p(self):
        return(1-self.get_q_pop())

    def get_unconditional_p(self):
        return(1-self.get_unconditional_q())

    def get_choice(self, rewards):
        pass


class Model_3(PlayerModel): 
    # based on Wallsten et al. (2008)
    def __init__(self, a0, m0, gamma, beta, naive = False):
        self.a0 = a0
        self.m0 = m0
        self.reset()
        self.gamma = gamma
        self.beta = beta
        self.i_max = I_MAX
        if naive:
            self.decision = self.naive_decision
        else:
            self.decision = self.not_naive_decision

    def observe(self, n_pump, n_pop):
        self.a += n_pump-n_pop
        self.m += n_pop

    def utility(self, rewards):
        return(cara_utility(rewards, self.gamma))

    def expected_utility(self, rewards, i, qs = np.linspace(0,1,100)):
        integral = 0
        integral = np.sum(self.q_pdf(qs)*qs**i)
        #for q in qs:
        #    integral+= self.q_pdf(q)*q**i
        integral = integral*self.utility(rewards[i])/len(qs)
        return(integral)

    def q_pdf(self, q):
        return(sp.beta.pdf(q, self.a, self.m))

    def argmax_expected_utility(self, rewards, qs = np.linspace(0,1,100)):
        us = [self.expected_utility(rewards, i) for i in range(self.i_max)]
        return(np.argmax(us))

    def not_naive_decision(self, rewards):
        return(self.argmax_expected_utility(rewards))

    def naive_decision(self, rewards):
        return(int(-self.gamma/np.log(self.a/(self.a+self.m))))

    def reset(self):
        self.a = self.a0
        self.m = self.m0

