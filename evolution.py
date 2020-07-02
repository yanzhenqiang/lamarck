
import torch.multiprocessing as mp

def es_params(self):
    """
    The params that should be trained by ES (all of them)
    """
    return [(k, v) for k, v in zip(self.state_dict().keys(),
                                    self.state_dict().values())]

def perturb_model(args, model, random_seed, env):
    new_model = ES(env.observation_space.shape[0],
                   env.action_space, args.small_net)
    new_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (k, v) in new_model.es_params():
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(args.sigma*eps).float()
    return new_model


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.env = gym.make('CartPole-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                if self.name == 'w00':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))
global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]

import collections
import itertools
import multiprocessing

from ./model import PolicyNet, EvaluateNet

class EvolutionManager(object):
    
    def __init__(self, args):
        self.pool = multiprocessing.Pool(args.worker)
        env = gym.make(args.env)
        o_dim = env.observation_space.shape[0]
        s_dim = 128
        a_dim = env.action_space.n
        self.nets = []
        for i in range(args.poputation):
            policy_net = PolicyNet(o_dim, s_dim, a_dim)
            policy_net.share_memory()
            self.nets.append(policy_net)
            evaluate_net = EvaluateNet(o_dim, s_dim, a_dim)
            evaluate_net.share_memory()
            self.nets.append(evaluate_net)
    
    def generation_forward(self):
        pass

    def mutate(self):
        pass

    def __call__(self, inputs, chunksize=1):
        """Process the inputs.
        
        inputs
          models/env_name/epochs.
        
        chunksize=1
          The portion of the input data to hand to each worker.
        """
        # rewards should contain the individual_name model_index and rewards
        step_results = self.pool.map(self.generation_forward, inputs, chunksize=chunksize)
        # Rank the resualt
        results = rank(results)
        mutate_results = self.pool.map(self.mutate, results)

def main(args):
    em = EvolutionManager(args)
    em()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, default="CartPole-v0")
    parser.add_argument('-s', '--step_per_generation', type=int, default=100)
    parser.add_argument('-w', '--worker', type=int, default=4)
    parser.add_argument('-g', '--generation', type=int, default=10)
    parser.add_argument('-p', '--population', type=int, help='population size', default=4)
    args = parser.parse_args()
    main(args)
    