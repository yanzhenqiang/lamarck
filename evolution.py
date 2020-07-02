
import numpy as np
import torch
import torch.multiprocessing

from ./model import PolicyNet, EvaluateNet

os.environ["OMP_NUM_THREADS"] = "1"

def es_params(self):
    return [(k, v) for k, v in zip(self.state_dict().keys(),
                                    self.state_dict().values())]

def perturb_model(args, model, random_seed, env):
    new_model.load_state_dict(model.state_dict())

def evolution_net_mutate(self, model, perturb):
    # TODO:seed
    if perturb:
        np.random.seed(1)
        for v in model.state_dict().values():
            eps = np.random.normal(0, 1, v.size())
            v += torch.from_numpy(0.05*eps).float()

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def generation_forward(policy_net, evolution_net, env_name, steps):
    env = gym.make(env).unwrapped
    opt = torch.optim.Adam(policy_net.parameters(), lr=1e-4, betas=(0.9, 0.999))
    gamma = 0.9
    r_sum = 0.
    while step < steps:
        s = self.env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        while True:
            if self.name == 'TODO':
                self.env.render()
            a = self.lnet.choose_action(v_wrap(s[None, :]))
            s_, r, done, _ = self.env.step(a)
            if done: r = -1
            r_sum += r
            buffer_a.append(a)
            buffer_s.append(s)
            buffer_r.append(r)

            if step % 10 == 0 or done:
                if done:
                    v_s_ = 0.
                else:
                    v_s_ = policy_net.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]
                buffer_v_target = []
                for r in br[::-1]:
                    v_s_ = r + gamma * v_s_
                    buffer_v_target.append(v_s_)
                buffer_v_target.reverse()

                loss = policy_net.loss_func(
                    v_wrap(np.vstack(bs)),
                    v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
                    v_wrap(np.array(buffer_v_target)[:, None]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                buffer_s, buffer_a, buffer_r = [], [], []
            if done:
                break
            s = s_
            step += 1
    return r_sum

class EvolutionManager(object):
    
    def __init__(self, args):
        self.args = args
        self.pool = multiprocessing.Pool(args.worker)
        env = gym.make(args.env)
        o_dim = env.observation_space.shape[0]
        s_dim = 128
        a_dim = env.action_space.n

        self.policy_nets = []
        self.evaluate_nets = []
        self.env_names = []
        self.steps = []

        self.perturb = []
        for i in range(args.poputation):
            policy_net = PolicyNet(o_dim, s_dim, a_dim)
            policy_net.share_memory()
            self.policy_nets.append(policy_net)
            evaluate_net = EvaluateNet(o_dim, s_dim, a_dim)
            evaluate_net.share_memory()
            self.evaluate_nets.append(evaluate_net)
            self.env_names.append(self.args.env_name)
            self.steps.append(self.args.step_per_generation)

    def __call__(self, generation_forward_fn, evolution_net_mutate, chunksize=1):
        # rewards should contain the individual_name model_index and rewards
        for _ in range(self.args.generation):
            step_results = self.pool.map(generation_forward_fn, self.policy_net, self.evolution_net, self.env_name, self.steps, chunksize=chunksize)
            # Rank the resualt
            # results = rank(results)
            for i in range(args.population):
                if step_result[i]:
                    self.perturb.append(True)
            mutate_results = self.pool.map(evolution_net_mutate, results)

def main(args):
    em = EvolutionManager(args)
    em(generation_forward, evolution_net_mutate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_name', type=str, default="CartPole-v0")
    parser.add_argument('-s', '--step_per_generation', type=int, default=100)
    parser.add_argument('-w', '--worker', type=int, default=4)
    parser.add_argument('-g', '--generation', type=int, default=10)
    parser.add_argument('-p', '--population', type=int, help='population size', default=4)
    args = parser.parse_args()
    main(args)
    