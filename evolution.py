
def es_params(self):
    """
    The params that should be trained by ES (all of them)
    """
    return [(k, v) for k, v in zip(self.state_dict().keys(),
                                    self.state_dict().values())]

def perturb_model(args, model, random_seed, env):
    """
    Modifies the given model with a pertubation of its parameters,
    as well as the negative perturbation, and returns both perturbed
    models.
    """
    new_model = ES(env.observation_space.shape[0],
                   env.action_space, args.small_net)
    anti_model = ES(env.observation_space.shape[0],
                    env.action_space, args.small_net)
    new_model.load_state_dict(model.state_dict())
    anti_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (k, v), (anti_k, anti_v) in zip(new_model.es_params(),
                                        anti_model.es_params()):
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(args.sigma*eps).float()
        anti_v += torch.from_numpy(args.sigma*-eps).float()
    return [new_model, anti_model]


import collections
import itertools
import multiprocessing

from ./model import PolicyNet, EvaluateNet

class EvolutionManager(object):
    
    def __init__(self, num_workers, num_population):
        self.pool = multiprocessing.Pool(num_workers)
        self.policy_nets = [PolicyNet(TODO) for i in range(num_workers)]
        self.evaluate_nets = [EvaluateNet(TODO) for i in range(num_population)]
    
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
    em = EvolutionManager()
    em()

if __name__ == "__main__":
    # Parse argument.
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--step_per_generation step', type=int, help="TODO", default=1000)
    parser.add_argument('-n', '--worker', type=int, help='number of cores to use', default=4)
    parser.add_argument('-g', '--generation_size', type=int, help='generation of evolution', default=100)
    parser.add_argument('-g', '--population_size', type=int, help='generation of population', default=8)
    args = parser.parse_args()
    main(args)
    