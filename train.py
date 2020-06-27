# -*- coding: utf-8 -*-
# https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease/WANN
# https://github.com/atgambardella/pytorch-es
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
import argparse
import os
import subprocess
import sys

from mpi4py import MPI


def mpi_fork(n):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    (from https://github.com/garymcintire/mpi_util/)
    """

    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            IN_MPI="1"
        )
        print(["mpirun", "-np", str(n), sys.executable] + sys.argv)
        subprocess.check_call(["mpirun", "-np", str(n), sys.executable]
                              + ['-u'] + sys.argv, env=env)
        return "parent"
    else:
        return "child"


def main(args):
    rank = MPI.COMM_WORLD.Get_rank()
    if (rank == 0):
        master()
    else:
        slave()


def master():
  global fileName, hyp
  data = DataGatherer(fileName, hyp)
  wann = Wann(hyp)

  for gen in range(hyp['maxGen']):        
    pop = wann.ask()            # Get newly evolved individuals from WANN  
    reward = batchMpiEval(pop)  # Send pop to evaluate
    wann.tell(reward)           # Send fitness to WANN    

    data = gatherData(data,wann,gen,hyp)
    print(gen, '\t - \t', data.display())

  # Clean up and data gathering at end of run
  data = gatherData(data,wann,gen,hyp,savePop=True)
  data.save()
  data.savePop(wann.pop,fileName)
  stopAllWorkers()


def slave():
  """Evaluation process: evaluates networks sent from master process. 

  PseudoArgs (recieved from master):
    wVec   - (np_array) - weight matrix as a flattened vector
             [1 X N**2]
    n_wVec - (int)      - length of weight vector (N**2)
    aVec   - (np_array) - activation function of each node 
             [1 X N]    - stored as ints, see applyAct in ann.py
    n_aVec - (int)      - length of activation vector (N)
    seed   - (int)      - random seed (for consistency across workers)

  PseudoReturn (sent to master):
    result - (float)    - fitness value of network
  """  
  global hyp  
  task = Task(games[hyp['task']], nReps=hyp['alg_nReps'])

  # Evaluate any weight vectors sent this way
  while True:
    n_wVec = comm.recv(source=0,  tag=1)# how long is the array that's coming?
    if n_wVec > 0:
      wVec = np.empty(n_wVec, dtype='d')# allocate space to receive weights
      comm.Recv(wVec, source=0,  tag=2) # recieve weights

      n_aVec = comm.recv(source=0,tag=3)# how long is the array that's coming?
      aVec = np.empty(n_aVec, dtype='d')# allocate space to receive activation
      comm.Recv(aVec, source=0,  tag=4) # recieve it

      seed = comm.recv(source=0, tag=5) # random seed as int

      result = task.getDistFitness(wVec,aVec,hyp,seed=seed) # process it

      comm.Send(result, dest=0) # send it back

    if n_wVec < 0: # End signal recieved
      print('Worker # ', rank, ' shutting down.')
      break

if __name__ == "__main__":
    # Parse argument.
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--windows', type=int, help="TODO", default=1000)
    parser.add_argument('-n', '--num_worker', type=int, help='number of cores to use', default=4)
    args = parser.parse_args()

    # Use MPI if parallel
    if "parent" == mpi_fork(args.num_worker+1): os._exit(0)
    main(args)