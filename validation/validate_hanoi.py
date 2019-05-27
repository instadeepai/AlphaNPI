from environments.hanoi_env import HanoiEnv, HanoiEnvEncoder
from core.policy import Policy
from core.network_only import NetworkOnly
import core.config as conf
import torch
import argparse
from core.mcts import MCTS
import numpy as np
import time

if __name__ == "__main__":

    # Path to load policy
    #default_load_path = '../models/hanoi_npi_2019_5_17-11_45_25-1.pth'
    default_load_path = '../models/hanoi_npi_2019_5_17-11_45_25-1.pth'

    # Get command line params
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help='random seed', default=1, type=int)
    parser.add_argument("--load-path", help='path to model to validate', default=default_load_path)
    parser.add_argument('--verbose', help='print training monitoring in console', action='store_true')
    parser.add_argument('--save-results', help='save training progress in .txt file', action='store_true')
    parser.add_argument('--num-cpus', help='number of cpus to use', default=8, type=int)
    args = parser.parse_args()

    # Get arguments
    seed = args.seed
    verbose = args.verbose
    save_results = args.save_results
    load_path = args.load_path
    num_cpus = args.num_cpus

    # Set number of cpus used
    torch.set_num_threads(num_cpus)

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if save_results:
        # get date and time
        ts = time.localtime(time.time())
        date_time = '{}_{}_{}-{}_{}_{}'.format(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])
        results_save_path = '../results/validation_hanoi_npi_{}-{}.txt'.format(date_time, seed)
        results_file = open(results_save_path, 'w')

    # Load environment constants
    env_tmp = HanoiEnv(n=5, encoding_dim=conf.encoding_dim)
    num_programs = env_tmp.get_num_programs()
    num_non_primary_programs = env_tmp.get_num_non_primary_programs()
    observation_dim = env_tmp.get_observation_dim()
    programs_library = env_tmp.programs_library

    # Load Alpha-NPI policy
    encoder = HanoiEnvEncoder(env_tmp.get_observation_dim(), conf.encoding_dim)
    indices_non_primary_programs = [p['index'] for _, p in programs_library.items() if p['level'] > 0]
    policy = Policy(encoder, conf.hidden_size, num_programs, num_non_primary_programs, conf.program_embedding_dim,
                    conf.encoding_dim, indices_non_primary_programs, conf.learning_rate)

    policy.load_state_dict(torch.load(load_path))


    # Start Validation
    if verbose:
        print('Start validation for model: {}'.format(load_path))

    if save_results:
        results_file.write('Validation on model: {}'.format(load_path) + ' \n')

    t_i = time.time()
    for n in [2, 3, 4, 5, 10, 12]:

        mcts_rewards_normalized = []
        mcts_rewards = []
        network_only_rewards = []

        max_depth_dict = {1: 8}
        mcts_test_params = {'number_of_simulations': conf.number_of_simulations_for_validation,
                            'max_depth_dict': max_depth_dict, 'temperature': conf.temperature,
                            'c_puct': conf.c_puct, 'exploit': True, 'level_closeness_coeff': conf.level_closeness_coeff,
                            'gamma': conf.gamma}

        for _ in range(1):

            env = HanoiEnv(n=n, encoding_dim=conf.encoding_dim)
            hanoi_index = env.programs_library['HANOI']['index']

            # Test with MCTS
            mcts = MCTS(policy, env, hanoi_index, **mcts_test_params)
            res = mcts.sample_execution_trace()
            mcts_reward = res[7]
            mcts_rewards.append(mcts_reward)
            if mcts_reward > 0:
                mcts_rewards_normalized.append(1.0)
            else:
                mcts_rewards_normalized.append(0.0)

            env = HanoiEnv(n=n, encoding_dim=conf.encoding_dim)
            hanoi_index = env.programs_library['HANOI']['index']

            # Test with network alone
            network_only = NetworkOnly(policy, env, max_depth_dict)
            netonly_reward, _ = network_only.play(hanoi_index)
            network_only_rewards.append(netonly_reward)

        mcts_rewards_normalized_mean = np.mean(np.array(mcts_rewards_normalized))
        mcts_rewards_mean = np.mean(np.array(mcts_rewards))
        network_only_rewards_mean = np.mean(np.array(network_only_rewards))

        if verbose:
            print('n: {}, mcts mean reward: {}, mcts mean normalized reward: {}, '
                  'network only mean reward: {}'.format(n, mcts_rewards_mean, mcts_rewards_normalized_mean,
                                                        network_only_rewards_mean))

        if save_results:
            str = 'n: {}, mcts mean reward: {}, mcts mean normalized reward: {}, ' \
                  'network only mean reward: {}'.format(n, mcts_rewards_mean, mcts_rewards_normalized_mean,
                                                        network_only_rewards_mean)
            results_file.write(str + ' \n')

    t_f = time.time()
    if verbose:
        print('End of Validation!')
        duration = t_f - t_i
        print('Duration: {} minutes'.format(duration / 60))

    if save_results:
        results_file.close()



