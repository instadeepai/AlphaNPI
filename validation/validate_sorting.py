from environments.list_env import ListEnv, ListEnvEncoder
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
    #default_load_path = '../models/list_npi_2019_5_16-10_19_59-1.pth'
    default_load_path = '../models/list_npi_2019_5_13-9_26_38-1.pth'

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
        results_save_path = '../results/validation_list_npi_{}-{}.txt'.format(date_time, seed)

    # Load environment constants
    env_tmp = ListEnv(length=5, encoding_dim=conf.encoding_dim)
    num_programs = env_tmp.get_num_programs()
    num_non_primary_programs = env_tmp.get_num_non_primary_programs()
    observation_dim = env_tmp.get_observation_dim()
    programs_library = env_tmp.programs_library

    # Load Alpha-NPI policy
    encoder = ListEnvEncoder(env_tmp.get_observation_dim(), conf.encoding_dim)
    indices_non_primary_programs = [p['index'] for _, p in programs_library.items() if p['level'] > 0]
    policy = Policy(encoder, conf.hidden_size, num_programs, num_non_primary_programs, conf.program_embedding_dim,
                    conf.encoding_dim, indices_non_primary_programs, conf.learning_rate)

    policy.load_state_dict(torch.load(load_path))


    # Start validation
    if verbose:
        print('Start validation for model: {}'.format(load_path))

    if save_results:
        with open(results_save_path, "a+") as results_file:
            results_file.write('Validation on model: {}'.format(load_path) + ' \n')

    for len in [5, 10, 20, 60, 100]:

        mcts_rewards_normalized = []
        mcts_rewards = []
        network_only_rewards = []

        max_depth_dict = {1: 5, 2: 2 * len + 3, 3: 2 * len + 3}
        mcts_test_params = {'number_of_simulations': conf.number_of_simulations_for_validation,
                            'max_depth_dict': max_depth_dict, 'temperature': conf.temperature,
                            'c_puct': conf.c_puct, 'exploit': True, 'level_closeness_coeff': conf.level_closeness_coeff,
                            'gamma': conf.gamma}

        for _ in range(40):

            env = ListEnv(length=len, encoding_dim=conf.encoding_dim)
            bubblesort_index = env.programs_library['BUBBLESORT']['index']

            # Test with mcts
            mcts = MCTS(policy, env, bubblesort_index, **mcts_test_params)
            res = mcts.sample_execution_trace()
            mcts_reward = res[7]
            mcts_rewards.append(mcts_reward)
            if mcts_reward > 0:
                mcts_rewards_normalized.append(1.0)
            else:
                mcts_rewards_normalized.append(0.0)

            # Test with network alone
            network_only = NetworkOnly(policy, env, max_depth_dict)
            netonly_reward, _ = network_only.play(bubblesort_index)
            network_only_rewards.append(netonly_reward)

        mcts_rewards_normalized_mean = np.mean(np.array(mcts_rewards_normalized))
        mcts_rewards_mean = np.mean(np.array(mcts_rewards))
        network_only_rewards_mean = np.mean(np.array(network_only_rewards))

        if verbose:
            print('Length: {}, mcts mean reward: {}, mcts mean normalized reward: {}, '
                  'network only mean reward: {}'.format(len, mcts_rewards_mean, mcts_rewards_normalized_mean,
                                                        network_only_rewards_mean))

        if save_results:
            str = 'Length: {}, mcts mean reward: {}, mcts mean normalized reward: {}, ' \
                  'network only mean reward: {}'.format(len, mcts_rewards_mean, mcts_rewards_normalized_mean,
                                                        network_only_rewards_mean)
            with open(results_save_path, "a+") as results_file:
                results_file.write(str + ' \n')
