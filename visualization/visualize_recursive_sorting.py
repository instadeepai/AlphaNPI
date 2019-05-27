from environments.recursive_list_env import RecursiveListEnv, RecursiveListEnvEncoder
from core.policy import Policy
import core.config as conf
import torch
import time
from core.mcts import MCTS
from visualization.visualise_mcts import MCTSvisualiser
import time

if __name__ == "__main__":

    # Path to load policy
    #load_path = '../models/recursive_list_npi_2019_5_15-16_9_19-1.pth'
    load_path = '../models/recursive_list_npi_2019_5_10-15_9_57-1.pth'

    # Load environment constants
    env_tmp = RecursiveListEnv(length=5, encoding_dim=conf.encoding_dim)
    num_programs = env_tmp.get_num_programs()
    num_non_primary_programs = env_tmp.get_num_non_primary_programs()
    observation_dim = env_tmp.get_observation_dim()
    programs_library = env_tmp.programs_library

    # Load Alpha-NPI policy
    encoder = RecursiveListEnvEncoder(env_tmp.get_observation_dim(), conf.encoding_dim)
    indices_non_primary_programs = [p['index'] for _, p in programs_library.items() if p['level'] > 0]
    policy = Policy(encoder, conf.hidden_size, num_programs, num_non_primary_programs, conf.program_embedding_dim,
                    conf.encoding_dim, indices_non_primary_programs, conf.learning_rate)

    policy.load_state_dict(torch.load(load_path))

    # Prepare mcts params
    max_depth_dict = {1: 5, 2: 5, 3: 5}
    mcts_train_params = {'number_of_simulations': conf.number_of_simulations, 'max_depth_dict': max_depth_dict,
                         'temperature': conf.temperature, 'c_puct': conf.c_puct, 'exploit': False,
                         'level_closeness_coeff': conf.level_closeness_coeff, 'gamma': conf.gamma,
                         'use_dirichlet_noise': True, 'dir_noise': 0.5, 'dir_epsilon': 0.9}

    mcts_test_params = {'number_of_simulations': conf.number_of_simulations_for_validation,
                        'max_depth_dict': max_depth_dict, 'temperature': conf.temperature,
                        'c_puct': conf.c_puct, 'exploit': True, 'level_closeness_coeff': conf.level_closeness_coeff,
                        'gamma': conf.gamma}

    # Start debugging ...
    env = RecursiveListEnv(length=5, encoding_dim=conf.encoding_dim)
    reset_index = env.programs_library['RESET']['index']
    lshift_index = env.programs_library['LSHIFT']['index']
    bubble_index = env.programs_library['BUBBLE']['index']
    bubblesort_index = env.programs_library['BUBBLESORT']['index']

    mcts = MCTS(policy, env, bubblesort_index, **mcts_test_params)

    t_i = time.time()
    res = mcts.sample_execution_trace()
    t_f = time.time()

    res = mcts.sample_execution_trace()
    root_node, r = res[6], res[7]
    print('Duration: {} min, reward: {}'.format((t_f - t_i)/60, r))

    visualiser = MCTSvisualiser(env=env)
    visualiser.print_mcts(root_node=root_node, file_path='mcts.gv')

