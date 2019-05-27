from environments.list_env import ListEnv, ListEnvEncoder
from core.policy import Policy
import core.config as conf
import torch
from core.mcts import MCTS
from visualization.visualise_mcts import MCTSvisualiser

if __name__ == "__main__":

    # Path to load policy
    load_path = "../models/list_npi_nohierarchy_2019_5_21-15_28_47-3_max_4_val_4.pth"

    # Load environment constants
    env_tmp = ListEnv(length=5, encoding_dim=conf.encoding_dim, hierarchy=False)
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

    # Prepare mcts params
    length = 3
    max_depth_dict = {1: 6 * length * length}
    mcts_train_params = {'number_of_simulations': conf.number_of_simulations, 'max_depth_dict': max_depth_dict,
                         'temperature': conf.temperature, 'c_puct': conf.c_puct, 'exploit': False,
                         'level_closeness_coeff': conf.level_closeness_coeff, 'gamma': conf.gamma,
                         'use_dirichlet_noise': True, 'dir_noise': 0.5, 'dir_epsilon': 0.9}

    mcts_test_params = {'number_of_simulations': conf.number_of_simulations_for_validation,
                        'max_depth_dict': max_depth_dict, 'temperature': conf.temperature,
                        'c_puct': conf.c_puct, 'exploit': True, 'level_closeness_coeff': conf.level_closeness_coeff,
                        'gamma': conf.gamma}

    # Start debugging ...
    env = ListEnv(length=length, encoding_dim=conf.encoding_dim, hierarchy=False)
    bubblesort_index = env.programs_library['BUBBLESORT']['index']

    mcts = MCTS(policy, env, bubblesort_index, **mcts_test_params)
    res = mcts.sample_execution_trace()
    root_node, r = res[6], res[7]
    print('reward: {}'.format(r))

    visualiser = MCTSvisualiser(env=env)
    visualiser.print_mcts(root_node=root_node, file_path='mcts.gv')

