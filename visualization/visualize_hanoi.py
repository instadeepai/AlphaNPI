from environments.hanoi_env import HanoiEnv, HanoiEnvEncoder
from core.policy import Policy
import core.config as conf
import torch
from core.mcts import MCTS
from visualization.visualise_mcts import MCTSvisualiser

if __name__ == "__main__":

    # Path to load policy
    load_path = '../models/hanoi_npi_2019_5_17-11_45_25-1.pth'

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

    # Prepare mcts params
    max_depth_dict = {1: 8}

    mcts_test_params = {'number_of_simulations': conf.number_of_simulations_for_validation,
                        'max_depth_dict': max_depth_dict, 'temperature': conf.temperature,
                        'c_puct': conf.c_puct, 'exploit': True, 'level_closeness_coeff': conf.level_closeness_coeff,
                        'gamma': conf.gamma}

    # Start debugging ...
    length = 6
    env = HanoiEnv(n=length, encoding_dim=conf.encoding_dim)
    hanoi_index = env.programs_library['HANOI']['index']

    mcts = MCTS(policy, env, hanoi_index, **mcts_test_params)
    res = mcts.sample_execution_trace()
    root_node, r = res[6], res[7]
    print('reward: {}'.format(r))

    visualiser = MCTSvisualiser(env=env)
    visualiser.print_mcts(root_node=root_node, file_path='mcts.gv')

