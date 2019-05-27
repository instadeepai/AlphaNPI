# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

import numpy as np
import torch


class MCTS:
    """This class is used to perform a search over the state space for different paths by building
    a tree of visited states. Then this tree is used to get an estimation distribution of
    utility over actions.

    Args:
      policy: Policy to be used as a prior over actions given an state.
      c_puct: Constant that modifies the exploration-exploitation tradeoff of the MCTS algorithm.
      env: The environment considered.
      task_index: The index of the task (index of the corresponding program) we are trying to solve.
      number_of_simulations: The number of nodes that we will be visiting when building an MCTS tree.
      temperature: Another parameter that balances exploration-exploitation in MCTS by adding noise to the priors output by the search.
      max_depth_dict: Dictionary that maps a program level to the allowed number of actions to execute a program of that level
      use_dirichlet_noise: Boolean authorizes or not addition of dirichlet noise to prior during simulations to encourage exploration
      dir_epsilon: Proportion of the original prior distribution kept in the newly-updated prior distribution with dirichlet noise
      dir_noise: Parameter of the Dirichlet distribution
      exploit: Boolean if True leads to sampling from the mcts visit policy instead of taking the argmax
      gamma: discount factor, reward discounting increases with depth of trace
      save_sub_trees: Boolean to save in a node the sub-execution trace of a non-zero program
      recursion_depth: Recursion level of the calling tree
      max_recursion_depth: Max recursion level allowed
      qvalue_temperature: Induces tradeoff between mean qvalue and max qvalue when estimating Q in PUCT criterion
      recursive_penalty: Penalty applied to discounted reward if recursive program does not call itself
    """

    def __init__(self, policy, env, task_index, level_closeness_coeff=1.0,
                 c_puct=1.0, number_of_simulations=100, max_depth_dict={1: 5, 2: 50, 3: 150},
                 temperature=1.0, use_dirichlet_noise=False,
                 dir_epsilon=0.25, dir_noise=0.03, exploit=False, gamma=0.97, save_sub_trees=False,
                 recursion_depth=0, max_recursion_depth=500, qvalue_temperature=1.0, recursive_penalty=0.9):

        self.policy = policy
        self.c_puct = c_puct
        self.level_closeness_coeff = level_closeness_coeff
        self.env = env
        self.task_index = task_index
        self.task_name = env.get_program_from_index(task_index)
        self.recursive_task = env.programs_library[self.task_name]['recursive']
        self.recursive_penalty = recursive_penalty
        self.number_of_simulations = number_of_simulations
        self.temperature = temperature
        self.max_depth_dict = max_depth_dict
        self.dirichlet_noise = use_dirichlet_noise
        self.dir_epsilon = dir_epsilon
        self.dir_noise = dir_noise
        self.exploit = exploit
        self.gamma = gamma
        self.save_sub_trees = save_sub_trees
        self.recursion_depth = recursion_depth
        self.max_recursion_depth = max_recursion_depth
        self.qvalue_temperature = qvalue_temperature

        # record if all sub-programs executed correctly (useful only for programs of level > 1)
        self.clean_sub_executions = True

        # recursive trees parameters
        self.sub_tree_params = {'number_of_simulations': 5, 'max_depth_dict': self.max_depth_dict,
            'temperature': self.temperature, 'c_puct': self.c_puct, 'exploit': True,
            'level_closeness_coeff': self.level_closeness_coeff, 'gamma': self.gamma,
            'save_sub_trees': self.save_sub_trees, 'recursion_depth': recursion_depth+1}


    def _expand_node(self, node):
        """Used for previously unvisited nodes. It evaluates each of the possible child and
        initializes them with a score derived from the prior output by the policy network.

        Args:
          node: Node to be expanded

        Returns:
          node now expanded, value, hidden_state, cell_state

        """
        program_index, observation, env_state, h, c, depth = (
            node["program_index"],
            node["observation"],
            node["env_state"],
            node["h_lstm"],
            node["c_lstm"],
            node["depth"]
        )

        with torch.no_grad():
            mask = self.env.get_mask_over_actions(program_index)
            priors, value, new_h, new_c = self.policy.forward_once(observation, program_index, h, c)
            # mask actions
            priors = priors * torch.FloatTensor(mask)
            priors = torch.squeeze(priors)
            priors = priors.cpu().numpy()
            if self.dirichlet_noise:
                priors = (1 - self.dir_epsilon) * priors + self.dir_epsilon * np.random.dirichlet([self.dir_noise] * priors.size)

        # Initialize its children with its probability of being chosen
        for prog_index in [prog_idx for prog_idx, x in enumerate(mask) if x == 1]:
            new_child = {
                "parent": node,
                "childs": [],
                "visit_count": 0.0,
                "total_action_value": [],
                "prior": float(priors[prog_index]),
                "program_from_parent_index": prog_index,
                "program_index": program_index,
                "observation": observation,
                "env_state": env_state,
                "h_lstm": new_h.clone(),
                "c_lstm": new_c.clone(),
                "selected": False,
                "depth": depth + 1
            }
            node["childs"].append(new_child)
        # This reward will be propagated backwards through the tree
        value = float(value)
        return node, value, new_h.clone(), new_c.clone()

    def _compute_q_value(self, node):
        if node["visit_count"] > 0.0:
            values = torch.FloatTensor(node['total_action_value'])
            softmax = torch.exp(self.qvalue_temperature * values)
            softmax = softmax / softmax.sum()
            q_val_action = float(torch.dot(softmax, values))
        else:
            q_val_action = 0.0
        return q_val_action

    def _estimate_q_val(self, node):
        """Estimates the Q value over possible actions in a given node, and returns the action
        and the child that have the best estimated value.

        Args:
          node: Node to evaluate its possible actions.

        Returns:
          best child found from this node.

        """
        best_child = None
        best_val = -np.inf
        # Iterate all the children to fill up the node dict and estimate Q val.
        # Then track the best child found according to the Q value estimation
        for child in node["childs"]:
            if child["prior"] > 0.0:
                q_val_action = self._compute_q_value(child)

                action_utility = (self.c_puct * child["prior"] * np.sqrt(node["visit_count"])
                                  * (1.0 / (1.0 + child["visit_count"])))
                q_val_action += action_utility
                parent_prog_lvl = self.env.programs_library[self.env.idx_to_prog[node['program_index']]]['level']
                action_prog_lvl = self.env.programs_library[self.env.idx_to_prog[child['program_from_parent_index']]]['level']

                if parent_prog_lvl == action_prog_lvl:
                    # special treatment for calling the same program
                    action_level_closeness = self.level_closeness_coeff * np.exp(-1)
                elif action_prog_lvl > -1:
                    action_level_closeness = self.level_closeness_coeff * np.exp(-(parent_prog_lvl - action_prog_lvl))
                else:
                    # special treatment for STOP action
                    action_level_closeness = self.level_closeness_coeff * np.exp(-1)

                q_val_action += action_level_closeness
                if q_val_action > best_val:
                    best_val = q_val_action
                    best_child = child

        return best_child

    def _sample_policy(self, root_node):
        """Sample an action from the policies and q_value distributions that were previously sampled.

        Args:
          root_node: Node to choose the best action from. It should be the root node of the tree.

        Returns:
          Tuple containing the sampled action and the probability distribution build normalizing visits_policy.
        """
        visits_policy = []
        for child in root_node["childs"]:
            if child["prior"] > 0.0:
                visits_policy.append([child['program_from_parent_index'], child["visit_count"]])

        mcts_policy = torch.zeros(1, self.env.get_num_programs())
        for prog_index, visit in visits_policy:
            mcts_policy[0, prog_index] = visit

        if self.exploit:
            mcts_policy = mcts_policy / mcts_policy.sum()
            return mcts_policy / mcts_policy.sum(), int(torch.argmax(mcts_policy))
        else:
            mcts_policy = torch.pow(mcts_policy, self.temperature)
            mcts_policy = mcts_policy / mcts_policy.sum()
            return mcts_policy, int(torch.multinomial(mcts_policy, 1)[0, 0])

    def _run_simulation(self, node):
        """Run one simulation in tree. This function is recursive.

        Args:
          node: root node to run the simulation from
          program_index: index of the current calling program

        Returns:
            (if the max depth has been reached or not, if a node has been expanded or not, node reached at the end of the simulation)

        """

        stop = False
        max_depth_reached = False
        max_recursion_reached = False
        has_expanded_a_node = False
        value = None
        program_level = self.env.get_program_level_from_index(node['program_index'])

        while not stop and not max_depth_reached and not has_expanded_a_node and self.clean_sub_executions and not max_recursion_reached:

            if node['depth'] >= self.max_depth_dict[program_level]:
                max_depth_reached = True

            elif len(node['childs']) == 0:
                _, value, state_h, state_c = self._expand_node(node)
                has_expanded_a_node = True

            else:
                node = self._estimate_q_val(node)
                program_to_call_index = node['program_from_parent_index']
                program_to_call = self.env.get_program_from_index(program_to_call_index)

                if program_to_call_index == self.env.programs_library['STOP']['index']:
                    stop = True

                elif self.env.programs_library[program_to_call]['level'] == 0:
                    observation = self.env.act(program_to_call)
                    node['observation'] = observation
                    node['env_state'] = self.env.get_state()

                else:
                    # check if call corresponds to a recursive call
                    if program_to_call_index == self.task_index:
                        self.recursive_call = True
                    # if never been done, compute new tree to execute program
                    if node['visit_count'] == 0.0:

                        if self.recursion_depth >= self.max_recursion_depth:
                            max_recursion_reached = True
                            continue

                        sub_mcts_init_state = self.env.get_state()
                        sub_mcts = MCTS(self.policy, self.env, program_to_call_index, **self.sub_tree_params)
                        sub_trace = sub_mcts.sample_execution_trace()
                        sub_task_reward, sub_root_node = sub_trace[7], sub_trace[6]

                        # if save sub tree is true, then store sub root node
                        if self.save_sub_trees:
                            node['sub_root_node'] = sub_root_node
                        # allows tree saving of first non zero program encountered

                        # check that sub tree executed correctly
                        self.clean_sub_executions &= (sub_task_reward > -1.0)
                        if not self.clean_sub_executions:
                            print('program {} did not execute correctly'.format(program_to_call))
                            self.programs_failed_indices.append(program_to_call_index)
                            #self.programs_failed_indices += sub_mcts.programs_failed_indices
                            self.programs_failed_initstates.append(sub_mcts_init_state)

                        observation = self.env.get_observation()
                    else:
                        self.env.reset_to_state(node['env_state'])
                        observation = self.env.get_observation()

                    node['observation'] = observation
                    node['env_state'] = self.env.get_state()

        return max_depth_reached, has_expanded_a_node, node, value

    def _play_episode(self, root_node):
        """Performs an MCTS search using the policy network as a prior and returns a sequence of improved decisions.

        Args:
          root_node: Root node of the tree.

        Returns:
            (Final node reached at the end of the episode, boolean stating if the max depth allowed has been reached).

        """
        stop = False
        max_depth_reached = False

        while not stop and not max_depth_reached and self.clean_sub_executions:

            program_level = self.env.get_program_level_from_index(root_node['program_index'])
            # tag node as from the final execution trace (for visualization purpose)
            root_node["selected"] = True

            if root_node['depth'] >= self.max_depth_dict[program_level]:
                max_depth_reached = True

            else:
                env_state = root_node["env_state"]

                # record obs, progs and lstm states only if they correspond to the current task at hand
                self.lstm_states.append((root_node['h_lstm'], root_node['c_lstm']))
                self.programs_index.append(root_node['program_index'])
                self.observations.append(root_node['observation'])
                self.previous_actions.append(root_node['program_from_parent_index'])
                self.rewards.append(None)

                # Spend some time expanding the tree from your current root node
                for _ in range(self.number_of_simulations):
                    # run a simulation
                    self.recursive_call = False
                    simulation_max_depth_reached, has_expanded_node, node, value = self._run_simulation(root_node)

                    # get reward
                    if not simulation_max_depth_reached and not has_expanded_node:
                        # if node corresponds to end of an episode, backprogagate real reward
                        reward = self.env.get_reward()
                        if reward > 0:
                            value = self.env.get_reward() * (self.gamma ** node['depth'])
                            if self.recursive_task and not self.recursive_call:
                                # if recursive task but do not called itself, add penalization
                                value -= self.recursive_penalty
                        else:
                            value = -1.0

                    elif simulation_max_depth_reached:
                        # if episode stops because the max depth allowed was reached, then reward = -1
                        value = -1.0

                    value = float(value)

                    # Propagate information backwards
                    while node["parent"] is not None:
                        node["visit_count"] += 1
                        node["total_action_value"].append(value)
                        node = node["parent"]
                    # Root node is not included in the while loop
                    self.root_node["total_action_value"].append(value)
                    self.root_node["visit_count"] += 1

                    # Go back to current env state
                    self.env.reset_to_state(env_state)

                # Sample next action
                mcts_policy, program_to_call_index = self._sample_policy(root_node)
                if program_to_call_index == self.task_index:
                    self.global_recursive_call = True

                # Set new root node
                root_node = [child for child in root_node["childs"] if child["program_from_parent_index"] == program_to_call_index][0]

                # Record mcts policy
                self.mcts_policies.append(mcts_policy)

                # Apply chosen action
                if program_to_call_index == self.env.programs_library['STOP']['index']:
                    stop = True
                else:
                    self.env.reset_to_state(root_node["env_state"])

        return root_node, max_depth_reached


    def sample_execution_trace(self):
        """
        Args:
          init_observation: initial observation before playing an episode

        Returns:
            (a sequence of (e_t, i_t), a sequence of probabilities over programs, a sequence of (h_t, c_t), if the maximum depth allowed has been reached)
        """

        # start the task
        init_observation = self.env.start_task(self.task_index)
        with torch.no_grad():
            state_h, state_c = self.policy.init_tensors()
            self.env_init_state = self.env.get_state()

            self.root_node = {
                "parent": None,
                "childs": [],
                "visit_count": 1,
                "total_action_value": [],
                "prior": None,
                "program_index": self.task_index,
                "program_from_parent_index": None,
                "observation": init_observation,
                "env_state": self.env_init_state,
                "h_lstm": state_h.clone(),
                "c_lstm": state_c.clone(),
                "depth": 0,
                "selected": True
            }

            # prepare empty lists to store trajectory
            self.programs_index = []
            self.observations = []
            self.previous_actions = []
            self.mcts_policies = []
            self.lstm_states = []
            self.rewards = []
            self.programs_failed_indices = []
            self.programs_failed_initstates = []

            self.global_recursive_call = False

            # play an episode
            final_node, max_depth_reached = self._play_episode(self.root_node)
            final_node['selected'] = True

        # compute final task reward (with gamma penalization)
        reward = self.env.get_reward()
        if reward > 0:
            task_reward = reward * (self.gamma**final_node['depth'])
            if self.recursive_task and not self.global_recursive_call:
                # if recursive task but do not called itself, add penalization
                task_reward -= self.recursive_penalty
        else:
            task_reward = -1

        # Replace None rewards by the true final task reward
        self.rewards = list(map(lambda x: torch.FloatTensor([task_reward]) if x is None else torch.FloatTensor([x]), self.rewards))

        # end task
        self.env.end_task()

        return self.observations, self.programs_index, self.previous_actions, self.mcts_policies, \
               self.lstm_states, max_depth_reached, self.root_node, task_reward, self.clean_sub_executions, self.rewards, \
               self.programs_failed_indices, self.programs_failed_initstates