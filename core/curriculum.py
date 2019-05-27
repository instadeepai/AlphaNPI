import numpy as np
import torch
from torch.distributions.categorical import Categorical

class CurriculumScheduler():
    """Implements a curriculum sequencer which is used to decide what is the next task to attempt in
    the curriculum learning process.
    """
    def __init__(self, reward_threshold, num_non_primary_programs, programs_library,
                 moving_average=0.95, temperature=2.0):
        self.reward_threshold = reward_threshold
        self.programs_library = programs_library
        self.moving_average = moving_average
        self.temperature = temperature

        self.indices_non_primary_programs = [p['index'] for _,p in self.programs_library.items() if p['level']>0]
        self.non_primary_programs = dict((p_name, p) for p_name, p in  self.programs_library.items() if p['level']>0)
        self.relative_indices = dict((prog_idx, relat_idx) for relat_idx, prog_idx in enumerate(self.indices_non_primary_programs))
        self.relative_indices_inverted = dict((b,a) for a,b in self.relative_indices.items())

        self.maximum_level = 1
        self.tasks_average_rewards = np.zeros(num_non_primary_programs)
        # count number of time each task has been attempted ( used to compute running average reward)
        self.tasks_stats_updates = np.zeros(num_non_primary_programs)


    def get_tasks_of_maximum_level(self):
        """Returns the list of programs indices which levels are lower or equals to attribute maximum_level.

        Returns:
            the list of programs indices which levels are lower or equals to attribute maximum_level.
        """
        return [key['index'] for _,key in self.non_primary_programs.items() if key['level'] <= self.maximum_level]

    def get_tasks_from_max_level(self):
        """Returns the list of programs indices which levels are equals to the attribute maximum_level.

        Returns:
            the list of programs indices which levels are lower or equals to attribute maximum_level.
        """
        return [key['index'] for _,key in self.non_primary_programs.items() if key['level'] == self.maximum_level]


    def get_next_task_index(self):
        """Sample next task to attempt according to tasks average rewards.

        Returns:
            Next task index.

        """
        index_mask = self.get_tasks_of_maximum_level()
        scores = 1.0 - self.tasks_average_rewards

        probs = torch.zeros(scores.shape[0])
        for index in index_mask:
            relat_index = self.relative_indices[index]
            probs[relat_index] = np.exp(self.temperature * scores[relat_index])

        # compute softmax of scores
        probs /= probs.sum()
        # print proba
        res = 'sample task with probabilities: '.format(self.maximum_level)
        for prog_name, prog in self.non_primary_programs.items():
            res += ' %s:%.2f ,' % (prog_name, probs[self.relative_indices[prog['index']]])
        print(res)
        # sample next task
        return self.relative_indices_inverted[int(torch.multinomial(probs, 1)[0])]


    def print_statistics(self):
        '''
        Print current learning statistics (in terms of rewards).
        '''
        res = 'max level: {}, mean rewards:'.format(self.maximum_level)
        for prog_name, prog in self.non_primary_programs.items():
            res += ' %s:%.3f ,' % (prog_name, self.tasks_average_rewards[self.relative_indices[prog['index']]])
        print(res)

    def get_statistic(self, task_index):
        """
        Returns the current average reward on the task.

        Args:
            task_index: task to know the statistic

        Returns:
            average reward on this task
        """
        return self.tasks_average_rewards[self.relative_indices[task_index]]

    def update_statistics(self, task_index, rewards):
        """This function must be called every time a new task has been attempted by NPI. It is used to
        update tasks average rewards as well as the maximum_task level.

        Args:
          task_index: the task that has been attempted
          reward: the reward obtained at the end of the task
          rewards: 

        """
        # Update task average reward
        for reward in rewards:
            # all non-zero rewards are considered to be 1.0 in the curriculum scheduler
            if reward > 0.0:
                reward = 1.0
            else:
                reward = 0.0

            self.tasks_average_rewards[self.relative_indices[task_index]] = self.moving_average*self.tasks_average_rewards[self.relative_indices[task_index]] + (1-self.moving_average)*reward

        # Determine if the maximum_level should be increased or not
        possible_indices = self.get_tasks_from_max_level()
        possible_relative_indices = [self.relative_indices[idx] for idx in possible_indices]
        min_reward = np.min(self.tasks_average_rewards[possible_relative_indices])
        if min_reward >= self.reward_threshold:
            self.maximum_level += 1

