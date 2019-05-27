from core.mcts import MCTS
import torch

class Trainer():
    """
    Trainer class. Used for a given environment to perform training and validation steps.
    """
    def __init__(self, environment, policy, replay_buffer, curriculum_scheduler, mcts_train_params,
                 mcts_test_params, num_validation_episodes, num_episodes_per_task, batch_size, num_updates_per_episode,
                 verbose=True):

        self.env = environment
        self.policy = policy
        self.buffer = replay_buffer
        self.curriculum_scheduler = curriculum_scheduler
        self.mcts_train_params = mcts_train_params
        self.mcts_test_params = mcts_test_params

        self.num_validation_episodes = num_validation_episodes
        self.num_episodes_per_task = num_episodes_per_task
        self.batch_size = batch_size
        self.num_updates_per_episode = num_updates_per_episode

        self.verbose = verbose


    def perform_validation_step(self, task_index):
        """
        Perform validation steps for the task from index task_index.

        Args:
            task_index: task index

        Returns:
            (rewards, traces lengths)

        """
        validation_rewards = []
        traces_lengths = []
        for _ in range(self.num_validation_episodes):
            # Start new episode
            mcts = MCTS(self.policy, self.env, task_index, **self.mcts_test_params)

            # Sample an execution trace with mcts using policy as a prior
            trace = mcts.sample_execution_trace()
            task_reward, trace_length, progs_failed_indices = trace[7], len(trace[3]), trace[10]

            validation_rewards.append(task_reward)
            traces_lengths.append(trace_length)
        return validation_rewards, traces_lengths, progs_failed_indices

    def play_iteration(self, task_index, verbose=False):
        """
        Play one training iteration, i.e. select a task, play episodes, store experience in buffer and sample batches
        to perform gradient descent on policy weights.

        """

        # Get new task to attempt
        task_name = self.env.get_program_from_index(task_index)
        if self.verbose:
            print('Attempt task {} for {} episodes'.format(task_name, self.num_episodes_per_task))

        # Start training on the task
        for episode in range(self.num_episodes_per_task):

            # Start new episode
            mcts = MCTS(self.policy, self.env, task_index, **self.mcts_train_params)

            # Sample an execution trace with mcts using policy as a prior
            res = mcts.sample_execution_trace()
            observations, prog_indices, previous_actions_indices, policy_labels, lstm_states, _, _, \
                task_reward, clean_sub_execution, rewards, programs_failed_indices, \
                programs_failed_initstates = res

            # record trace and store it in buffer only if no problem in sub-programs execution
            if clean_sub_execution:
                # Generates trace
                trace = list(zip(observations, prog_indices, lstm_states, policy_labels, rewards))
                # Append trace to buffer
                self.buffer.append_trace(trace)
            else:
                if self.verbose:
                    print("Trace has not been stored in buffer.")

                # Decrease statistics of programs that failed
                #for idx in programs_failed_indices:
                    #self.curriculum_scheduler.update_statistics(idx, torch.FloatTensor([0.0]))


            # Train policy on batch
            if self.buffer.get_memory_length() > self.batch_size:
                for _ in range(self.num_updates_per_episode):
                    batch = self.buffer.sample_batch(self.batch_size)
                    if batch is not None:
                        self.policy.train_on_batch(batch)
            if verbose:
                print("Done episode {}/{}".format(episode + 1, self.num_episodes_per_task))

    def perform_validation(self):
        """
        Perform validation for all the tasks and update curriculum scheduelr statistics.
        """
        if self.verbose:
            print("Start validation .....")
        for idx in self.curriculum_scheduler.get_tasks_of_maximum_level():
            # Evaluate performance on task idx
            v_rewards, v_lengths, programs_failed_indices = self.perform_validation_step(idx)
            # Update curriculum statistics
            self.curriculum_scheduler.update_statistics(idx, v_rewards)

            # Decrease statistics of programs that failed
            #for idx_ in programs_failed_indices:
                #self.curriculum_scheduler.update_statistics(idx_, torch.FloatTensor([0.0]))