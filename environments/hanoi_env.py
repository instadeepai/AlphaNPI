import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from environments.environment import Environment

class EmptyTowerException(Exception):
    """Error: Tried to remove a disk from and empty pillar"""
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class InvertedTowerException(Exception):
    """Error: Placed large disk on small disk"""
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class HanoiEnvEncoder(nn.Module):
    '''
    Implement an encoder (f_enc) specific to the List environment. It encodes observations e_t into
    vectors s_t of size D = encoding_dim.
    '''
    def __init__(self, observation_dim, encoding_dim):
        super(HanoiEnvEncoder, self).__init__()
        self.l1 = nn.Linear(observation_dim, 100)
        self.l2 = nn.Linear(100, encoding_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x


class HanoiEnv(Environment):
    """
    Class that represents a tower of hanoi environment.
    The board is represented as a tuple of pillars
    Each pillar is represented as a list of disks
    Each disk is represented as a number (it's width).
    """

    def __init__(self, n=5, encoding_dim=32):
        #MAX_N = 10
        #assert n < MAX_N, 'cant consider hanoi towers with n larger than {}'.format(MAX_N)
        assert n > 0, 'cant consider hanoi puzzle with a non positive number of towers'
        self.n = n
        self.init_n = n
        self.pillars = ([], [], [])
        self.roles = []
        self.init_roles = []
        self.encoding_dim = encoding_dim # todo: check if actually used

        self.programs_library = {
                                 'SWAP_S_A': {'level': 0, 'recursive': False},
                                 'SWAP_A_T': {'level': 0, 'recursive': False},
                                 'MOVE_DISK': {'level': 0, 'recursive': False},
                                 'STOP': {'level': -1, 'recursive': False},
                                 'HANOI': {'level': 1, 'recursive': True}}

        for i, key in enumerate(sorted(list(self.programs_library.keys()))):
            self.programs_library[key]['index'] = i

        self.prog_to_func = {
                             'SWAP_S_A': self._swap_s_a,
                             'SWAP_A_T': self._swap_a_t,
                             'MOVE_DISK': self._move_disk,
                             'STOP': self._stop
                            }

        self.prog_to_precondition = {
            'SWAP_S_A': self._swap_s_a_precondition,
            'SWAP_A_T': self._swap_a_t_precondition,
            'MOVE_DISK': self._move_disk_precondition,
            'STOP': self._stop_precondition,
            'HANOI': self._hanoi_precondition}

        self.prog_to_postcondition = {'HANOI': self._hanoi_postcondition}

        super(HanoiEnv, self).__init__(self.programs_library, self.prog_to_func,
                                               self.prog_to_precondition, self.prog_to_postcondition)

    def reset_env(self):
        """Reset the environment. The list are values are draw randomly. The pointers are initialized at position 0
        (at left position of the list).

        """
        self.n = self.init_n

        self.roles = ['source', 'auxiliary', 'target']
        random.shuffle(self.roles)
        self.init_roles_stack = [self.roles.copy()]

        src_pos = self.roles.index('source')
        self.pillars = ([], [], [])
        for i in range(1, self.n+1):
            self.pillars[src_pos].append(self.n - i)
        self.has_been_reset = True

    def _incr_n(self):
        assert self._incr_n_precondition(), 'precondition not verified'
        self.n += 1

    def _incr_n_precondition(self):
        return self.n < self.init_n

    def _decr_n(self):
        assert self._decr_n_precondition(), 'precondition not verified'
        self.n -= 1

    def _decr_n_precondition(self):
        return self.n > 1

    def _stop(self):
        assert self._stop_precondition(), 'precondition not verified'
        pass

    def _stop_precondition(self):
        return True

    def _swap_s_a(self):
        assert self._swap_s_a_precondition(), 'precondition not verified'
        src_pos, aux_pos = self.roles.index('source'), self.roles.index('auxiliary')
        self.roles[src_pos], self.roles[aux_pos] = self.roles[aux_pos], self.roles[src_pos]

    def _swap_s_a_precondition(self):
        return self.n > 1

    def _swap_s_t(self):
        assert self._swap_s_t_precondition(), 'precondition not verified'
        src_pos, targ_pos = self.roles.index('source'), self.roles.index('target')
        self.roles[src_pos], self.roles[targ_pos] = self.roles[targ_pos], self.roles[src_pos]

    def _swap_s_t_precondition(self):
        return self.n > 1

    def _swap_a_t(self):
        assert self._swap_a_t_precondition(), 'precondition not verified'
        aux_pos, targ_pos = self.roles.index('auxiliary'), self.roles.index('target')
        self.roles[aux_pos], self.roles[targ_pos] = self.roles[targ_pos], self.roles[aux_pos]

    def _swap_a_t_precondition(self):
        return self.n > 1

    def _pop(self, i):
        """Take a disk off the top of pillars[i] and return it"""
        if len(self.pillars[i]) > 0:
            return self.pillars[i].pop()
        else:
            raise EmptyTowerException("Tried to pull a disk off a pillar which is empty")

    def _push(self, i, disk):
        """Put a disk on top of pillars[i]"""
        if len(self.pillars[i]) == 0 or self.pillars[i][-1] > disk:
            self.pillars[i].append(disk)
        else:
            raise InvertedTowerException("Tried to put larger disk on smaller disk")

    def _move_disk(self):
        assert self._move_disk_precondition(), 'precondition not verified'
        src_pos, targ_pos = self.roles.index('source'), self.roles.index('target')
        disk = self._pop(src_pos)
        self._push(targ_pos, disk)

    def _move_disk_precondition(self):
        src_pos, targ_pos = self.roles.index('source'), self.roles.index('target')
        is_move_possible = self._is_move_possible(self.pillars[src_pos], self.pillars[targ_pos])
        return is_move_possible

    def get_state(self):
        return (self.pillars[0].copy(), self.pillars[1].copy(), self.pillars[2].copy()), self.roles.copy(), self.n, \
               self._get_init_roles().copy()

    def reset_to_state(self, state):
        """

        Args:
          state: a given state of the environment
        reset the environment is the given state

        """
        self.pillars = (state[0][0].copy(), state[0][1].copy(), state[0][2].copy())
        self.roles = state[1].copy()
        self.n = state[2]
        self.init_roles_stack[-1] = state[3].copy()

    def compare_state(self, state1, state2):
        bool = True
        for i in range(3):
            bool &= (state1[0][i] == state2[0][i])
        bool &= (state1[1] == state2[1])
        bool &= (state1[2] == state2[2])
        bool &= (state1[3] == state2[3])
        return bool

    def get_state_str(self, state):
        """Return a text representation of the environment state"""
        roles = state[1]
        init_roles = state[3]
        n = state[2]
        updated_pillars = self._get_updated_pillars(state)
        str = ''
        for i in range(3):
            str += 'PILLAR {}, I: {}, C: {}: {},  '.format(i+1, init_roles[i], roles[i], updated_pillars[i])
        str += 'n = {}'.format(n)
        return str

    def _hanoi_precondition(self):
        hanoi_index = self.programs_library['HANOI']['index']
        src_pos, targ_pos = self.roles.index('source'), self.roles.index('target')
        if self.current_task_index != hanoi_index:
            bool = self._is_move_possible(self.pillars[src_pos], self.pillars[targ_pos], self.n)
        else:
            bool = self._decr_n_precondition()
            if self._decr_n_precondition():
                state = self.get_state()
                n = state[2] - 1
                new_state = (state[0], state[1], n)
                pillars = self._get_updated_pillars(new_state)
                bool &= self._is_move_possible(pillars[src_pos], pillars[targ_pos], n)
        return bool

    def _hanoi_postcondition(self, init_state, state):
        updated_pillars = self._get_updated_pillars(state)
        bool = init_state[2] == state[2] # size of problem has not changed
        bool &= init_state[1] == state[1] # check roles are the same
        bool &= init_state[3] == state[3]  # check roles are the same
        targ_pos = self._get_init_roles().index('target')
        bool &= len(updated_pillars[targ_pos]) == init_state[2]
        return bool

    def _get_updated_pillars(self, state=None):
        updated_pillars = []
        if state is None:
            state = self.get_state()
        pillars = state[0]
        n = state[2]
        for i in range(3):
            pillar = pillars[i].copy()
            pillar = list(filter(lambda x: x < n, pillar))
            updated_pillars.append(pillar)
        return tuple(updated_pillars)

    def _get_init_roles(self):
        return self.init_roles_stack[-1]

    def _is_solved(self):
        targ_pos = self._get_init_roles().index('target')
        updated_pillars = self._get_updated_pillars()
        return len(updated_pillars[targ_pos]) == self.n

    def get_observation(self):
        src_pos, aux_pos, targ_pos = self.roles.index('source'), self.roles.index('auxiliary'), self.roles.index('target')
        move_1 = int(self._is_move_possible(self.pillars[src_pos], self.pillars[aux_pos]))
        move_2 = int(self._is_move_possible(self.pillars[aux_pos], self.pillars[targ_pos]))
        move_3 = int(self._is_move_possible(self.pillars[src_pos], self.pillars[targ_pos]))
        n_1 = int(self.n == 1)
        is_solved = int(self._is_solved())
        arr = np.array([move_1, move_2, move_3, n_1, is_solved])
        return arr

    def get_observation_dim(self):
        return 5

    def _is_move_possible(self, pillar1, pillar2, n=1):
        """
        Check if it is possible to move disk from pillar1 to pillar2.
        Args:
            pillar1: pillar
            pillar2: pillar
            n: number of disks to be moved

        Returns:
            True if the move is possible, False otherwise.
        """
        if len(pillar1) == 0:
            # no disk in pillar1 so no move possible of course!
            return False
        elif len(pillar2) == 0:
            return True
        else:
            bool = len(pillar1) >= n
            if bool:
                bool &= pillar2[-1] > pillar1[-n]
            return bool

    def start_task(self, task_index):
        if self.tasks_list.count(task_index) > 0:
            task = self.get_program_from_index(task_index)
            if task == 'HANOI':
                self.init_roles_stack.append(self.roles.copy())
                self._decr_n()
        return super(HanoiEnv, self).start_task(task_index)

    def end_task(self):
        current_task = self.get_program_from_index(self.current_task_index)
        if current_task == 'HANOI':
            self.init_roles_stack.pop()
            if self.tasks_list.count(self.current_task_index) > 1:
                self._incr_n()
        super(HanoiEnv, self).end_task()
