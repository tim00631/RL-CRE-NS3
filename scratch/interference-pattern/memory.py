import numpy as np

from collections import namedtuple

EpisodeStep = namedtuple("EpisodeStep", field_names=["state", "action", "next_state", "reward"])

class Memory(object):
    def __init__(self, max_mem_size=5000, batch_size=32):
        self.params = {
            'mem_cntr': 0, # memory counter
            'max_mem_size': max_mem_size,
            'batch_size': batch_size,
            'episode_steps': []
        }
 
    def __len__(self):
        return len(self.params['episode_steps'])
    
    def push(self, state, action, next_state, reward):
        if len(self.params['episode_steps']) < self.params['max_mem_size']:
            self.params['episode_steps'].append(None)
        self.params['episode_steps'][self.params['mem_cntr']] = EpisodeStep(state=state, 
                                                        action=action, 
                                                        next_state=next_state, 
                                                        reward=reward)
        self.params['mem_cntr'] = (self.params['mem_cntr'] + 1) % self.params['max_mem_size']

    def sample(self):
        max_mem = self.params['mem_cntr'] if self.params['mem_cntr'] < self.params['max_mem_size'] \
                    else self.params['max_mem_size']
        batch = np.random.choice(max_mem, self.params['batch_size'])

        batch_state = []
        batch_action = []
        batch_target = []

        for batch_index in batch:
            batch_state.append(self.params['episode_steps'][batch_index].state)
            batch_action.append(self.params['episode_steps'][batch_index].action)
            batch_target.append(self.params['episode_steps'][batch_index].reward)
        return np.asarray(batch_state), np.asarray(batch_action), np.asarray(batch_target)