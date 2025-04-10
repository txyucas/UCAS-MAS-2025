from .cnn import Actor, Critic
from .fnn import fnnActor, fnnCritic

def get_model(config):
    '''
    returns the actor and critic models based on the config
    '''
    if config.model == 'cnn':
        actor = Actor(config)
        critic = Critic(config)
    elif config.model == 'fnn':
        actor = fnnActor(config)
        critic = fnnCritic(config)
    else:
        raise ValueError('Model not supported')
    #print('Actor model: ', actor)
    #print('Critic model: ', critic)
    print('Model loaded')
    return actor, critic