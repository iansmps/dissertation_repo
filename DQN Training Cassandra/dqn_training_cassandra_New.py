from __future__ import absolute_import, division, print_function

import logging, sys
import collections
import importlib
import configparser
import traceback
import time
import os
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

#Start logging
timestr = time.strftime("%Y%m%d-%H%M%S")
if sys.argv[2]:
  ident = sys.argv[2]
if not os.path.isdir("model/model_"+ident):
  os.makedirs("model/model_"+ident) 

logging.basicConfig(filename='model/model_'+ident+'/trainingDqnCassandra.log', level=logging.DEBUG)
####################################################

# Load configuration
config = configparser.ConfigParser()
if len(sys.argv) == 1:
  sys.exit("Configuration file missing. You need to call with a configuration file as argument.")
else:
  config.read(sys.argv[1])


########################################################3

##Hyperparameters##

hyp_config = config['Hyperparameters']

max_iterations = hyp_config.getint('MaxIterations')
max_episodes = hyp_config.getint('MaxEpisodes')

initial_collect_steps = hyp_config.getint('InitialCollectSteps')
collect_steps_per_iteration = hyp_config.getint('CollectStepsPerIteration')
replay_buffer_max_length = hyp_config.getint('ReplayBufferMaxLength')

e_greedy = hyp_config.getfloat('Egreedy')
batch_size = hyp_config.getint('BatchSize')
learning_rate = hyp_config.getfloat('LearningRate')
target_update_tau = hyp_config.getfloat('TargetUpdateTau')

# #############################################################

# #Configure the environment
# #Let's create a RandomPyEnvironment to generate structured observations and validate our implementation.

# action_spec = array_spec.BoundedArraySpec((), np.integer, name='action',minimum=0, maximum=20)
# #print(action_spec.shape)

# observation_spec = array_spec.BoundedArraySpec((3,), np.float32, name='observation', minimum=[-100, 0, 0], maximum=[100, 100, 20])

# random_env = random_py_environment.RandomPyEnvironment(observation_spec, action_spec=action_spec)

# # Convert the environment to a TFEnv to generate tensors.
# env = tf_py_environment.TFPyEnvironment(random_env)

# #################################################################

#############################################################

#Configure the environment
env_config = config['Environment']

## Use the correct adaptor (simulation or real deployment)
adaptor = importlib.import_module(env_config['Adaptor'])

logging.info('Creating environment...')
py_env = adaptor.CassandraEnv(env_config)
logging.info('Environment created!')

# Convert the environment to a TFEnv to generate tensors.
env = tf_py_environment.TFPyEnvironment(py_env)

#################################################################

#Create Network
fc_layer_params = (50, 25)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# it's output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])

logging.info('Network created!')

def observation_and_action_constraint_splitter(obs):
    return obs['observations'], obs['legal_moves'] 

##############################################################################
#Create Agent

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)
train_episode_counter = tf.Variable(0)

#global_step = tf.compat.v1.train.get_or_create_global_step()
start_epsilon = 0.5
n_of_steps = 500
end_epsilon = 0.1
epsilon = tf.compat.v1.train.polynomial_decay(
    start_epsilon,
    train_episode_counter,
    n_of_steps,
    end_learning_rate=end_epsilon)



agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    target_update_tau = target_update_tau,
    observation_and_action_constraint_splitter = observation_and_action_constraint_splitter,
    epsilon_greedy = e_greedy,
    #gamma = 0.9,
    td_errors_loss_fn = common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()
#logging.info('Teste:')
#logging.info(train_episode_counter.numpy())
logging.info('Agent created!')
#logging.info('Agent 0: %d',agent.optimizer.numpy())
###############################################################################


#make a step in the environment with the current agent policy
#Returns the previous state, the action taken and the next state
def moveStep (environment, agent):
  time_step = env.current_time_step()
  action_step = agent.collect_policy.action(time_step)
  #logging.debug('   current state = {0}, selected action = {1}'.format(time_step.observation['observations'].numpy()[0], int(action_step.action)))
  logging.debug('   current state = {0},{1},{2},{3},{4},{5},{6}, selected action = {7}'.format(time_step.observation['observations'].numpy()[0][0],time_step.observation['observations'].numpy()[0][1],time_step.observation['observations'].numpy()[0][2],time_step.observation['observations'].numpy()[0][3],time_step.observation['observations'].numpy()[0][4],time_step.observation['observations'].numpy()[0][5],time_step.observation['observations'].numpy()[0][6], int(action_step.action)))
  next_time_step = env.step(action_step.action)
  #logging.debug('   new state = {0}, reward = {1}'.format(next_time_step.observation['observations'].numpy()[0], next_time_step.reward.numpy()[0]))
  logging.debug('   new state = {0},{1},{2},{3},{4},{5},{6}, reward = {7}'.format(next_time_step.observation['observations'].numpy()[0][0],next_time_step.observation['observations'].numpy()[0][1],next_time_step.observation['observations'].numpy()[0][2],next_time_step.observation['observations'].numpy()[0][3],next_time_step.observation['observations'].numpy()[0][4],next_time_step.observation['observations'].numpy()[0][5],next_time_step.observation['observations'].numpy()[0][6], next_time_step.reward.numpy()[0]))
  return time_step, action_step, next_time_step

######################################################################################

def trainDqnAgent (agent, environment, buffer, batch_size=64, max_episodes=50, max_iterations=100):
  log_file = open('model/model_'+ident+'/trainingLog.txt', 'a')
  
  logging.info('Training...')

  dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

  iterator = iter(dataset)

  # (Optional) Optimize by wrapping some of the code in a graph using TF function.
  agent.train = common.function(agent.train)

  total_return = 0.0


  episode_returns = collections.deque(maxlen=10) #to calculate a moving average of 10 episodes

  try:

    for i_episode in range(max_episodes):

      logging.debug('New episode: %d', i_episode)
     
      #logging.info('Episode: %d',agent.train_step_counter.numpy())
      
      # Initialize the environment
      environment.reset()
            
      penaltys = 0
      cost = 0.0
      episode_return = 0.0

      train_episode_counter.assign(i_episode)
      logging.info('Episode Counter:')
      logging.info(train_episode_counter.numpy())

      # Reset the train step
      agent.train_step_counter.assign(0)


      for i in range(max_iterations):

        logging.debug('step = {0}'.format(i))


        
        #make a step in the environment
        time_step, action_step, next_time_step = moveStep(environment, agent)
        episode_return += next_time_step.reward
        #logging.info(next_time_step.observation['observations'][0][1].numpy())
        cost += next_time_step.observation['observations'][0][1].numpy() ################################
        if next_time_step.reward.numpy() <= -0.5 or next_time_step.reward.numpy() == 0:
          #logging.info('PENALTY')                                   ################################
          penaltys += 1


        # Add trajectory to the replay buffer
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        buffer.add_batch(traj)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if next_time_step.is_last():
          break

      episode_returns.append(episode_return)
      avg_return = sum(episode_returns) / (10)
      avg_cost = round(cost/(i+1),2)
      avg_penaltys = round(penaltys/(i+1),2)
      logging.debug('step2 = {0}'.format(i))
      print(i_episode, episode_return.numpy()[0], avg_return.numpy()[0] , avg_penaltys, avg_cost, sep=", ", file=log_file)
      log_file.flush()
      logging.debug('EPISODE = {0}: Episode Return = {1}, Average Return = {2}, Penaltys = {3}, Total Cost = {4}'.format(i_episode, episode_return.numpy()[0], avg_return.numpy()[0],penaltys,avg_cost))
  
  except Exception as err:
    logging.debug('Training terminated manually or with error:')
    logging.debug(err)
    logging.debug(traceback.format_exc())

  log_file.close()


##########################################################################3

#Create the Replay Buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=env.batch_size,
    max_length=replay_buffer_max_length)

logging.info('Replay Buffer created!')

##########################################################################
#Create checkpointer to save training
checkpoint_config = config['Checkpoint']

if checkpoint_config.getboolean('LoadExistingModel') or checkpoint_config.getboolean('SaveModel'):
  
  logging.info('Creating Checkpoint and policySaver...')
  checkpoint_dir = 'model/model_'+ident+'/checkpoint'
  train_checkpointer = common.Checkpointer(
      ckpt_dir=checkpoint_dir,
      max_to_keep=1,
      agent=agent,
      policy=agent.policy,
      replay_buffer=replay_buffer,
      global_step=train_step_counter
  )

  #Create policySaver to save policy
  policy_dir = 'model/model_'+ident+'/policy'
  tf_policy_saver = policy_saver.PolicySaver(agent.policy)

################################################################################

if checkpoint_config.getboolean('LoadExistingModel') & train_checkpointer.checkpoint_exists:
  logging.info('Loading existing model...')
  train_checkpointer.initialize_or_restore()
  train_step_counter = agent.train_step_counter
else:
  #Initialize Buffer
  logging.info('Initializing Replay Buffer...')
  for i in range(batch_size+1):
    time_step, action_step, next_time_step = moveStep(env, agent)
    logging.debug('   step = {0}: reward = {1}'.format(i, next_time_step.reward.numpy()[0]))
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)

  logging.info('Reply Buffer initialized!')

log_file2 = open('model/model_'+ident+'/conf.txt', 'a')
print('PC2: ',max_iterations, max_episodes, e_greedy, file=log_file2)
log_file2.flush()
log_file2.close()

trainDqnAgent(agent, env, replay_buffer, batch_size, max_episodes, max_iterations)

#Save training and policy
if checkpoint_config.getboolean('SaveModel'):
  train_checkpointer.save(train_step_counter)
  tf_policy_saver.save(policy_dir)


#saved_policy = tf.saved_model.load(policy_dir)
#run_episodes_and_create_video(saved_policy, eval_env, eval_py_env)