from environment.network_env import NetworkEnv
import a3c
import tensorflow as tf
import numpy as np

from constants import SUMMARY_DIR, TRAIN_EPOCHS, TRAIN_EPISODES, \
    TRAIN_ATTEMPTS, NN_MODEL, NUM_AGENTS, MODEL_DIR, MODEL_SAVE_INTERVAL

'''
Models Agent - Environment interaction. 
'''
class Agent:

    def __init__(self, agent_id, nnet_params_queue, train_queue):
        self.env = NetworkEnv()
        self.agent_id = agent_id
        self.nnet_params_queue = nnet_params_queue
        self.train_queue = train_queue
        
    def run(self):
        with tf.Session() as sess, open(SUMMARY_DIR + '/log_agent_' + str(self.agent_id), 'wb') as log_file:
            actor = a3c.ActorNetwork(sess)
            critic = a3c.CriticNetwork(sess)

            # synchronize local parameters with global parameters
            actor_nnet_params, critic_nnet_params = self.nnet_params_queue.get()
            actor.set_network_params(actor_nnet_params)
            critic.set_network_params(critic_nnet_params)

            # each epoch is with new actor - critic neural network
            for epoch in range(TRAIN_EPOCHS):

                state_batch = []
                action_batch = []
                reward_batch = []

                # each episode is a new instance of routing with constraints problem
                for episode in range(TRAIN_EPISODES):
                    self.env.reset_hard()

                    # each attempt is trying to solve the same instance again
                    for attempt in range(TRAIN_ATTEMPTS):
                        state = self.env.reset()

                        while(True):
                            state_batch.append(state)

                            action_prob = actor.predict(state)
                            action_index = np.random.choice(range(len(action_prob)), p=action_prob)

                            action_vector = np.zeros(len(action_prob))
                            action_vector[action_index] = 1
                            action_batch.append(action_vector)

                            action = state["next_hops"][action_index]

                            state, reward, done = self.env.step(action)
                            reward_batch.append(reward)

                            log_file.write(
                                'epoch:' + str(epoch) + 
                                ', episode: ' + str(episode) +
                                ', attempt:' + str(attempt) + 
                                ', reward:' + str(np.sum(reward)) + 
                                ', step:' + str(len(reward_batch))
                            )
                            
                            if done:
                                break
                    
                    log_file.flush()
                
                self.train_queue.put([state_batch, action_batch, reward_batch, done])
                actor_nnet_params, critic_nnet_params = self.nnet_params_queue.get()
                actor.set_network_params(actor_nnet_params)
                critic.set_network_params(critic_nnet_params)

'''
Central Agent responsible for training every epoch.
'''
class CentralAgent:

    def __init__(self, nnet_params_queues, train_queues):
        self.nnet_params_queues = nnet_params_queues
        self.train_queues = train_queues
    
    def run(self):
        with tf.Session() as sess, open(SUMMARY_DIR + '/log_central', 'wb') as log_file:
            actor = a3c.ActorNetwork(sess)
            critic = a3c.CriticNetwork(sess)

            summary_ops, summary_vars = a3c.build_summaries()

            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph) # monitor training progress
            saver = tf.train.Saver() # save neural net parameters

            # restore neural net parameters
            nn_model = NN_MODEL
            if nn_model is not None:
                saver.restore(sess, nn_model)
                print("Model restored.")

            for epoch in range(TRAIN_EPOCHS):
                actor_nnet_params = actor.get_network_params()
                critic_nnet_params = critic.get_network_params()

                for agent_id in range(NUM_AGENTS):
                    self.nnet_params_queues[agent_id].put([actor_nnet_params, critic_nnet_params])
                
                # record average reward and td loss change
                # in the experiences from the agents
                total_batch_len = 0.0
                total_reward = 0.0
                total_td_loss = 0.0
                total_agents = 0.0 

                # assemble experiences from the agents
                actor_gradient_batch = []
                critic_gradient_batch = []

                for agent_id in range(NUM_AGENTS):
                    state_batch, action_batch, reward_batch, terminal = self.train_queues[agent_id].get()

                    actor_gradient, critic_gradient, td_batch = \
                        a3c.compute_gradients(
                            state_batch=np.vstack(state_batch),
                            action_batch=np.vstack(action_batch),
                            reward_batch=np.vstack(reward_batch),
                            terminal=terminal, actor=actor, critic=critic)

                    actor_gradient_batch.append(actor_gradient)
                    critic_gradient_batch.append(critic_gradient)

                    total_reward += np.sum(reward_batch)
                    total_td_loss += np.sum(td_batch)
                    total_batch_len += len(reward_batch)
                    total_agents += 1.0

                # compute aggregated gradient
                assert NUM_AGENTS == len(actor_gradient_batch)
                assert len(actor_gradient_batch) == len(critic_gradient_batch)

                for agent_id in range(len(actor_gradient_batch)):
                    actor.apply_gradients(actor_gradient_batch[agent_id])
                    critic.apply_gradients(critic_gradient_batch[agent_id])

                # log training information
                avg_reward = total_reward  / total_agents
                avg_td_loss = total_td_loss / total_batch_len

                log_file.write(
                    'Epoch: ' + str(epoch) +
                    ' TD_loss: ' + str(avg_td_loss) +
                    ' Avg_reward: ' + str(avg_reward) + '\n'
                )
                log_file.flush()

                summary_str = sess.run(
                    summary_ops, 
                    feed_dict={
                        summary_vars[0]: avg_td_loss,
                        summary_vars[1]: avg_reward
                    }
                )

                writer.add_summary(summary_str, epoch)
                writer.flush()

                if epoch % MODEL_SAVE_INTERVAL == 0:
                    # Save the neural net parameters to disk.
                    save_path = saver.save(
                        sess, MODEL_DIR + "/nn_model_ep_" +
                        str(epoch) + ".ckpt"
                    )
                    print('Model Saved %s', save_path)
