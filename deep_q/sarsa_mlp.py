from meta_controller import MetaController
from gym_utils import featurize_cont
import numpy as np
from sklearn import neural_network

class SarsaMLP(MetaController):
    def build(self):
        self.model = neural_network.MLPRegressor(hidden_layer_sizes=self.config.hidden_layer_sizes)
        self.model.fit(np.reshape(np.zeros(self.observation_dim), (1, -1)), np.reshape(np.zeros(self.action_dim), (1, -1)))
        self.lr = self.config.lr
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon
        self.rev_action_map = {tuple(v):k for k, v in self.env.action_map.items()}

    def get_action(self, state):
        a = np.zeros(self.env.num_stations)
        fs = np.reshape(featurize_cont(state), (1, -1))
        out = self.model.predict(fs)
        idxs = np.cumsum([0] + [self.env.config.NUM_POWER_STEPS] * self.env.num_stations)
        if np.random.rand() < 0.997**self.epsilon:
            for i, station in enumerate(state['stations']):
                if station['is_car'] and station['per_char'] < 1.0:
                    a[i] = np.random.choice(self.env.actions[i])
        else:
            for i, station in enumerate(state['stations']):
                if station['is_car'] and station['per_char'] < 1.0:
                    a_idx = np.argmax(out[0][idxs[i]:idxs[i+1]])
                    a[i] = self.env.actions[i][a_idx]

        return self.rev_action_map[tuple(a)]

    def train(self):
        """
        Performs training
        """
        last_eval = 0
        last_record = 0
        scores_eval = []  # list of scores computed at iteration time

        self.init_averages()

        for t in range(self.config.num_batches):
            self.total_train_steps += 1
            last_record += 1
            # collect a minibatch of samples
            paths, total_rewards = self.sample_gameplay(
            self.env,
            max_ep_len=self.config.max_ep_len,
            )
            scores_eval = scores_eval + total_rewards
            observations = np.concatenate([path["observation"] for path in paths])
            observations_p = np.copy(observations)[1:]
            observations = observations[:-1]
            actions = np.concatenate([path["action"] for path in paths])[:-1]
            rewards = np.concatenate([path["reward"] for path in paths])[:-1]

            # run training operations
            fs_list = [featurize_cont(s) for s in observations]
            fsp_list = [featurize_cont(sp) for sp in observations_p]
            pred_fs = self.model.predict(fs_list)
            pred_fsp = self.model.predict(fsp_list)

            masks = np.zeros((len(actions), np.prod([self.env.config.NUM_POWER_STEPS] * self.env.num_stations)))
            for i, a in enumerate(actions):
                masks[i, a] = 1
            masks_p = np.vstack((np.copy(masks)[1:, :], np.copy(masks)[-1:, :]))

            self.model.partial_fit(fs_list,
                                   pred_fs + self.lr * masks * ((np.array(rewards) + self.gamma * np.sum(
                                       pred_fsp * masks_p, axis=1)).reshape(-1, 1) - pred_fs))

            # tf stuff
            if (t % self.config.summary_freq == 0):
                self.update_averages(total_rewards, scores_eval)

                # compute reward statistics for this batch and log
                avg_reward = np.mean(total_rewards)
                sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
                msg = "Average reward: {:9.1f} +/- {:.2f}; batch {}/{}; epsilon={:.2f}".format(avg_reward, sigma_reward, t,
                                                                                           self.config.num_batches,
                                                                                           0.997 ** self.epsilon)
                self.logger.info(msg)

            if self.config.record and (last_record >= self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

            self.epsilon += 1


        self.logger.info("- Training done.")
