import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#allows for plotting from training or eval
def _plot_with_infos(infos, title, env, out_dir):
    plot_data = {"times": [], "actions": [[] for _ in range(env.num_stations)],
                 "per_chars": [[] for _ in range(env.num_stations)],
                 "des_chars": [[] for _ in range(env.num_stations)],
                 "is_cars": [[] for _ in range(env.num_stations)],
                 "price": []}
    for info in infos:
        update_plot_data(plot_data, info)
    # actually plot
    _plot_episode(plot_data, title, env, out_dir)

def plot_episodes(paths, train_step, env, out_dir, num=None):
    if num is None:
        num = len(paths)
    for e in range(num):
        path = paths[e]
        infos = path['infos']
        title = "Step {}, Episode {}".format(train_step, e)
        _plot_with_infos(infos, title, env, out_dir)

def _plot_episode(plot_data, title, env, out_dir):
    f, axarr = plt.subplots(env.num_stations, sharex=True, figsize=(14,7))
    f.suptitle(title)
    for stn in range(env.num_stations):
        if env.num_stations > 1:
            ax_stn = axarr[stn]
        else:
            ax_stn = axarr

        no_car_mask = np.array(plot_data['is_cars'][stn])
        pow_violation_mask = np.array(np.sum(plot_data['actions'], axis=0) > env.transformer_capacity)
        ax_stn.plot(plot_data['times'], plot_data['actions'][stn], 'b-', label='Commanded Power [kW]')
        ax_stn.plot(np.array(plot_data['times'])[pow_violation_mask],
                        np.array(plot_data['actions'][stn])[pow_violation_mask], 'r.', markersize=5,
                        label='Transformer Capacity Violation')
        ax_stn.plot(np.array(plot_data['times'])[~no_car_mask], np.array(plot_data['per_chars'][stn])[~no_car_mask],
                        'kx', markersize=7,
                        label='Car Not Present')
        ax_stn.plot(plot_data['times'], plot_data['price'], 'r:', label = 'Normalized Price') 

        full_charge_mask = np.array(np.array(plot_data['per_chars'][stn]) > 0.99)
        axarr2_stn = ax_stn.twinx()
        axarr2_stn.plot(plot_data['times'],
                        np.array(plot_data['per_chars'][stn]) * np.array(plot_data['des_chars'][stn]), 'g-',
                        label='Charge [kWh]', alpha=0.75)
        axarr2_stn.plot(plot_data['times'], np.ones_like(plot_data['times']) * np.array(plot_data['des_chars'][stn]),
                        'g--', label="Full Charge [kWh]", alpha=0.5)
        axarr2_stn.plot(np.array(plot_data['times'])[full_charge_mask],
                        np.array(np.array(plot_data['per_chars'][stn]) * np.array(plot_data['des_chars'][stn]))[
                            full_charge_mask], 'g.', markersize=5,
                        label='Fully Charged')

        ax_stn.set(ylabel="Power [kW] for Station #{}".format(stn))
        ax_stn.set_ylim(env.min_power, env.max_power * 1.1)

        axarr2_stn.set(ylabel="Vehicle's charge [kWh]")
        axarr2_stn.set_ylim(0, np.amax(plot_data['des_chars']) * 1.1)

        if stn == env.num_stations - 1:
            ax_stn.legend(loc='upper left')
            axarr2_stn.legend(loc='upper right')
    # self.config.plot_output
    filename = out_dir + title
    plt.savefig(filename)
    plt.show()
    plt.close()

def update_plot_data(plot_data, info):
    new_state, charge_rates = info['new_state'], info['charge_rates']
    num_stations = len(new_state['stations'])
    plot_data["times"].append(new_state['time'])
    plot_data["price"].append(info['price'])
    for stn in range(num_stations): plot_data['actions'][stn].append(charge_rates[stn])
    for stn in range(num_stations): plot_data['per_chars'][stn].append(new_state['stations'][stn]['per_char'])
    for stn in range(num_stations): plot_data['des_chars'][stn].append(
        new_state['stations'][stn]['des_char'])
    for stn in range(num_stations): plot_data['is_cars'][stn].append(
        new_state['stations'][stn]['is_car'])


def print_evaluation_statistics(rewards, paths, config, logger, env):
    infos = [info for path in paths for info in path['infos']]
    # scale to be comparable to training rewards:
    _plot_with_infos(infos, 'Eval', env, config.plot_output)
    rewards = np.array(rewards) * config.max_ep_len / config.max_ep_len_eval
    avg_reward = np.mean(rewards)
    sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
    msg = "Evaluation reward: {:9.1f} +/- {:.2f}".format(avg_reward, sigma_reward)
    logger.info(msg)
    #compute price per day and average percent best possible charge
    best_possible_percents, prices = [], []
    for info in infos:
        best_possible_percents.extend(info['finished_cars_stats'])
        prices.append(info['elec_cost'])
    avg_percent, avg_price = np.mean(best_possible_percents), np.mean(prices)
    sigma_percent = np.sqrt(np.var(best_possible_percents) / len(best_possible_percents))
    avg_daily_price = avg_price*24/env.time_step
    msg = "Avg best possible charge percentage:  {:9.1f} +/- {:.2f}".format(avg_percent, sigma_percent)
    logger.info(msg)      
    msg = "Avg daily price: {}".format(avg_daily_price)
    logger.info(msg)
    return avg_reward
