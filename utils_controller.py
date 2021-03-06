import numpy as np
import matplotlib.pyplot as plt
import json
import os
plt.switch_backend("TkAgg")
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates

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


def plot_episodes(paths, train_step, env, contr_name, out_dir, num=None):
    if num is None:
        num = len(paths)
    for e in range(num):
        path = paths[e]
        infos = path['infos']
        mode = 'Eval' if env.evaluation_mode else 'Train'
        title = "{} - {} Step {}, Episode {}".format(contr_name, mode, train_step, e)
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
        ax_stn.set(xlabel ="Time of Day")

        axarr2_stn.set(ylabel="Vehicle's charge [kWh]")
        axarr2_stn.set_ylim(0, np.amax(plot_data['des_chars']) * 1.1)
        
        hours = mdates.HourLocator(interval = 4)
        h_fmt = mdates.DateFormatter('%H:%M')
        ax_stn.xaxis.set_major_locator(hours)
        ax_stn.xaxis.set_major_formatter(h_fmt)

        
        if stn == env.num_stations - 1:
            ax_stn.legend(loc='upper left')
            axarr2_stn.legend(loc='upper right')
    plt.gcf().autofmt_xdate()
    # self.config.plot_output
    filename = out_dir + title
    plt.savefig(filename)
    # plt.show()
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


def price_energy_histogram(paths, plot_dir, contr_name, mode, num_bins=10):
    infos = [info for path in paths for info in path['infos']]
    prices = [info['price'] for info in infos]
    energies = [info['energy_delivered'] for info in infos]
    plt.hist(prices, bins = num_bins, range=(0.0, 1.0), weights = energies, density = False)
    plt.title('{} - Energy Delivered vs Price (Density)'.format(contr_name))
    plt.xlabel('Price')
    plt.ylabel('Energy Delivered [kWh]')
    title = "{}_energy_price_hist.png".format(mode)
    filename = plot_dir + title
    plt.savefig(filename)
    # plt.show()
    plt.close()


def compute_stats(rewards, paths, config, logger, env, save=False):
    # compute and print
    msgs = ["\n - - - - - - - \nFor the past {} paths:".format(len(paths))]
    stats = {}
    infos = [info for path in paths for info in path['infos']]
    # price_energy_histogram(infos, num_bins=20)
    # _plot_with_infos(infos, 'Eval', env, config.plot_output)
    # scale to be comparable to training rewards:
    rewards = np.array(rewards) * config.max_ep_len / config.max_ep_len_eval
    avg_reward = np.mean(rewards)
    sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
    mode = "Eval" if env.evaluation_mode else "Training"
    msgs.append("{} reward: {:9.1f} +/- {:.2f}".format(mode, avg_reward, sigma_reward))
    stats["reward_avg"] = avg_reward
    stats["reward_sigma"] = sigma_reward

    #compute price per day and average percent best possible charge
    best_possible_energy, tot_energy_delivered, elec_costs = [], [], []
    for info in infos:
        tot_energy_delivered.extend(info['tot_energy_delivered'])
        best_possible_energy.extend(info['best_possible_energy'])
        elec_costs.append(info['elec_cost'])
    best_possible_percents = np.divide(tot_energy_delivered, best_possible_energy)
    avg_percent, avg_elec_cost_per_time_step = np.mean(best_possible_percents), np.mean(elec_costs)
    sigma_percent = np.sqrt(np.var(best_possible_percents) / len(best_possible_percents))
    msgs.append("Avg best possible charge completion (avg(energy_deliv/best_possible)) [1]:  {:9.1f} +/- {:.2f}".format(avg_percent, sigma_percent))
    stats["charge_percent_avg"] = avg_percent
    stats["charge_percent_sigma"] = sigma_percent

    tot_energy_percent = np.sum(tot_energy_delivered) / np.sum(best_possible_energy)
    msgs.append("Tot charge completion (sum(energy_deliv)/sum(best_possible)) [1]: {}".format(tot_energy_percent))
    stats["tot_energy_percent"] = tot_energy_percent

    avg_elec_cost_per_day = avg_elec_cost_per_time_step * 24 / env.time_step
    msgs.append("Avg elec cost per day [$/day]: {}".format(avg_elec_cost_per_day))
    # stats["avg_elec_cost_per_day"] = avg_elec_cost_per_day

    avg_elec_price = np.sum(elec_costs)/np.sum(tot_energy_delivered)
    msgs.append("Avg elec price [$/kWh]: {}".format(avg_elec_price))
    stats["avg_elec_price"] = avg_elec_price

    msgs.append("\n - - - - - - - \n")
    for msg in msgs:
        logger.info(msg)
    # save
    if save:
        _save_statistics(stats, out_path=config.plot_output, evaluation=env.evaluation_mode)
    return avg_reward


def _save_statistics(stats, out_path, evaluation):
    name = "stats_{}.json".format("eval" if evaluation else "train")
    outfile = os.path.join(out_path, name)
    with open(outfile, 'w') as f_out:
        json.dump(stats, f_out, sort_keys=True, indent=4, separators=(',', ': '))


def save_object(obj, out_path, name):
    name = "{}.json".format(name)
    outfile = os.path.join(out_path, name)
    obj_dict = obj.__dict__
    print(obj_dict)
    with open(outfile, 'w') as f_out:
        json.dump(obj_dict, f_out, sort_keys=True, indent=4, separators=(',', ': '))
