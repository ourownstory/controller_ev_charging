import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")


def plot_episode(plot_data, eps_num, env, out_dir):
    f, axarr = plt.subplots(env.num_stations, sharex=True)
    f.suptitle("Episode #{}".format(eps_num))
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
    # plt.show()
    # self.config.plot_output
    filename = out_dir + "episode_{}".format(eps_num)
    plt.savefig(filename)
    plt.close()


def update_plot_data(plot_data, info):
    new_state, charge_rates = info['new_state'], info['charge_rates']
    num_stations = len(new_state['stations'])
    plot_data["times"].append(new_state['time'])
    for stn in range(num_stations): plot_data['actions'][stn].append(charge_rates[stn])
    for stn in range(num_stations): plot_data['per_chars'][stn].append(new_state['stations'][stn]['per_char'])
    for stn in range(num_stations): plot_data['des_chars'][stn].append(
        new_state['stations'][stn]['des_char'])
    for stn in range(num_stations): plot_data['is_cars'][stn].append(
        new_state['stations'][stn]['is_car'])