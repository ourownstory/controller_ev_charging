class config_qn():
    # for experiment, expected:
    controller_name = None
    eval_episodes = 100

    # env config
    render_train     = False
    render_test      = False
    # overwrite_render = True
    record           = False
    # high             = 255.

    # model and training config
    num_episodes_test = 50
    grad_clip         = False
    clip_val          = 1000
    saving_freq       = 5000
    log_freq          = 50
    eval_freq         = 5000
    soft_epsilon      = 0

    # hyper params
    nsteps_train       = 20000
    batch_size         = 64
    buffer_size        = 5000
    target_update_freq = 500
    gamma              = 0.98
    learning_freq      = 1
    state_history      = 1
    lr_begin           = 0.003
    lr_end             = 0.0003
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = nsteps_train/2
    learning_start     = 1000


class config_linear_qn(config_qn):
    # for experiment, expected:
    controller_name = "LinearQN"

    # output config
    output_path  = "results/q_linear/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"


class config_nature_qn(config_qn):
    # for experiment, expected:
    controller_name = "NatureQN"

    # output config
    output_path  = "results/q_nature/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
