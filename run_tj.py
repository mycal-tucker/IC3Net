import os

# TODO: Run proto version

env = "traffic_junction"
seeds = [1, 2]
exp_name = "tj_g0.01"
nagents = 5
num_epochs = 2000
hid_size=  128
dim =  6
max_steps = 20
vision = 0
    save_every = 100
gating_head_cost_factor = 0.01
comm_dim = hid_size
# comm_action_one = True

# run for all seeds
for seed in seeds:
    os.system(f"python main.py --env_name {env} --comm_dim {comm_dim} --nagents {nagents} --nprocesses 1 "
              f"--num_epochs {num_epochs} "
              f"--hid_size {hid_size} --seed {seed}"
              f" --detach_gap 10 --lrate 0.001 --dim {dim} --max_steps {max_steps} --ic3net --vision {vision} "
              f"--recurrent --gating_head_cost_factor {gating_head_cost_factor} "
              f"--add_rate_min 0.1 --add_rate_max 0.3 --curr_start 250 --curr_end 1250 --difficulty easy "
              f"--exp_name {exp_name} --save_every {save_every}")

# plot the avg and error graphs using multiple seeds.
os.system(f"python plot.py --env_name {env} --exp_name {exp_name} --nagents {nagents}")


