import os

env = "predator_prey"
seeds = [1,2,3]
mode = "cooperative"
exp_name = "cooperative_g0.01"
nagents = 3
num_epochs = 3000
hid_size = 128
dim = 5
max_steps = 20
vision = 0
save_every = 100
gating_head_cost_factor = 0.01
comm_dim = hid_size
enemy_comm = False

# run for all seeds
for seed in seeds:
    os.system(f"python main.py --comm_dim {comm_dim} --env_name {env} --exp_name {exp_name} --nagents {nagents} "
              f"--mode {mode} --seed {seed} "
              f"--nprocesses 1 --gating_head_cost_factor {gating_head_cost_factor} --num_epochs {num_epochs} "
              f"--hid_size {hid_size} --detach_gap 10 --lrate 0.001 "
              f"--dim {dim} --max_steps {max_steps} --ic3net --vision {vision} --recurrent --save_every {save_every}")

# plot the avg and error graphs using multiple seeds.
os.system(f"python plot.py --env_name {env} --exp_name {exp_name} --nagents {nagents}")
