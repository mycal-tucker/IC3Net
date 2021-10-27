from gym.envs.registration import register

register(
    id='Spurious-v0',
    entry_point='custom_envs.spurious_predator_prey_env:SpuriousPredatorPreyEnv',
)
