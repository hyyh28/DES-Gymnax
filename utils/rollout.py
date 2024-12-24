import jax


def rollout(rng_input, env, env_params):
    steps_in_episode = env_params.max_time_step
    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, state = env.reset(rng_reset, env_params)

    def policy_step(state_input, _):
        obs, state, rng = state_input
        # jax.debug.print("rng {rng} time_step {bar}", rng=rng, bar=state.time)
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        action = 0  # Example policy: always choose action 0
        next_obs, next_state, reward, done, _ = env.step(rng_step, state, action, env_params)
        carry = (next_obs, next_state, rng)
        return carry, (obs, action, reward, next_obs, done)

    # Scan over episode steps
    _, scan_out = jax.lax.scan(
        policy_step,
        (obs, state, rng_episode),
        None,
        length=steps_in_episode,  # Pass dynamically here
    )
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done


def batch_rollout(rng_input, env, env_params):
    # Vectorize rollouts over batch dimension (e.g., RNG input)
    batch_rollout_fn = jax.vmap(rollout, in_axes=(0, None, None))
    return batch_rollout_fn(rng_input, env, env_params)
