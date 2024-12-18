import jax
import jax.numpy as jnp


def g(x, n):
    i = 0
    while i < n:
        i += 1
    return x + i


g_jit_correct = jax.jit(g, static_argnames=['n'])
print(g_jit_correct(10, 20))
