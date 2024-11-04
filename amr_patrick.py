"""A 1-dimensional example of adaptive mesh refinement in JAX. In this case, a simple
implementation of quadrature.
Static shapes don't mean you can't do this. Heap allocation is *not* necessary!
Not extensively tested; any bugs leave a comment below.
"""

import functools as ft
from collections.abc import Callable

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


FloatScalar = float | Float[Array, ""]
IntScalar = int | Int[Array, ""]
BoolScalar = bool | Bool[Array, ""]


class _List(eqx.Module):
    buffer: Float[Array, " buffer_size"]
    length: IntScalar

    def append(self, value: FloatScalar) -> "_List":
        # TODO: this error-checking is quite an expensive operation. We could probably
        # move it somewhere else cheaper, rather than checking repeatedly for every
        # list.
        value = eqx.error_if(
            value, self.length >= len(self.buffer), "Buffer maximum size reached."
        )
        buffer = self.buffer.at[self.length].set(value)
        return _List(buffer, self.length + 1)

    def delete(self) -> "_List":
        # We don't bother deleting the old value, we just move the length point back
        # one spot.
        return _List(self.buffer, self.length - 1)

    def overwrite(self, index: IntScalar, value: FloatScalar) -> "_List":
        buffer = self.buffer.at[index].set(value)
        return _List(buffer, self.length)

    def __getitem__(self, item: IntScalar) -> FloatScalar:
        return self.buffer[item]


def _agenda_non_empty(carry) -> BoolScalar:
    agenda, *_ = carry
    return agenda.length > 0


def _trapeze(t0, t1, f0, f1):
    return (f0 + 0.5 * (f1 - f0)) * (t1 - t0)


def _delete_from_agenda(value):
    agenda, ts, fs, interval_starts, interval_ends = value
    agenda = agenda.delete()
    return agenda, ts, fs, interval_starts, interval_ends


def _append_to_agenda(t_half, f_half, interval_index, index_end, value):
    agenda, ts, fs, interval_starts, interval_ends = value
    # Store the evaluation we just made.
    index_half = ts.length
    ts = ts.append(t_half)
    fs = fs.append(f_half)
    # Overwrite the existing interval with the new left subinterval.
    interval_ends = interval_ends.overwrite(interval_index, index_half)
    # Now append the new right subinterval.
    new_interval_index = interval_starts.length
    interval_starts = interval_starts.append(index_half)
    interval_ends = interval_ends.append(index_end)
    # We need to check our two new subintervals to see if we need to split them further.
    # `interval_index` (now corresponding to the left subinterval) is already on there
    # so leave it. We just need to append the new right subinterval.
    agenda = agenda.append(new_interval_index)
    return agenda, ts, fs, interval_starts, interval_ends


def _split_interval(f, carry):
    agenda, ts, fs, interval_starts, interval_ends, eps = carry
    interval_index = agenda[agenda.length - 1]
    index_start = interval_starts[interval_index]
    index_end = interval_ends[interval_index]
    t0 = ts[index_start]
    t1 = ts[index_end]
    f0 = fs[index_start]
    f1 = fs[index_end]
    # More numerically stable than `(t0 + t1) / 2`.
    t_half = t0 + (t1 - t0) / 2
    f_half = f(t_half)
    approx1 = _trapeze(t0, t1, f0, f1)
    approx2 = _trapeze(t0, t_half, f0, f_half) + _trapeze(t_half, t1, f_half, f1)
    predicate = jnp.abs(approx1 - approx2) < eps
    agenda, ts, fs, interval_starts, interval_ends = lax.cond(
        predicate,
        _delete_from_agenda,
        ft.partial(_append_to_agenda, t_half, f_half, interval_index, index_end),
        (agenda, ts, fs, interval_starts, interval_ends),
    )
    return agenda, ts, fs, interval_starts, interval_ends, eps


def _over_intervals(ts, carry):
    step, _ = carry
    return step < ts.length


def _sum_intervals(ts, fs, interval_starts, interval_ends, carry):
    step, accumulator = carry
    index_start = interval_starts[step]
    index_end = interval_ends[step]
    t0 = ts[index_start]
    t1 = ts[index_end]
    f0 = fs[index_start]
    f1 = fs[index_end]
    accumulator = accumulator + _trapeze(t0, t1, f0, f1)
    return step + 1, accumulator


@eqx.filter_jit
def quadrature(
    f: Callable[[FloatScalar], FloatScalar],
    t0: FloatScalar,
    t1: FloatScalar,
    eps: FloatScalar,
    buffer_size: int,
) -> tuple[FloatScalar, Float[Array, " {buffer_size}"], Float[Array, " {buffer_size}"]]:
    r"""Computes `\int_t0^t1 f(t) dt` with the trapezium rule over an adaptively-sized
    grid.
    **Arguments:**
    - `f`, `t0`, `t1`: as above. It is required that `t0 <= t1`.
    - `eps` corresponds to the desired tolerance: intervals will not be mesh-refined
        once doing so would only change their value by this much.
    - `buffer_size` corresponds to the maximum number of evaluations that are allowed.
        The current implementation does not check this and will silently return the
        wrong value if this is exceeded.
    **Returns:**
    A 3-tuple of `(result, ts, fs)`, where:
    - `result` is the value of the integration.
    - `ts` is an array of shape `(buffer_size,)` holding the t-values at which the
        function was evaluated.
    - `fs` is an array of shape `(buffer_size,)` holding `f` evaluated at each value in
        `ts`.
    Both `ts` and `fs` will have their unused buffer space padded with `jnp.inf`.
    """

    t0, t1 = eqx.error_if((t0, t1), t0 > t1, "t0 must be less than t1")
    t_dtype = jnp.result_type(t0, t1, jnp.float32)
    t0 = jnp.astype(t0, t_dtype)
    t1 = jnp.astype(t1, t_dtype)
    f_dtype = jnp.result_type(jax.eval_shape(f, t0), jnp.float32)

    # Records the (t, f(t)) pairs. Not using `jnp.nan` for the pad value because that
    # disrupts the use of `JAX_DEBUG_NANS=1`.
    ts = _List(jnp.full(buffer_size, jnp.inf, dtype=t_dtype), 0)
    fs = _List(jnp.full(buffer_size, jnp.inf, dtype=f_dtype), 0)
    # These store indices into `ts` and `fs`. We split our `[t0, f0]` into a disjoint
    # union of intervals. The `i`th interval (not stored in any particular order) begins
    # at `ts[interval_starts[i]]` and ends at `ts[interval_ends[i]]`.
    interval_starts = _List(jnp.zeros(buffer_size - 1, dtype=jnp.int32), 0)
    interval_ends = _List(jnp.zeros(buffer_size - 1, dtype=jnp.int32), 0)

    # Use `vmap` so that we only trace `f` once, which reduces compile time.
    f0, f1 = jax.vmap(f)(jnp.stack([t0, t1]))
    ts = ts.append(t0)
    ts = ts.append(t1)
    fs = fs.append(f0)
    fs = fs.append(f1)
    interval_starts = interval_starts.append(0)
    interval_ends = interval_ends.append(1)

    # The agenda stores indices into `interval_{starts, ends}`, corresponding to the
    # intervals we still need to process.
    agenda = _List(jnp.zeros(buffer_size - 1, dtype=jnp.int32), 0)
    agenda = agenda.append(0)

    _, ts, fs, interval_starts, interval_ends, _ = lax.while_loop(
        _agenda_non_empty,
        ft.partial(_split_interval, f),
        (agenda, ts, fs, interval_starts, interval_ends, eps),
    )

    _, total = lax.while_loop(
        ft.partial(_over_intervals, ts),
        ft.partial(_sum_intervals, ts, fs, interval_starts, interval_ends),
        (0, 0),
    )
    sort_indices = jnp.argsort(ts.buffer)
    return total, ts.buffer[sort_indices], fs.buffer[sort_indices]


if __name__ == "__main__":
    result, _, _ = quadrature(lambda t: t**3, t0=0, t1=1, eps=1e-5, buffer_size=128)
    print(result)