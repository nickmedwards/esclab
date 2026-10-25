"""Input, Output, and Parameter expose the stored value's array protocols."""
import numpy as np
import pandas as pd
import pytest

from esclab.simulate import Component


@pytest.fixture(params=[Component.Input, Component.Output, Component.Parameter])
def wrap(request):
    def make_wrapper(value):
        wrapper = request.param()
        wrapper.v = value
        return wrapper
    return make_wrapper


@pytest.mark.parametrize('shape', [(1,), (3,), (64,), (1, 3)])
@pytest.mark.parametrize('dtype', [float, int, bool])
def test_copy_preserves_array_and_has_independent_storage(wrap, shape, dtype):
    value = np.ones(shape, dtype=dtype)
    wrapper = wrap(value)
    snapshot = wrapper.copy()

    assert isinstance(snapshot, np.ndarray)
    assert snapshot.shape == value.shape
    assert snapshot.dtype == value.dtype
    assert not np.shares_memory(snapshot, value)
    np.testing.assert_array_equal(snapshot, value)

    wrapper[...] = 0
    np.testing.assert_array_equal(snapshot, np.ones(shape, dtype=dtype))
    snapshot[...] = 2
    np.testing.assert_array_equal(wrapper, np.zeros(shape, dtype=dtype))


@pytest.mark.parametrize('value', [True, False, 0, 2.5])
def test_copy_preserves_scalar_value(wrap, value):
    snapshot = wrap(value).copy()
    assert type(snapshot) is type(value)
    assert snapshot == value


def test_copy_commits_independent_component_state(wrap):
    class State(Component):
        def __init__(self):
            super().__init__()
            self.current = wrap(np.array([1., 2., 3.]))
            self.accepted = wrap(np.zeros(3))

    state = State()
    accepted_wrapper = state.accepted
    state.accepted = state.current.copy()
    assert state.accepted is accepted_wrapper
    state.current[:] = -1
    np.testing.assert_array_equal(state.accepted, [1., 2., 3.])
    state.accepted[0] = 9
    np.testing.assert_array_equal(state.current, [-1., -1., -1.])


@pytest.mark.parametrize('shape', [(3,), (1, 3)])
def test_iteration_matches_stored_array(wrap, shape):
    value = np.arange(3).reshape(shape)
    np.testing.assert_array_equal(list(wrap(value)), list(value))


@pytest.mark.parametrize('count', [1, 3, 64])
def test_dataframe_columns_are_numeric_and_support_aggregation(wrap, count):
    indoor = np.arange(count, dtype=float) + 20
    mode = np.arange(count, dtype=int) % 2
    energy = np.arange(count, dtype=float) * 0.5
    frame = pd.DataFrame(dict(indoor=wrap(indoor), mode=wrap(mode), energy=wrap(energy)))
    expected = pd.DataFrame(dict(indoor=indoor, mode=mode, energy=energy))

    pd.testing.assert_frame_equal(frame, expected)
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes)
    pd.testing.assert_series_equal(frame.mean(), expected.mean())
    pd.testing.assert_series_equal(frame.sum(), expected.sum())


@pytest.mark.parametrize('value', [True, False, 0, 1, np.bool_(True), np.bool_(False),
                                  np.array([True]), np.array([False])])
def test_truth_testing_matches_stored_value(wrap, value):
    assert bool(wrap(value)) is bool(value)


def test_truth_testing_preserves_numpy_ambiguous_truth_error(wrap):
    with pytest.raises(ValueError, match='ambiguous'):
        bool(wrap(np.array([True, False])))


def test_scalar_iteration_remains_invalid(wrap):
    with pytest.raises(TypeError):
        iter(wrap(1.))
