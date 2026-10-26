"""Simulation completion, extension, and component post-simulation hooks."""

import pytest

from esclab.simulate import Component, Model


class Source(Component):
    signal = Component.Output()

    def calculate(self):
        self.signal = self.model.time + 1

    def postsim_calcs(self):
        self.model.events.append((self.name, self.model.time))
        self.history_at_finalization = list(self.model.historian[self.signal.name])


class Sink(Component):
    inlet = Component.Input()
    result = Component.Output()

    def calculate(self):
        self.result = self.inlet * 2

    def postsim_calcs(self):
        self.model.events.append((self.name, self.model.time))
        # Source must have finalized first, despite reversed attribute order.
        self.source_history = list(self.model.z_source.history_at_finalization)
        self.history_at_finalization = list(self.model.historian[self.result.name])


def make_model():
    model = Model()
    model.events = []
    model.z_source = Source()
    model.z_source.name = "source"
    model.a_sink = Sink()
    model.a_sink.name = "sink"
    model.connect(model.z_source.signal, model.a_sink.inlet)
    model.settings.stop_time = 2.5
    return model


@pytest.mark.parametrize("max_iterations", [1, 10])
def test_automatic_finalization_order_history_and_convergence(max_iterations, capsys):
    model = make_model()
    model.settings.max_iterations = max_iterations
    model.step()
    model.step()
    assert model.events == []

    model.step()  # Overshoots the exclusive endpoint, logging t=2 before finalizing.

    assert model.events == [("source", 3), ("sink", 3)]
    assert model.historian["time"] == [0, 1, 2]
    assert model.z_source.history_at_finalization == [1, 2, 3]
    assert model.a_sink.history_at_finalization == [2, 4, 6]
    assert model.a_sink.source_history == [1, 2, 3]
    assert model.a_sink.inlet.connection.last_step_is_converged == (max_iterations > 1)
    assert ("not converged" in capsys.readouterr().out) == (max_iterations == 1)

    model.finalize()
    assert len(model.events) == 2
    model.settings.stop_time = 10
    with pytest.raises(RuntimeError, match="Cannot step after finalization"):
        model.step()
    assert model.historian["time"] == [0, 1, 2]


@pytest.mark.parametrize("automatic_final_extension", [False, True])
def test_extend_run_then_finalize(automatic_final_extension):
    model = make_model()
    model.settings.auto_postsim_calcs = False
    while model.time < model.settings.stop_time:
        model.step()
    assert model.events == []

    model.settings.stop_time = 5
    model.settings.auto_postsim_calcs = automatic_final_extension
    assert model.events == []
    while model.time < model.settings.stop_time:
        model.step()
    if not automatic_final_extension:
        assert model.events == []
        model.finalize()

    assert model.events == [("source", 5), ("sink", 5)]
    assert model.z_source.history_at_finalization == [1, 2, 3, 4, 5]


def test_manual_early_finalization():
    model = make_model()
    model.step()
    model.finalize()
    assert model.events == [("source", 1), ("sink", 1)]
    model.finalize()
    assert len(model.events) == 2


def test_finalize_requires_completed_step():
    model = make_model()
    with pytest.raises(RuntimeError, match="before a timestep has completed"):
        model.finalize()
    model.step()  # Rejected premature finalization does not close the run.
    assert model.events == []


def test_failed_finalization_propagates_and_cannot_repeat():
    model = make_model()
    model.settings.stop_time = 1

    def fail():
        raise ValueError("postsim failure")

    model.a_sink.postsim_calcs = fail
    with pytest.raises(ValueError, match="postsim failure"):
        model.step()
    assert model.events == [("source", 1)]
    assert model.historian["time"] == [0]
    with pytest.raises(RuntimeError, match="finalization is failed"):
        model.finalize()
    with pytest.raises(RuntimeError, match="Cannot step after finalization"):
        model.step()
    assert model.events == [("source", 1)]


def test_calculation_exception_does_not_finalize():
    model = make_model()
    model.settings.stop_time = 1

    def fail():
        raise ValueError("calculation failure")

    model.z_source.calculate = fail
    with pytest.raises(ValueError, match="calculation failure"):
        model.step()
    assert model.events == []
    assert model.historian["time"] == []
