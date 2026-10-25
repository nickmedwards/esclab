"""Minimal NumPy array P/I/O example

Each array element is one realization: a sine source feeds an elementwise gain.
Run this file to compare percentile bands, spaghetti, and selected-index plots.
"""
import numpy as np
from esclab.simulate import Component, Model


class SineSource(Component):
    amplitude = Component.Parameter(np.array([1., 1.5, 2.]))
    period = Component.Parameter(60., units="sec")
    signal = Component.Output()

    def calculate(self):
        self.signal = self.amplitude.v * np.sin(2 * np.pi * self.model.time / self.period.v)


class ArrayGain(Component):
    signal = Component.Input()
    gain = Component.Parameter(np.array([0.5, 1., 1.5]))
    result = Component.Output()

    def calculate(self):
        self.result = self.gain.v * self.signal.v


def create_model(realization_count=20, show_plots=True, plot_realization=0):
    model = Model()
    model.source = SineSource()
    model.response = ArrayGain()
    model.settings.timestep = 1.  # seconds
    model.settings.stop_time = 120.

    # Each parameter and I/O value has shape (realization_count,).
    model.source.amplitude = np.linspace(1., 2., realization_count)
    model.response.gain = np.linspace(0.5, 1.5, realization_count)
    model.initialize()
    model.connect(model.source.signal, model.response.signal)

    if show_plots:
        model.settings.global_plotter_args = dict(show_live=False, time_units="sec", grid=False)

        # Show the middle 50% of the realizations, plus their mean.
        model.add_plotter(model.response.result, y1label="Output",
                          tab_title="Percentile band",
                          array_config=dict(mode="fill_between", percentiles=(25, 75),
                                            fill_alpha=0.4))

        # Show every realization as a faint line, plus the mean.
        model.add_plotter(model.response.result, y1label="Output",
                          tab_title="Spaghetti",
                          array_config=dict(mode="spaghetti", spaghetti_alpha=0.4))

        # Select any zero-based index; a list can select several realizations.
        model.add_plotter(model.response.result, y1label="Output",
                          tab_title="Selected realization",
                          array_config=dict(mode="indices", indices=[plot_realization]))
    return model


def main():
    model = create_model()
    while model.time < model.settings.stop_time:
        model.step()
    model.wait_for_plots()
    return model


if __name__ == "__main__":
    main()
