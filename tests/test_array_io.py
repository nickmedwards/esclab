"""Vector signals through simulation, history, and live/deferred plots."""
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from esclab.plotting import OnlinePlotter
from esclab.simulate import Component, Connection, Model


class VectorSource(Component):
    scale = Component.Parameter(np.array([1., 2., 4.]))
    output = Component.Output()

    def presim_setup(self):
        self.output = self.scale.v.copy()

    def calculate(self):
        self.output.v[:] = self.scale.v * (1 + self.model.time)


class VectorSink(Component):
    inlet = Component.Input(np.zeros(3))
    result = Component.Output()

    def calculate(self):
        self.result = self.inlet * 2


@pytest.mark.parametrize('row', [False, True])
def test_simulation_vector_connections_history_and_parameter_defaults(row):
    model = Model()
    model.source = VectorSource()
    model.sink = VectorSink()
    other = VectorSource()
    model.source.scale[0] = 3
    assert other.scale[0] == 1
    if row:
        model.source.scale = model.source.scale.v.reshape(1, -1)
        model.sink.inlet = np.zeros((1, 3))
    model.connect(model.source.output, model.sink.inlet, log_n_iter=2)
    for _ in range(3):
        model.step()
    expected = np.array([3., 2., 4.])
    if row:
        expected = expected[None, :]
    history = model.historian[model.source.output.name]
    for i, snapshot in enumerate(history):
        np.testing.assert_allclose(snapshot, expected * (i + 1))
    np.testing.assert_allclose(model.sink.result.v, expected * 6)
    model.source.output.v[:] = -1
    np.testing.assert_allclose(history[0], expected)
    assert model.sink.inlet.connection.last_step_is_converged
    assert model.sink.inlet.connection.iter_log.shape == (2, 3) + expected.shape
    np.testing.assert_allclose(np.array(model.source.scale, copy=True), expected)


def test_connection_requires_every_element_and_copies_snapshots():
    source = Component.Output()
    source.v = np.array([1., 2., 3.])
    inlet = Component.Input()
    inlet.connection = Connection(source, 0., 0.01, 2, 0.5)
    assert not inlet.update_from_connection()
    source.v[2] = 5
    assert not inlet.update_from_connection()
    np.testing.assert_allclose(inlet.v, [1, 2, 4])
    np.testing.assert_allclose(inlet.connection.err_abs, [0, 0, 1])
    assert inlet.update_from_connection()
    source.v[:] = 9
    np.testing.assert_allclose(inlet.v, [1, 2, 5])
    np.testing.assert_allclose(inlet.connection.source_last_value, [1, 2, 5])
    # Rolling logs must roll time only, preserving value/error/member axes.
    np.testing.assert_allclose(inlet.connection.iter_log[-1, 0], [1, 2, 5])


@pytest.fixture
def plots():
    created = []
    yield created
    for plotter in created:
        if hasattr(plotter, 'win'):
            OnlinePlotter.tab_widget.removeTab(OnlinePlotter.tab_widget.indexOf(plotter.win))
            OnlinePlotter.instances.remove(plotter)
            plotter._tip_label.close()
            plotter.win.close()


@pytest.mark.parametrize('live', [True, False])
@pytest.mark.parametrize('mode', ['fill_between', 'spaghetti', 'indices'])
def test_array_rendering_both_axes_and_buffer_growth(live, mode, plots, monkeypatch):
    output = Component.Output()
    output.name = 'ensemble'
    output.units = 'kW'
    config = dict(mode=mode, percentiles=(10, 90))
    if mode == 'indices':
        config.update(indices=[0, 2])
    plotter = OnlinePlotter([output, 7.], [output], None, None, 'Left', 'Right',
                            100, 1, show_live=live, time_units='hr',
                            array_config=config)
    plots.append(plotter)
    plotter.preallocate(1)
    rows = []
    for t in range(3):
        output.v = np.array([[1., 3., 9.]]) + t
        rows.append(output.v[0].copy())
        plotter.log_step(t * 3600, timestep=1800)
    if not live:
        plotter._finalize()
    expected = np.array(rows)
    np.testing.assert_allclose(plotter.y1_data[0][:3], expected)
    np.testing.assert_allclose(plotter.y2_data[0][:3], expected)
    np.testing.assert_allclose(plotter.y1_lines[1].yData, [7, 7, 7])
    for index, mean in [(0, plotter.y1_lines[0]), (2, plotter.y2_lines[0])]:
        np.testing.assert_allclose(mean.yData, expected.mean(axis=1))
        members, bounds = plotter._array_artists[index]
        assert mean.isVisible() == (mode != 'indices')
        if mode == 'fill_between':
            percentiles = np.percentile(expected, [10, 90], axis=1)
            np.testing.assert_allclose(bounds[0].yData, percentiles[0])
            np.testing.assert_allclose(bounds[1].yData, percentiles[1])
            assert bounds[2].brush().color().alphaF() == pytest.approx(0.2, abs=0.01)
            assert bounds[2].path().contains(QtCore.QPointF(1, 5))
            # Verify a pixel inside the band is actually filled at the chosen
            # opacity; correct percentile data alone can hide an empty path.
            band_image = QtGui.QImage(32, 24, QtGui.QImage.Format_ARGB32)
            band_image.fill(QtGui.QColor('white'))
            band_painter = QtGui.QPainter(band_image)
            try:
                band_painter.scale(10, 2)
                bounds[2].paint(band_painter, QtWidgets.QStyleOptionGraphicsItem())
            finally:
                band_painter.end()
            color = bounds[2].brush().color()
            expected_rgb = np.array(color.getRgb()[:3]) * 0.2 + 255 * 0.8
            np.testing.assert_allclose(band_image.pixelColor(10, 10).getRgb()[:3], expected_rgb, atol=1)
        else:
            assert [i for i, _ in members] == ([0, 1, 2] if mode == 'spaghetti' else [0, 2])
            for member, line in members:
                np.testing.assert_allclose(line.yData, expected[:, member])
    plotter._fit_to_full_data()
    plotter.ax1.setXRange(1.4, 1.6, padding=0)
    plotter._render_visible_window()
    np.testing.assert_allclose(plotter.y1_lines[0].xData, [1, 2])
    plotter._fit_to_full_data()
    OnlinePlotter.app.processEvents()
    # Paint legends explicitly: offscreen widgets may not receive paint events.
    image = QtGui.QImage(32, 24, QtGui.QImage.Format_ARGB32)
    painter = QtGui.QPainter(image)
    try:
        for legend in (plotter.legend_y1, plotter.legend_y2):
            for sample, _ in legend.items:
                sample.paint(painter)
    finally:
        painter.end()
    plotter._plot_tooltip_enabled = True
    position = plotter.ax1.vb.mapViewToScene(QtCore.QPointF(1, 5))
    percentile_calls = []
    original_percentile = np.percentile

    def track_percentile(*args, **kwargs):
        percentile_calls.append(np.asarray(args[0]).shape)
        return original_percentile(*args, **kwargs)

    monkeypatch.setattr(np, 'percentile', track_percentile)
    plotter._on_strip_mouse_move((position,))
    summary = 'mean 5.333; p10 2.4; p90 8.8; min 2; max 10 kW'
    assert plotter._tip_label.text().count(summary) == 2
    # Compute on each axis's hovered row only; moving within that timestep
    # reuses the visible tooltip rather than recomputing the statistics.
    assert percentile_calls == [(3,), (3,)]
    plotter._on_strip_mouse_move((position,))
    assert percentile_calls == [(3,), (3,)]


@pytest.mark.parametrize('config', [
    {'mode': 'unknown'}, {'percentiles': [95, 5]}, {'percentiles': [-1, 95]},
    {'percentiles': [1]}, {'indices': []}, {'indices': [-1]},
    {'indices': [0.5]}, {'mode': 'indices'}, {'fill_alpha': 2},
    {'spaghetti_width': -1}, {'mean_width': float('nan')}, {'typo': 1},
])
def test_invalid_config_rejected_before_widgets(config, monkeypatch):
    monkeypatch.setattr(OnlinePlotter, '_build_widgets', lambda self: pytest.fail('widgets created'))
    with pytest.raises((ValueError, TypeError)):
        OnlinePlotter([1.], None, None, None, '', '', 100, 1, array_config=config)


def test_model_global_and_per_signal_config_and_shape_validation():
    model = Model()
    model.initialize()
    output = Component.Output()
    model.settings.global_plotter_args = dict(show_live=False, array_config={'mode': 'spaghetti'})
    model.add_plotter(output)
    assert model._plotters[-1].array_config[0]['mode'] == 'spaghetti'
    model.add_plotter(output, output, array_config=[None, {'mode': 'indices', 'indices': 1}])
    plotter = model._plotters[-1]
    output.v = np.ones((2, 3))
    with pytest.raises(ValueError, match='shaped'):
        plotter.log_step(0)
    output.v = np.ones(1)
    with pytest.raises(ValueError, match='out of range'):
        plotter.log_step(0)
    assert plotter._fill_idx == 0
    output.v = np.ones(3)
    plotter.log_step(0)
    output.v = np.ones(4)
    with pytest.raises(ValueError, match='shape changed'):
        plotter.log_step(1)


def test_selected_indices_fit_excludes_hidden_members(plots):
    output = Component.Output()
    output.v = np.array([2., 3., 10000.])
    plotter = OnlinePlotter([output], None, None, None, '', '', 100, 1,
                            show_live=False, array_config={'mode': 'indices', 'indices': [0, 1]})
    plots.append(plotter)
    plotter.log_step(0)
    plotter._finalize()
    plotter._fit_to_full_data()
    low, high = plotter.ax1.viewRange()[1]
    assert low < 2 < 3 < high < 4


def test_sample_model_short_run(plots):
    from esclab.models.sample_array_io import create_model
    model = create_model(realization_count=3, show_plots=True, plot_realization=1)
    plots.extend(model._plotters)
    for _ in range(3):
        model.step()
    assert [p.array_config[0]['mode'] for p in model._plotters] == [
        'fill_between', 'spaghetti', 'indices']
    for plotter in model._plotters:
        plotter._finalize()
        assert plotter.y1_data[0][:3].shape == (3, 3)
        assert np.isfinite(plotter.y1_data[0][:3]).all()
    members, _ = model._plotters[-1]._array_artists[0]
    np.testing.assert_allclose(members[0][1].yData, model._plotters[-1].y1_data[0][:3, 1])
