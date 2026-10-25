"""Exercise real Qt legend clicks during live updates and deferred viewing."""
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest

from esclab.plotting import OnlinePlotter
from esclab.simulate import Component


@pytest.fixture
def plotters():
    instances = []
    yield instances
    for plotter in instances:
        OnlinePlotter.tab_widget.removeTab(OnlinePlotter.tab_widget.indexOf(plotter.win))
        OnlinePlotter.instances.remove(plotter)
        plotter._tip_label.close()
        plotter.win.close()


def click(plotter, item):
    position = plotter.win.mapFromScene(item.mapToScene(item.boundingRect().center()))
    QtTest.QTest.mouseClick(plotter.win.viewport(), QtCore.Qt.LeftButton, pos=position)
    OnlinePlotter.app.processEvents()


@pytest.mark.parametrize('live', [True, False])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('mode', ['scalar', 'fill_between', 'spaghetti', 'indices'])
def test_click_labels_and_swatches_preserves_visibility_and_history(live, axis, mode, plotters):
    output = Component.Output()
    output.name = 'signal'
    output.v = 10000. if mode == 'scalar' else np.array([2., 4., 10000.])
    reference = Component.Output()
    reference.name = 'reference'
    reference.v = 5.
    config = None if mode == 'scalar' else dict(mode=mode)
    if mode == 'indices':
        config['indices'] = [0, 2]
    plotter = OnlinePlotter([output, reference], [output, reference],
                            None, None, 'Left', 'Right', 100, 1,
                            show_live=live, array_config=config)
    plotters.append(plotter)
    plotter.log_step(0)
    plotter.log_step(1)
    if not live:
        plotter._finalize()
    OnlinePlotter.tab_widget.setCurrentWidget(plotter.win)
    OnlinePlotter.app.processEvents()
    legend = (plotter.legend_y1, plotter.legend_y2)[axis]
    target = (plotter.ax1, plotter.ax2)[axis]
    entries = [(sample, label) for sample, label in legend.items if label.text != 'reference']
    if mode == 'spaghetti':
        assert {label.text for _, label in entries} == {'signal realizations', 'signal mean'}

    for sample, label in entries:
        assert sample.item.isVisible()
        click(plotter, label)
        assert not sample.item.isVisible()
        assert label.opacity() == pytest.approx(0.4)
        # Toggling each component must leave every other entry alone.
        others = [(other.item, other.item.isVisible()) for other, _ in legend.items if other is not sample]
        click(plotter, sample)
        assert sample.item.isVisible()
        assert label.opacity() == 1.
        assert all(item.isVisible() == visible for item, visible in others)
        click(plotter, label)
        assert not sample.item.isVisible()

    output.v = output.v * 2
    plotter.log_step(2)
    plotter._render_visible_window()
    assert all(not sample.item.isVisible() for sample, _ in entries)
    plotter.apply_font_size()
    assert all(label.opacity() == pytest.approx(0.4) for _, label in entries)

    # Fit only the remaining reference, then preserve Y when it too is hidden.
    plotter._fit_to_full_data()
    low, high = target.viewRange()[1]
    assert 4 < low < 5 < high < 6
    reference_sample, reference_label = next((s, l) for s, l in legend.items if l.text == 'reference')
    click(plotter, reference_label)
    plotter._fit_to_full_data()
    np.testing.assert_allclose(target.viewRange()[1], [low, high])
    assert not reference_sample.item.isVisible()

    # Restore the data and verify the hidden interval was logged on both axes.
    for sample, label in entries:
        click(plotter, label)
        assert sample.item.isVisible()
    expected = np.array([output.v / 2, output.v / 2, output.v])
    np.testing.assert_allclose(plotter.y1_data[0][:3], expected)
    np.testing.assert_allclose(plotter.y2_data[0][:3], expected)
    if mode == 'spaghetti':
        index = 0 if axis == 0 else 2
        members, _ = plotter._array_artists[index]
        assert all(line.isVisible() for _, line in members)
        for member, line in members:
            np.testing.assert_allclose(line.yData, expected[:, member])
