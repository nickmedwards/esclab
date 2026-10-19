import pytest
from pyqtgraph.Qt import QtCore, QtGui

from esclab.plotting import OnlinePlotter
from esclab.simulate import Model


def make_plotter(**kwargs):
    return OnlinePlotter([1., 2.], [3.], None, None, 'Primary', 'Secondary',
                         100, 1, **kwargs)


@pytest.mark.parametrize('stats', [False, True])
def test_global_appearance_and_explicit_overrides(stats):
    model = Model()
    model.initialize()
    model.settings.global_plotter_args = dict(
        show_live=False, font_size=14, font_color='black', bg_color='white',
        line_weights=3, line_styles='s--', line_colors='tab:red')
    add = model.add_plotter_simstats if stats else lambda **kw: model.add_plotter(1., 2., **kw)
    add()
    inherited = model._plotters[-1]
    assert inherited.plot_font_size == 14
    assert inherited.font_color.name() == '#000000'
    assert inherited.bg_color.name() == '#ffffff'
    for options in inherited._line_options:
        assert options['pen'].widthF() == 3
        assert options['pen'].style() == QtCore.Qt.DashLine
        assert options['pen'].color().name() == '#d62728'
        assert not options['symbol'].isEmpty()
    add(font_size=10, font_color='blue', bg_color=None,
        line_weights=2, line_styles=None, line_colors=None)
    explicit = model._plotters[-1]
    assert explicit.plot_font_size == 10
    assert explicit.bg_color is None
    assert explicit._line_options[0]['pen'].style() == QtCore.Qt.SolidLine
    assert explicit._line_options[1]['pen'].style() == QtCore.Qt.DashLine
    assert inherited.plot_font_size == 14


@pytest.mark.parametrize('live', [False, True])
def test_rendered_series_and_fonts(live, monkeypatch):
    monkeypatch.setattr(OnlinePlotter, 'font_size_pt', 10)
    plotter = make_plotter(show_live=live, font_size=15, font_color='black',
                           bg_color='white', line_weights=[1, 2, 3],
                           line_styles=['--', ':', 's--'],
                           line_colors=['red', 'green', (0., 0., 1.)])
    try:
        plotter.log_step(1.)
        if not live:
            plotter._finalize()
        for line, width, color, style in zip(
            plotter.y1_lines + plotter.y2_lines, [1, 2, 3],
            ['#ff0000', '#008000', '#0000ff'],
            [QtCore.Qt.DashLine, QtCore.Qt.DotLine, QtCore.Qt.DashLine],
        ):
            assert line.opts['pen'].widthF() == width
            assert line.opts['pen'].color().name() == color
            assert line.opts['pen'].style() == style
        assert isinstance(plotter.y2_lines[0].opts['symbol'], QtGui.QPainterPath)
        assert plotter.win.backgroundBrush().color().name() == '#ffffff'
        for name in ('left', 'bottom', 'right'):
            axis = plotter.ax1.getAxis(name)
            assert axis.style['tickFont'].pointSizeF() == 15
            assert axis.textPen().color().name() == '#000000'
        for legend in (plotter.legend_y1, plotter.legend_y2):
            for _, label in legend.items:
                assert label.opts['color'].name() == '#000000'
                assert label.opts['size'] == '15pt'
        OnlinePlotter.adjust_font_size(1)
        assert plotter.plot_font_size == 16
        assert plotter.ax1.getAxis('right').style['tickFont'].pointSizeF() == 16
    finally:
        if hasattr(plotter, 'win'):
            OnlinePlotter.tab_widget.removeTab(OnlinePlotter.tab_widget.indexOf(plotter.win))
            OnlinePlotter.instances.remove(plotter)
            plotter._tip_label.close()
            plotter.win.close()


@pytest.mark.parametrize('option', ['line_weights', 'line_styles', 'line_colors'])
def test_wrong_series_count_rejected(option):
    with pytest.raises(ValueError, match='exactly 3 values'):
        make_plotter(show_live=False, **{option: [1, 2]})


@pytest.mark.parametrize('options', [
    {'font_size': 0}, {'font_size': float('nan')}, {'line_weights': -1},
    {'line_weights': float('inf')}, {'line_styles': 'invalid'},
    {'line_colors': 'invalid'}, {'bg_color': 'invalid'},
])
def test_invalid_appearance_rejected_before_widgets(options, monkeypatch):
    monkeypatch.setattr(OnlinePlotter, '_build_widgets', lambda self: pytest.fail('created widgets'))
    with pytest.raises(ValueError):
        make_plotter(**options)


def test_marker_only_embedded_color_and_rgb_scalar():
    plotter = make_plotter(show_live=False, line_styles='ro')
    assert all(o['pen'] is None for o in plotter._line_options)
    assert all(o['symbolPen'].color().name() == '#ff0000' for o in plotter._line_options)
    plotter = make_plotter(show_live=False, line_styles='ro', line_colors=(0., 0., 1.))
    assert all(o['symbolPen'].color().name() == '#0000ff' for o in plotter._line_options)


@pytest.mark.parametrize('global_args', [False, True])
def test_nested_axis_options_match_flat_options(global_args):
    nested = dict(line_weights=[[1, 2], [3]],
                  line_styles=[['--', ':'], ['s--']],
                  line_colors=[['red', (0., 1., 0.)], ['blue']])
    model = Model()
    if global_args:
        model.settings.global_plotter_args = nested
    model.add_plotter([1., 2.], [3.], show_live=False,
                      **({} if global_args else nested))
    plotter = model._plotters[-1]
    flat = make_plotter(show_live=False, line_weights=[1, 2, 3],
                        line_styles=['--', ':', 's--'],
                        line_colors=['red', (0., 1., 0.), 'blue'])
    for actual, expected in zip(plotter._line_options, flat._line_options):
        assert actual['pen'] == expected['pen']
        assert actual['symbol'] == expected['symbol']
        assert actual['symbolBrush'] == expected['symbolBrush']
    assert nested['line_weights'] == [[1, 2], [3]]


def test_single_entry_lists_broadcast():
    plotter = make_plotter(show_live=False, line_weights=[4],
                           line_styles=[':'], line_colors=['red'])
    assert all(o['pen'].widthF() == 4 for o in plotter._line_options)
    assert all(o['pen'].style() == QtCore.Qt.DotLine for o in plotter._line_options)
    assert all(o['pen'].color().name() == '#ff0000' for o in plotter._line_options)


def test_nested_options_without_secondary_axis():
    plotter = OnlinePlotter([1., 2.], None, None, None, '', '', 100, 1,
                            show_live=False, line_colors=[['red', 'blue'], []])
    assert [o['pen'].color().name() for o in plotter._line_options] == ['#ff0000', '#0000ff']


@pytest.mark.parametrize('groups', [[[1], [2, 3]], [[1, 2, 3], []], [[1, 2], 3]])
def test_nested_options_validate_each_axis(groups):
    with pytest.raises(ValueError, match='2 y1 and 1 y2'):
        make_plotter(show_live=False, line_weights=groups)


def test_flat_rgb_lists_remain_individual_colors():
    plotter = make_plotter(show_live=False,
                           line_colors=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    assert [o['pen'].color().name() for o in plotter._line_options] == [
        '#ff0000', '#00ff00', '#0000ff']


@pytest.mark.parametrize('live', [True, False])
@pytest.mark.parametrize('grid,horizontal,vertical', [
    (True, True, True), (True, True, False), (True, False, True),
    (True, False, False), (False, True, True),
])
def test_grid_visibility(live, grid, horizontal, vertical):
    plotter = make_plotter(show_live=live, grid=grid,
                           grid_horizontal=horizontal, grid_vertical=vertical)
    try:
        if not live:
            plotter._finalize()
        assert bool(plotter.ax1.getAxis('bottom').grid) == (grid and vertical)
        assert bool(plotter.ax1.getAxis('left').grid) == (grid and horizontal)
    finally:
        if hasattr(plotter, 'win'):
            OnlinePlotter.tab_widget.removeTab(OnlinePlotter.tab_widget.indexOf(plotter.win))
            OnlinePlotter.instances.remove(plotter)
            plotter._tip_label.close()
            plotter.win.close()


@pytest.mark.parametrize('stats', [False, True])
def test_global_grid_options_and_direct_overrides(stats):
    model = Model()
    model.initialize()
    model.settings.global_plotter_args = dict(
        show_live=False, grid=False, grid_horizontal=False, grid_vertical=False)
    add = model.add_plotter_simstats if stats else lambda **kw: model.add_plotter(1., **kw)
    add()
    inherited = model._plotters[-1]
    assert (inherited.grid, inherited.grid_horizontal, inherited.grid_vertical) == (False, False, False)
    add(grid=True, grid_horizontal=True, grid_vertical=True)
    explicit = model._plotters[-1]
    assert (explicit.grid, explicit.grid_horizontal, explicit.grid_vertical) == (True, True, True)
    assert model.settings.global_plotter_args['grid'] is False


def test_grid_defaults_remain_enabled():
    plotter = make_plotter(show_live=False)
    assert plotter.grid is plotter.grid_horizontal is plotter.grid_vertical is True
