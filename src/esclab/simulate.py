import copy
from collections import deque
import numpy as np
import sys
import time 
import sys
import os
                    
from esclab.plotting import OnlinePlotter, NetworkTopologyView

_NOT_SET = object()

# ------------------------------------------------------------------------
class Connection:
    """
    
    """
    def __init__(self, source, tol_rel, tol_abs, log_n_iter, learn_rate, solve_group=None):

        self.tol_rel = tol_rel
        self.tol_abs = tol_abs 
        self.learn_rate = learn_rate
        
        BIG = 9e19
        self.err_rel = BIG
        self.err_abs = BIG

        self.source = source
        self.source_last_value = float('nan')
        self.solve_group = solve_group

        # Last completed-step snapshot for UI/debug views.
        self.has_step_history = False
        self.last_step_is_converged = None
        self.last_step_n_iter = None
        self.last_step_err_abs = None
        self.last_step_err_rel = None

        self.log_n_iter = log_n_iter

        self.reset_for_step()
        self.clear_iteration_log()
        return
    
    def reset_for_step(self):
        if hasattr(self, 'n_iter'):
            self.has_step_history = True
            self.last_step_is_converged = self.is_converged
            self.last_step_n_iter = self.n_iter
            self.last_step_err_abs = self.err_abs
            self.last_step_err_rel = self.err_rel

        self.is_converged = False 

    def clear_iteration_log(self):
        self.n_iter = 0
        # [[value, abs err, rel err],]
        self.iter_log = np.zeros((max(self.log_n_iter,1),3))

    def compute_new_value(self):
        # Check if previous source value is nan, if not, use it to compute a new value with the learn rate. 
        # If it is nan, return the current source value.
        if not np.any(np.isnan(self.source_last_value)):
            return self.source_last_value + (self.source.v - self.source_last_value)*self.learn_rate
        else: 
            # nan case
            return self.source.v

    def check_convergence(self, new_value):
        # new_value = self.source.v 
        old_value = self.source_last_value
            
        self.err_abs = np.abs(new_value-old_value)
        scale = np.maximum(np.maximum(np.abs(old_value), np.abs(new_value)), 1.e-19)
        self.err_rel = self.err_abs/scale
        self.is_converged = np.all(self.err_abs <= (self.tol_abs + self.tol_rel * scale))

        if self.log_n_iter > 0:
            # Keep one [value, absolute error, relative error] record per
            # element for vector signals, preserving the scalar log layout.
            la = np.stack(np.broadcast_arrays(self.source.v, self.err_abs, self.err_rel))
            if self.iter_log.shape[1:] != la.shape:
                self.iter_log = np.zeros((max(self.log_n_iter, 1),) + la.shape)
            if self.n_iter < self.log_n_iter: 
                self.iter_log[self.n_iter,:] = la
            else:
                self.iter_log = np.roll(self.iter_log, -1, axis=0)
                self.iter_log[-1,:] = la

        self.n_iter += 1        
        return self.is_converged
    
# =========================================================================================   

class Component:
    # ------------------------------------------------------------------------
    class __iop_base:
        """
        Base for Input, Parameter, and Output classes.

        Contains the value (v), units, name, but no connection status.
        Numeric values may be scalars or NumPy vectors, with shape (m,)
        or (1, m). Arithmetic, NumPy conversion, iteration, truth testing,
        and copying delegate to the stored value without requiring .v.
        """
        def __init__(self):
            self.name = ''
            self.units = ''
            self.v = float('nan')
            self.is_connected = False
            return
        def __repr__(self):
            return f'{self.name} = {self.v} {self.units}'
        
        
        def set(self, value=None, units=None, name=None):
            if value is not None: 
                self.v = value
            if units is not None:
                self.units = units
            if name is not None:
                self.name = name

        # Numeric protocol — forward arithmetic and comparisons to self.v so
        # instances can be used directly in expressions without .v
        # Note that assignment (itema.v = itemb.v) is handled in the component
        # __setattr__ method. 
        def __float__(self):             return float(self.v)
        def __int__(self):               return int(self.v)
        def __bool__(self):              return bool(self.v)
        def __array__(self, dtype=None, copy=None):
            if copy is None:
                return np.asarray(self.v, dtype=dtype)
            return np.array(self.v, dtype=dtype, copy=copy)

        def copy(self):
            """Return a shallow value copy, with independent data for NumPy arrays."""
            return copy.copy(self.v)

        def __neg__(self):               return -self.v
        def __pos__(self):               return +self.v
        def __abs__(self):               return abs(self.v)

        def __add__(self, o):            return self.v + o
        def __radd__(self, o):           return o + self.v
        def __sub__(self, o):            return self.v - o
        def __rsub__(self, o):           return o - self.v
        def __mul__(self, o):            return self.v * o
        def __rmul__(self, o):           return o * self.v
        def __truediv__(self, o):        return self.v / o
        def __rtruediv__(self, o):       return o / self.v
        def __floordiv__(self, o):       return self.v // o
        def __rfloordiv__(self, o):      return o // self.v
        def __mod__(self, o):            return self.v % o
        def __rmod__(self, o):           return o % self.v
        def __pow__(self, o):            return self.v ** o
        def __rpow__(self, o):           return o ** self.v

        def __lt__(self, o):             return self.v <  (o.v if hasattr(o, 'v') else o)
        def __le__(self, o):             return self.v <= (o.v if hasattr(o, 'v') else o)
        def __gt__(self, o):             return self.v >  (o.v if hasattr(o, 'v') else o)
        def __ge__(self, o):             return self.v >= (o.v if hasattr(o, 'v') else o)
        def __eq__(self, o):             return self.v == (o.v if hasattr(o, 'v') else o)
        def __ne__(self, o):             return self.v != (o.v if hasattr(o, 'v') else o)
        __hash__ = object.__hash__  # restore identity hash (Python sets it to None when __eq__ is defined)

        def __len__(self):               return len(self.v)
        def __iter__(self):              return iter(self.v)
        def __getitem__(self, idx):      return self.v[idx]
        def __setitem__(self, idx, val): self.v[idx] = val
    # ------------ end class __io_base ----------------------------------------
    class __io_base(__iop_base):
        """
        Base for Input and Output classes.

        Contains connection status and logic for updating from connections.
        """
        def __init__(self):
            super().__init__()
            self.connection = None  #instance of class Connection()
            return
    # ------------ end class __io_base ----------------------------------------

    # -------------------------------------------------------------------------
    class Input(__io_base):
        def __init__(self, initial_value=1.):
            super().__init__()
            self.connection = None  #instance of class Connection()
            self.v = initial_value

        def update_from_connection(self):
            if not self.connection == None:
                #check for nan
                if not np.any(np.isnan(self.connection.source.v)):
                    new_value = self.connection.compute_new_value()
                    converged = self.connection.check_convergence(new_value)
                    source = self.connection.source.v
                    self.connection.source_last_value = source.copy() if isinstance(source, np.ndarray) else source
                    self.v = new_value.copy() if isinstance(new_value, np.ndarray) else new_value
                    return converged
                else:
                    return False
    # ------------ end class Input -------------------------------------------
    # ------------------------------------------------------------------------
    class Output(__io_base):
        def __init__(self):
            super().__init__()
    # ----------- end class Output ------------------------------------------
    # ------------------------------------------------------------------------
    class Parameter(__iop_base):
        def __init__(self, value=float('nan'), units='', std_dev = None):
            super().__init__()
            self.v = value
            self.units = units
            self.std_dev = std_dev
    # ----------- end class Parameter -----------------------------------------
    
    # ------------------------------------------------------------------------
    #  Component class methods
    # ------------------------------------------------------------------------
    def __init__(self):
        """
        Define inputs, outputs, and parameters. 
        """
        # Create per-instance copies of all Input/Output/Parameter **class** attributes.
        # Without this, subclass instances that don't manually instantiate new members of the 
        # same name will share the same mutable objects, causing auto_assign_names() and 
        # value assignments share between multiple class instances.
        for cls in type(self).__mro__:
            for attr_name, attr_val in vars(cls).items():
                if isinstance(attr_val, (Component.Input, Component.Output, Component.Parameter)):
                    if attr_name not in self.__dict__:
                        instance_item = copy.copy(attr_val)
                        if isinstance(attr_val.v, np.ndarray):
                            instance_item.v = attr_val.v.copy()
                        object.__setattr__(self, attr_name, instance_item)
        self.trnsys_type = ''     # TRNSYS type number, if applicable <string>
        self.name = ''
        self.coupled_eqs = None  # Set by Model during the matrix-build phase; None otherwise.
        # Caches for get_inputs()/get_outputs(), avoiding repeated dir() reflection scans.
        self._io_all_inputs = None
        self._io_all_outputs = None
        self._io_locked = False
        self._io_locked_connected_inputs = None
        self._io_locked_connected_outputs = None
        return 
    
    def __scan_io_items(self, item_type):
        """
        Scan dir(self) once for all inputs or outputs of this component, depending on the
        item_type argument. Only called the first time each item_type is requested; results
        are memoized by get_inputs()/get_outputs().

        Parameters
        ----------
        item_type : type
            Component.Input or Component.Output
        """
        io_list = []
        for item_name in dir(self):
            item = getattr(self, item_name)
            if isinstance(item, item_type):
                io_list.append(item)
        return tuple(io_list)

    def _lock_io_cache(self):
        """
        Called by Model once stepping begins, after which connect() disallows new connections.
        Precomputes the connected-only tuples so the per-iteration hot path avoids any filtering.
        """
        if self._io_all_inputs is None:
            self._io_all_inputs = self.__scan_io_items(Component.Input)
        if self._io_all_outputs is None:
            self._io_all_outputs = self.__scan_io_items(Component.Output)
        self._io_locked_connected_inputs = tuple(i for i in self._io_all_inputs if i.is_connected)
        self._io_locked_connected_outputs = tuple(o for o in self._io_all_outputs if o.is_connected)
        self._io_locked = True

    def get_inputs(self, connected_only=False):
        """
        Helper function to collect all inputs from this component.

        Parameters
        ----------
        connected_only : bool
            If True, only return inputs that are connected to another component. 
            If False, return all inputs regardless of connection status.
        """
        if self._io_all_inputs is None:
            self._io_all_inputs = self.__scan_io_items(Component.Input)
        if not connected_only:
            return self._io_all_inputs
        if self._io_locked:
            return self._io_locked_connected_inputs
        return tuple(i for i in self._io_all_inputs if i.is_connected)

    def get_outputs(self, connected_only=False):
        """
        Helper function to collect all outputs from this component.
        
        Parameters
        ----------
        connected_only : bool
            If True, only return outputs that are connected to another component. 
            If False, return all outputs regardless of connection status.
        """
        if self._io_all_outputs is None:
            self._io_all_outputs = self.__scan_io_items(Component.Output)
        if not connected_only:
            return self._io_all_outputs
        if self._io_locked:
            return self._io_locked_connected_outputs
        return tuple(o for o in self._io_all_outputs if o.is_connected)
    
    def set_values_from_dict(self, values_dict):
        """
        Helper function to set values of inputs and parameters from a dictionary. 
        The keys of the dictionary should match the attribute names of the inputs and parameters.

        Parameters
        ----------
        values_dict : dict
            Dictionary containing values to set for inputs and parameters. Keys should match attribute names.
        """
        for key, value in values_dict.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(attr, Component.Input) or isinstance(attr, Component.Parameter):
                    attr.set(value=value)
                else:
                    raise RuntimeError(f"Attribute '{key}' is not an input or parameter.")
            else:
                raise RuntimeError(f"Component does not have an attribute named '{key}'.")
    
    def auto_assign_names(self):
        """
        Called automatically by the model during initialization.

        Automatically assign names to all inputs and outputs of this component based on the 
        component's name and the attribute name of the input/output.
        Make sure all inputs/outputs have a name assigned
        """
        # make sure the current component has a name assigned

        allnames = []
        for member in dir(self):
            try:
                mo = getattr(self, member)
            except Exception:
                continue
            if isinstance(mo, Component.Input) or isinstance(mo, Component.Output):
                if self.name == '':
                    raise RuntimeError(f"Internal error: component {type(self).__name__} has no name assigned. Please assign a name to the component before calling auto_assign_names().")
                mo.name = self.name + '.' + member
                if isinstance(mo, Component.Output):
                    allnames.append(mo.name)
        return allnames

    def __setattr__(self, name, value):
        # Override setattr to allow setting the value of an input or parameter directly, while 
        # still allowing assignment of new attributes. If the attribute being set already 
        # exists and is an Input or Parameter, update its value instead of replacing the attribute.
        try:
            existing = object.__getattribute__(self, name)
        except AttributeError:
            existing = None
        if isinstance(existing, (Component.Input, Component.Output, Component.Parameter)):
            existing.v = value.v if isinstance(value, (Component.Input, Component.Output, Component.Parameter)) else value
        else:
            object.__setattr__(self, name, value)

    def presim_setup(self, **kwargs):
        """
        Do all calculations needed for the simulation here. This is called after all models have been
        instantiated and connected, but before the first time step. Data from other components is not available
        at this step. Use this for any calculations that should only be done once at the beginning of the simulation.
        
        Overwrite in child class, as needed. 

        kwargs can be used to pass any relevant data from the model or other components that is needed for setup.
        """
        pass
    
    def calculate(self):
        """
        Do regular calculations for this component. This is where the main logic of the component goes, 
        and where outputs are calculated from inputs and parameters.
        
        Within the calculate step, the following flags are available to control the flow of calculations:
        ---------------------------------------------------------------------------------------------------------
        is_first_step       | True only on the first step() call of the simulation. Use for any calculations that 
                            | should only be done once at the beginning of the simulation.
        ---------------------------------------------------------------------------------------------------------
        is_first_iteration  | True only on the first iteration of each step() call. Use for any calculations that 
                            | should only be done once per time step, such as calculations that should be done 
                            | in order of components.
        ---------------------------------------------------------------------------------------------------------
        is_converged        | True only on the final iteration of each step() call, after convergence is reached. 
                            | Use for any calculations that should only be done once per time step, and that 
                            | require convergence to be reached first.
        ---------------------------------------------------------------------------------------------------------
        """
        pass 
    
    def postsim_calcs(self):
        """
        Calculations done after the final time step of the simulation. Data from all time steps is available.
        Called once by Model.finalize(), in component execution order, including when
        the final step reaches the iteration limit without converging. Results stored
        here do not retroactively update historian samples or plots.
        """
        pass


# =========================================================================================
class NetworkEquationContext:
    """Context object passed to components during the matrix-build phase of each iteration.

    API provider for components to add their linear equations into the matrix solver.
    """

    def __init__(self, unknown_index, n_unknown, _add_row_fn, input_owner_by_id=None):
        self._unknown_index = unknown_index
        self._n_unknown = n_unknown
        self._add_row_fn = _add_row_fn
        self._input_owner_by_id = {} if input_owner_by_id is None else input_owner_by_id

    def _describe_input_port(self, input_port):
        input_name = str(getattr(input_port, "name", "") or "").strip()
        owner = self._input_owner_by_id.get(id(input_port))

        instance_name = ""
        if owner is not None:
            instance_name = str(getattr(owner, "name", "") or "").strip()
            if not instance_name:
                raise RuntimeError(
                    f"Internal error: component {type(owner).__name__} has no name assigned. Please assign a name to the component before calling auto_assign_names()."
                )

        if not input_name and owner is not None:
            for attr_name in dir(owner):
                try:
                    if getattr(owner, attr_name) is input_port:
                        input_name = f"{instance_name}.{attr_name}" if instance_name else attr_name
                        break
                except Exception:
                    continue

        if not instance_name and input_name and "." in input_name:
            instance_name = input_name.split(".", 1)[0]

        if not input_name:
            input_name = "<unnamed input>"
        if not instance_name:
            instance_name = "<unknown component>"

        return input_name, instance_name

    def add_equation(self, terms, rhs=0.0):
        """Add one equation to the linear system.

        Parameters
        ----------
        terms : dict[Component.Output, float]
            Maps Output objects to their coefficients in this equation. If an Output is
            not part of the unknown set, its current .v value is substituted and moved
            to the right-hand side automatically.
        rhs : float
            Right-hand side constant.
        """
        row = np.zeros(self._n_unknown)
        adjusted_rhs = float(rhs)
        for output, coeff in terms.items():
            if output is None:
                raise RuntimeError(
                    "Network equation contains an unconnected source term (None). "
                    "This usually means context.source(input_port) was used on an input "
                    "that is not connected. Be sure to add a connection for that input "
                    "before running the simulation."
                )
            idx = self._unknown_index.get(output)
            if idx is not None:
                row[idx] += float(coeff)
            else:
                adjusted_rhs -= float(coeff) * float(output.v)
        self._add_row_fn(row, adjusted_rhs)

    def is_unknown(self, output):
        """Return True if the given Output is an unknown in the current network solve."""
        return output in self._unknown_index

    def source(self, input_port):
        """Return the source Output connected to input_port.

        Raises RuntimeError when input_port is not connected, with details about
        the offending input and component instance to aid debugging.
        """
        if input_port.connection is not None:
            return input_port.connection.source
        input_name, instance_name = self._describe_input_port(input_port)
        raise RuntimeError(
            "Network equation requested context.source(input_port) for an unconnected input. "
            f"Input: {input_name}. Component instance: {instance_name}. "
            "Add the missing connection before running the simulation."
        )

    @property
    def unknowns(self):
        """Frozenset of all unknown Output objects in this subnetwork."""
        return frozenset(self._unknown_index)


# =========================================================================================        
class Model:
    """
    Model class to hold components and run the simulation.

    The model workflow is as follows:
    * Use a script to create an instance of the Model class, and add components as attributes of the model.
    * Use the connect() method to connect component outputs to other component inputs.
    * Create any plotters using the add_plotter() method, which takes lists of component inputs and outputs 
      to be plotted on the y1 and y2 axes, respectively.
    * Optionally, call initialize() to initialize the model before running the simulation. This will automatically call the presim_setup() method of each component, and assign names to all inputs and outputs.
    * In a loop for all time steps, call the step() method to run the simulation. This will run through all components and update inputs from connections, calculate outputs, check for convergence, and log data for plotting and historian.
    * The final step automatically calls finalize(), dispatching component postsim_calcs()
      once in execution order, whether or not the step converged. Set
      settings.auto_postsim_calcs=False to extend the run and finalize() manually later.

    Substeps in the time series workflow are as follows:

    Function        | Flags active       | Description
    ----------------|--------------------|------------------------------------------------------
    presim_setup()  |                    | Calculations before simulation begins. Data from other components is not available.
    step()          | is_first_step      | Calculations done only on the first timestep call 
    step()          | is_first_iteration | Calculations done only on the first iteration of each timestep. Components called in order. 
    step()          |                    | Subsequent iteration calculations requiring convergence.
    step()          | is_converged       | Final pass of calculations done after convergence is reached.
    postsim_calcs() |                    | Calculations done after the final time step is reached. Data from all time steps is available.
    """

    # ------------------------------------------------------------------------
    class Settings:
        timestep = 1            # [sec] Simulation time step
        start_time = 0          # [sec] Simulation start time
        stop_time = 24*3600     # [sec] Simulation stop time (not inclusive)
        auto_postsim_calcs = True  # Finalize automatically when step() reaches stop_time.
        tol_rel_global = 1.e-6  # Relative tolerance for connections. Used if not overridden by individual connections
        tol_abs_global = 1.e-4  # Absolute tolerance for connections. Used if not overridden by individual connections
        max_iterations = 100    # Maximum number of iterations for convergence, per step(). Raises warning if exceeded.
        learn_rate = None       
        # (0..1) if not None, replaces individual connection learn rates when connecting components. The learn rate 
        # determines the fraction of the difference between the new value and the previous value that is applied at 
        # each iteration, for values updated from connections. Lower learn rates can help with convergence of 
        # difficult problems, at the cost of more iterations.
        progress_update = 200  # [msec] Minimum clock time between terminal progress updates, in milliseconds
        network_solver_debug = False  # Print every coupled solve, including labeled A, b, and x.
        network_solver_condition_warning = 1.e12  # Warning threshold when diagnostics are computed.
        global_plotter_args = {}  # Defaults for new data plotters; excludes y1 and y2.

        def __init__(self):
            # Copy class defaults so models do not share mutable plotter settings.
            self.global_plotter_args = copy.deepcopy(type(self).global_plotter_args)
    # End class Settings -----------------------------------------------------
    
    # ----------------------------------------------------------------------------
    # Model class methods --------------------------------------------------------
    # ----------------------------------------------------------------------------
    def __init__(self):
        self.settings = Model.Settings()
        self.is_initialized = False           # Flag - True after initialize() is called        
        self.plotters_initialized = False     # Flag - True after plotters are initialized (at the end of initialize())
        self.is_first_step = True             # Flag - True only throughout the first step() call of the simulation
        self.is_first_iteration = True        # Flag - True only throughout the first iteration of each step() call
        self.is_converged = False             # Flag - True when the current step() call has reached convergence
        self.time = 0                         # Current simulation time (sec)
        # placeholder for plotters
        self._plotters = []
        # Internal bookkeeping for components 
        self._components = []
        # Internal flag to track whether stepping has begun
        self._has_started_stepping = False
        self._finalization_state = "pending"  # pending, running, completed, or failed
        # Internal bookkeeping for network equation building and solving
        self._output_owner_by_id = {}
        self._input_owner_by_id = {}
        self._network_analysis = None
        self._network_solver_warned = False
        self._network_solver_reported_issues = set()
        self.last_network_solver_diagnostics = None  # Latest debug/failure snapshot; arrays are copies.
        self._network_views = []
        self._last_progress_line_len = 0
        self._last_progress_step = -1
        self._last_progress_wall_clock = 0.0
        self._deferred_network_graphs = []
        self._step_count = 0
        return
    
    _PLOTTER_DEFAULT_ARGS = {
        'y1lim': None,
        'y2lim': None,
        'y1label': '',
        'y2label': '',
        'nmax_points': 1000,
        'update_every': 1,
        'tab_title': None,
        'show_live': True,
        'time_units': 'sec',
        'plotter_size': (0.9, 0.9),
        'font_size': None,
        'font_color': '#c0c0c0',
        'bg_color': None,
        'line_weights': 2,
        'line_styles': None,
        'line_colors': None,
        'grid': True,
        'grid_horizontal': True,
        'grid_vertical': True,
        'array_config': None,
    }

    def _resolve_plotter_args(self, explicit_args, *, simstats=False):
        """Resolve built-in defaults, model defaults, and explicit arguments."""
        global_args = self.settings.global_plotter_args
        if not isinstance(global_args, dict):
            raise TypeError("settings.global_plotter_args must be a dict")
        for source, args in (("settings.global_plotter_args", global_args),
                             ("plotter arguments", explicit_args)):
            for key in args:
                if key in ('y1', 'y2'):
                    raise ValueError(f"{source} cannot contain {key!r}; supply plot inputs separately")
                if key not in self._PLOTTER_DEFAULT_ARGS:
                    raise TypeError(f"Unsupported plotter argument {key!r} in {source}")

        resolved = self._PLOTTER_DEFAULT_ARGS.copy()
        if simstats:
            resolved.update(tab_title="Simulation Stats", y1label="Timestep", y2label="Iterations")
        resolved.update(global_args)
        resolved.update({key: value for key, value in explicit_args.items()
                         if value is not _NOT_SET})
        return resolved

    def add_plotter_simstats(self, tab_title=_NOT_SET,
                            nmax_points=_NOT_SET,
                            update_every=_NOT_SET,
                            show_live=_NOT_SET,
                            time_units=_NOT_SET, **kwargs):
        """
        Add a plotter to the model to visualize simulation statistics over time.

        Omitted options use settings.global_plotter_args, then built-in
        defaults. Explicit values, including None, override global settings.
        Additional keyword options accepted by add_plotter (except y1 and
        y2) can also be supplied here. Without overrides, the tab title is
        "Simulation Stats" and the y-axis labels are "Timestep" and "Iterations".
        
        Parameters
        ----------
        update_every : int, optional
            Update the plot every N time steps.
        tab_title : str, optional
            Title of the tab for the current plotter.
        show_live : bool, optional
            Flag indicating whether to show the plotter live during the simulation. If false, the plots will render after the simulation has completed.
        time_units : str, optional
            Units for the time axis: nsec, msec, sec, min, hr, day, week,
            or year. Default is 'sec'; a year is 365 days. Simulation time
            and the timestep values on the y-axis remain in seconds.
        """
        options = self._resolve_plotter_args(
            dict(kwargs, tab_title=tab_title, nmax_points=nmax_points,
                 update_every=update_every, show_live=show_live, time_units=time_units),
            simstats=True,
        )
        self._add_plotter([self.historian['timestep']], [self.historian['iterations']], options)

    def add_plotter(self,
                    y1, y2=None,
                    y1lim=_NOT_SET, y2lim=_NOT_SET,
                    y1label=_NOT_SET, y2label=_NOT_SET,
                    nmax_points=_NOT_SET, update_every=_NOT_SET,
                    tab_title=_NOT_SET, show_live=_NOT_SET,
                    time_units=_NOT_SET, plotter_size=_NOT_SET, *,
                    font_size=_NOT_SET, font_color=_NOT_SET, bg_color=_NOT_SET,
                    line_weights=_NOT_SET, line_styles=_NOT_SET, line_colors=_NOT_SET,
                    grid=_NOT_SET, grid_horizontal=_NOT_SET, grid_vertical=_NOT_SET,
                    array_config=_NOT_SET):
        """
        Add a plotter to the model to visualize component inputs and outputs over time.

        Omitted options use settings.global_plotter_args, then built-in
        defaults. Explicit values, including None, override global settings.
        Global settings apply only to newly created plotters and cannot
        supply the y1 or y2 inputs.
        
        Parameters
        ----------
        y1 : Component.Input, Component.Output, or list
            Component input or output (or list of them) to plot on the primary y-axis.
        y2 : Component.Input, Component.Output, or list, optional
            Component input or output (or list of them) to plot on the secondary y-axis.
        y1lim : tuple, optional
            Limits for the primary y-axis (min, max). Default is None (automatic).
        y2lim : tuple, optional
            Limits for the secondary y-axis (min, max). Default is None (automatic).
        y1label : str, optional
            Label for the primary y-axis. Default is ''.
        y2label : str, optional
            Label for the secondary y-axis. Default is ''.
        nmax_points : int, optional
            Maximum number of points to show in the scrolling plotter time horizon. Default is 1000.
        update_every : int, optional
            Update the plot every N time steps. Default is 1.
        tab_title : str, optional
            Title of the tab for the current plotter. Default is None (automatic).
        show_live : bool, optional
            Whether to show the plotter live (default True). If false, the plots render after the simulation has completed.
        time_units : str, optional
            Units for the time axis: nsec, msec, sec, min, hr, day, week,
            or year. Default is 'sec'; a year is 365 days. Simulation time
            remains in seconds. Units are independent for each plotter.
        plotter_size : tuple, optional
            Window size as fractions of the screen or pixel dimensions.
            Default is (0.9, 0.9).
        font_size : float, optional
            Initial plot font size in points. Font buttons adjust this size.
        font_color, bg_color : color, optional
            Plot text and background colors; accept Matplotlib colors.
        line_weights, line_styles, line_colors : scalar or list, optional
            Line widths, Matplotlib format strings (e.g. 's--'), and colors.
            Supply a flat list ordered y1 then y2, or nested [y1 values,
            y2 values] lists matching each axis's series count. Scalars or
            single-entry lists apply to every series. None restores defaults.
        grid, grid_horizontal, grid_vertical : bool, optional
            Show the grid and its horizontal/vertical lines (all default True).
            Setting grid=False hides both directions.
        array_config : dict or list of dict, optional
            Configuration for vector signals shaped (m,) or (1, m).
            A dict applies to all signals; a list matches y1 then y2.
            mode is 'fill_between' (default), 'spaghetti', or 'indices'.
            percentiles=(5, 95) specifies the shaded band. indices selects
            zero-based members (all by default for spaghetti). show_mean
            defaults to True except in indices mode. fill_alpha=0.2,
            spaghetti_alpha=0.15, spaghetti_width=1, and mean_width=None
            control appearance; None uses the signal's line weight.
            Scalar signals ignore this configuration.
        """
        options = self._resolve_plotter_args(dict(
            y1lim=y1lim, y2lim=y2lim, y1label=y1label, y2label=y2label,
            nmax_points=nmax_points, update_every=update_every,
            tab_title=tab_title, show_live=show_live, time_units=time_units,
            plotter_size=plotter_size,
            font_size=font_size, font_color=font_color, bg_color=bg_color,
            line_weights=line_weights, line_styles=line_styles, line_colors=line_colors,
            grid=grid, grid_horizontal=grid_horizontal, grid_vertical=grid_vertical,
            array_config=array_config,
        ))
        self._add_plotter(y1, y2, options)

    def _add_plotter(self, y1, y2, options):
        """Normalize inputs and construct a plotter with resolved options."""
        if not isinstance(y1, type([])):
            y1t = [y1]
        else:
            y1t = y1
        y2t = y2
        if y2 is not None:
            if not isinstance(y2, type([])):
                y2t = [y2]

        self._plotters.append(OnlinePlotter(y1=y1t, y2=y2t, **options))

    def add_network_graph(
        self,
        show_tab=True,
        save_png=False,
        save_svg=False,
        path_base=None,
        include_subnetworks=True,
        show_connection_labels=True,
        tab_title="Connections",
        show_live=True,
        layout_file=None,
    ):
        """
        Create a topology graph tab and optionally export it to image files.

        Parameters
        ----------
        show_tab : bool, optional
            Whether to show the graph in a tab. Default is True.
        save_png : bool, optional
            Whether to export the graph to a PNG file. Default is False.
        save_svg : bool, optional
            Whether to export the graph to an SVG file. Default is False.
        path_base : str, optional
            Base path (excluding extension) for exported image files. Required if save_png or save_svg is True.
        include_subnetworks : bool, optional
            Whether to draw bounding boxes around identified subnetworks. Default is True.
        show_connection_labels : bool, optional
            Whether to display port names on the connection edges. Default is True.
        tab_title : str, optional
            Title for the graph tab. Default is "Connections".
        show_live : bool, optional
            Whether to instantiate the tab immediately. Default is True. Set to False to defer until wait_for_plots().
        layout_file : str, optional
            Path to a JSON layout file produced by the Save Layout button. When provided the node
            positions stored in the file supersede the automatically generated layout.

        Returns
        -------
        dict
            A dictionary containing status information:
            - 'view_created' (bool): True if the view was successfully created.
            - 'exported_paths' (tuple): Paths to any exported image files.
            - 'n_components' (int): Number of components in the model.
            - 'n_edges' (int): Number of edges (connections) analyzed.
        """
        if not self.is_initialized:
            self.initialize()
        if self._network_analysis is None:
            self._build_network_analysis()

        if (save_png or save_svg) and not path_base:
            raise ValueError("path_base is required when save_png or save_svg is enabled.")

        if not show_live:
            self._deferred_network_graphs.append({
                "show_tab": show_tab,
                "save_png": save_png,
                "save_svg": save_svg,
                "path_base": path_base,
                "include_subnetworks": include_subnetworks,
                "show_connection_labels": show_connection_labels,
                "tab_title": tab_title,
                "layout_file": layout_file,
            })
            return {
                "view_created": False,
                "exported_paths": (),
                "n_components": len(self._components),
                "n_edges": len(self._network_analysis["edges"]) if self._network_analysis is not None else 0,
            }

        view = None
        if show_tab or save_png or save_svg:
            # Try loading a stored layout file of the same name as the model script
            try:
                if layout_file is None:
                    lf_name = f"{os.path.splitext(sys.argv[0])[0]}.json"
                    if os.path.isfile(lf_name):
                        layout_file = lf_name
            except Exception:
                pass

            view = NetworkTopologyView(
                self,
                tab_title=tab_title,
                include_subnetworks=include_subnetworks,
                show_connection_labels=show_connection_labels,
                layout_file=layout_file,
            )
            self._network_views.append(view)

        exported_paths = []
        if view is not None and path_base:
            if save_png:
                png_path = f"{path_base}.png"
                view.export_png(png_path)
                exported_paths.append(png_path)
            if save_svg:
                svg_path = f"{path_base}.svg"
                view.export_svg(svg_path)
                exported_paths.append(svg_path)

        return {
            "view_created": view is not None,
            "exported_paths": tuple(exported_paths),
            "n_components": len(self._components),
            "n_edges": len(self._network_analysis["edges"]) if self._network_analysis is not None else 0,
        }

    def wait_for_plots(self):
        """
        Call to keep the plotter window open after simulation is complete.
        """
        # Finalize deferred (show_live=False) plotters first — this creates the Qt app
        # and window if they don't exist yet.
        for plotter in self._plotters:
            if not plotter.show_live:
                plotter._finalize()

        # Instantiate any deferred network graph views.
        for kwargs in self._deferred_network_graphs:
            if kwargs.get("show_tab") or kwargs.get("save_png") or kwargs.get("save_svg"):
                view = NetworkTopologyView(
                    self,
                    tab_title=kwargs["tab_title"],
                    include_subnetworks=kwargs["include_subnetworks"],
                    show_connection_labels=kwargs["show_connection_labels"],
                    layout_file=kwargs.get("layout_file"),
                )
                self._network_views.append(view)
                if kwargs.get("path_base"):
                    if kwargs.get("save_png"):
                        view.export_png(f"{kwargs['path_base']}.png")
                    if kwargs.get("save_svg"):
                        view.export_svg(f"{kwargs['path_base']}.svg")
        self._deferred_network_graphs.clear()

        app = OnlinePlotter.app
        main_window = OnlinePlotter.main_window
        # Check whether plotting was ever initialized
        if app is None or main_window is None:
            return
        # bring the plot window to the front if it is minimized or behind other windows
        if main_window.isMinimized():
            main_window.showNormal()
        # raise and activate the window to bring it to the front
        main_window.raise_()
        main_window.activateWindow()

        # start the Qt event loop to display the plot window and allow interaction
        app.exec()

    def connect(self, source, destination, tol_rel = None, tol_abs=None, log_n_iter = 0, learn_rate = None, solve_group=None):
        """
        Connect an output from one component to an input of another component.
        
        Parameters
        ----------
        source : Component.Output
            The source output port providing the value.
        destination : Component.Input
            The destination input port receiving the value.
        tol_rel : float, optional
            Relative tolerance for convergence. Defaults to model settings if not given.
        tol_abs : float, optional
            Absolute tolerance for convergence. Defaults to model settings if not given.
        log_n_iter : int, optional
            Number of iterations to keep history for logging.
        learn_rate : float, optional
            Learn rate / relaxation factor for convergence iterating (0..1].
        solve_group : str, optional
            Group identifier for coupled network matrix solving. This is a unique string that 
            is shared by all connected equations within a particular coupled networks.
        """
        if self._has_started_stepping:
            raise RuntimeError("Cannot add new connections after step() has started.")
        
        # In order for connections to be established correctly, the model 
        # must be initialized first to assign names to all inputs and outputs. 
        # If the model is not initialized, initialize it now.
        if not self.is_initialized:
            self.initialize()

        if not isinstance(source, Component.Output):
            raise RuntimeError(f"Source connection object must be of type 'Component.Output'")
        if not isinstance(destination, Component.Input):
            raise RuntimeError(f"Destination connection object must be of type 'Component.Input'")
        
        # override learn_rate with the model default if that's provided and learn_rate is default
        if learn_rate == None and self.settings.learn_rate is not None:
            learn_rate_temp = self.settings.learn_rate
        else:
            learn_rate_temp = 1.0 if learn_rate is None else learn_rate
        # Override tol_rel and tol_abs with model defaults if not provided
        tol_rel_temp = self.settings.tol_rel_global if tol_rel is None else tol_rel
        if tol_rel_temp is None:
            tol_rel_temp = 1.e-6
        tol_abs_temp = self.settings.tol_abs_global if tol_abs is None else tol_abs
        if tol_abs_temp is None:
            tol_abs_temp = 1.e-6

        destination.connection = Connection(source, tol_rel_temp, tol_abs_temp, log_n_iter, learn_rate_temp, solve_group=solve_group)
        destination.is_connected = True
        source.is_connected = True
        return

    def _report_network_solver(self, A, b, unknown_outputs, row_labels, status, x=None, rank=None):
        """Print and retain diagnostics without letting an SVD failure hide the original issue.

        Enable settings.network_solver_debug for every solve. Otherwise reports are
        emitted once per subnetwork/status for skipped, failed, or rank-deficient solves.
        The snapshot's residual is A @ x - b before learning-rate relaxation.
        """
        columns = tuple(output.name or f"output[{i}]" for i, output in enumerate(unknown_outputs))
        key = (columns, status)
        if not self.settings.network_solver_debug and key in self._network_solver_reported_issues:
            return
        singular_values = None
        condition = None
        diagnostic_error = None
        if np.all(np.isfinite(A)):
            try:
                singular_values = np.linalg.svd(A, compute_uv=False)
                if rank is None:
                    tolerance = singular_values[0] * max(A.shape) * np.finfo(A.dtype).eps
                    rank = int(np.count_nonzero(singular_values > tolerance))
                condition = (float(singular_values[0] / singular_values[-1])
                             if singular_values[-1] > 0 else float('inf'))
            except np.linalg.LinAlgError as exc:
                diagnostic_error = str(exc)
        residual = None if x is None else A @ x - b
        report = {
            "time": self.time, "iteration": getattr(self, "iteration", -1),
            "status": status, "A": A.copy(), "b": b.copy(),
            "row_labels": tuple(row_labels), "column_labels": columns,
            "rank": rank, "condition_number": condition,
            "singular_values": singular_values,
            "x": None if x is None else x.copy(), "residual": residual,
            "diagnostic_error": diagnostic_error,
        }
        self.last_network_solver_diagnostics = report
        self._network_solver_reported_issues.add(key)
        print(f"\nNetwork solver | time={self.time:g}, iteration={report['iteration']} | {status}")
        if np.any(np.isnan(A)) or np.any(np.isnan(b)) or (x is not None and np.any(np.isnan(x))):
            print("  ERROR: 'nan' values detected in the matrix solve. "
                  "Make sure output and input values used in the matrix coefficients "
                  "are given finite initial guess values before solving.")
        print(f"  A: {A.shape[0]} rows x {A.shape[1]} columns; rank={rank}; condition number (2-norm)={condition}")
        if condition is not None and condition >= self.settings.network_solver_condition_warning:
            print("  WARNING: ill-conditioned matrix; small coefficient changes may strongly affect the solution.")
        if rank is not None and rank < A.shape[1]:
            print("  WARNING: equations do not uniquely determine all output columns.")
        if diagnostic_error:
            print(f"  SVD diagnostics failed: {diagnostic_error}")
        for j, name in enumerate(columns):
            print(f"  column {j}: {name}")
        with np.printoptions(precision=6, threshold=np.inf, linewidth=160):
            for i, label in enumerate(row_labels):
                print(f"  row {i} ({label}): {A[i]} | b={b[i]:.6g}")
            print(f"  zero rows: {np.flatnonzero(np.all(A == 0, axis=1))}")
            print(f"  zero columns: {np.flatnonzero(np.all(A == 0, axis=0))}")
            if singular_values is not None:
                print(f"  singular values: {singular_values}")
            if x is not None:
                print(f"  x (before relaxation): {x}")
                print(f"  residual A @ x - b: {residual}; norm={np.linalg.norm(residual):.6g}")
        for label, values in (("A", A), ("b", b)):
            if not np.all(np.isfinite(values)):
                print(f"  non-finite {label} entries: {np.argwhere(~np.isfinite(values)).tolist()}")

    def _solve_coupled_subnetwork(self, plan):
        """Solve one coupled subnetwork if equation builders and semantic roles provide enough equations."""
        
        # Collect the subnetwork edges
        plan_components = set(plan["components"])
        subnetwork_edges = []
        for source_component, destination_component, source_output, destination_input in self._network_analysis["edges"]:
            if source_component in plan_components and destination_component in plan_components:
                subnetwork_edges.append((source_output, destination_input))
        
        # Collect the unknown outputs
        unknown_outputs = []
        for source_output, destination_input in subnetwork_edges:
            if destination_input.connection.solve_group is None:
                continue
            if source_output not in unknown_outputs:
                unknown_outputs.append(source_output)
        unknown_outputs = tuple(unknown_outputs)  #make immutable
        n_unknown = len(unknown_outputs)
        if n_unknown == 0:
            return False, False

        unknown_index = {output: idx for idx, output in enumerate(unknown_outputs)}
        A_rows = []
        b_rows = []
        row_labels = []

        def _add_row(row, rhs):
            A_rows.append(row)
            b_rows.append(float(rhs))
            row_labels.append(component.name or type(component).__name__)

        eq_context = NetworkEquationContext(
            unknown_index,
            n_unknown,
            _add_row,
            input_owner_by_id=self._input_owner_by_id,
        )
        for component in plan["components"]:
            component.coupled_eqs = eq_context
            try:
                component.calculate()
            finally:
                component.coupled_eqs = None

        if not A_rows:
            return False, False

        A = np.vstack(A_rows)
        b = np.asarray(b_rows)
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            self._report_network_solver(A, b, unknown_outputs, row_labels, "skipped: non-finite coefficients or RHS")
            return False, True
        if A.shape[0] < n_unknown:
            self._report_network_solver(A, b, unknown_outputs, row_labels, "skipped: fewer equations than unknowns")
            return False, True

        rank = None
        try:
            if A.shape[0] == n_unknown:
                rank = int(np.linalg.matrix_rank(A))
            if A.shape[0] == n_unknown and rank == n_unknown:
                x_new = np.linalg.solve(A, b)
                method = "solve"
            else:
                x_new, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
                method = "least squares"
        except np.linalg.LinAlgError as exc:
            self._report_network_solver(A, b, unknown_outputs, row_labels, f"failed: {exc}", rank=rank)
            return False, True
        if not np.all(np.isfinite(x_new)):
            self._report_network_solver(A, b, unknown_outputs, row_labels, "failed: non-finite solution", x=x_new, rank=rank)
            return False, True
        if self.settings.network_solver_debug or rank < n_unknown:
            status = method if rank == n_unknown else f"{method}: rank deficient"
            self._report_network_solver(A, b, unknown_outputs, row_labels, status, x=x_new, rank=rank)

        updated_any = False
        # For the matrix solve, we need to use the global learning rate since individual connection
        # learn rates can't be enforced. If the global isn't specified, just default to the full step.
        relax = self.settings.learn_rate if self.settings.learn_rate is not None else 1.0
        for output, idx in unknown_index.items():
            old_value = output.v
            if np.any(np.isnan(old_value)):
                new_value = x_new[idx]
            else:
                new_value = old_value + (x_new[idx] - old_value) * relax
            output.v = new_value
            if np.any(np.isnan(old_value)) or np.abs(new_value - old_value) > self.settings.tol_abs_global:
                updated_any = True

        return updated_any, True

    def _find_back_edges(self, nodes, adjacency):
        """
        >> Internal method - shouldn't be called by the user directly. <<

        Find DFS back-edges in a directed graph induced by ``nodes``.

        A back-edge is an edge from a node to another node currently on the
        active DFS stack. Presence of at least one back-edge indicates a
        directed cycle.
        """
        node_set = set(nodes)
        if not node_set:
            return set()

        # 0 = unvisited, 1 = active DFS stack, 2 = finished
        state = {node: 0 for node in node_set}
        back_edges = set()

        for start in nodes:
            if start not in node_set or state[start] != 0:
                continue

            state[start] = 1
            stack = [(start, iter(adjacency[start]))]

            while stack:
                node, children = stack[-1]
                try:
                    child = next(children)
                except StopIteration:
                    state[node] = 2
                    stack.pop()
                    continue

                if child not in node_set:
                    continue

                child_state = state[child]
                if child_state == 1:
                    back_edges.add((node, child))
                elif child_state == 0:
                    state[child] = 1
                    stack.append((child, iter(adjacency[child])))

        return back_edges

    def _classify_subnetwork(self, subnetwork, adjacency, reverse_adjacency, subnetwork_back_edges=None):
        """
        >> Internal method - shouldn't be called by the user directly. <<

        Classify a connected subnetwork as sequential or coupled.

        Coupled : A subnetwork is classified as coupled if it contains any of the following:
            * A directed cycle (feedback loop)
            * A branching node (one output feeding multiple inputs)
            * A merging node (multiple outputs feeding one input)
        Sequential : A subnetwork is classified as sequential if it contains none of the above.
        """
        in_degree = {}
        out_degree = {}

        for component in subnetwork:
            out_degree[component] = sum(1 for neighbor in adjacency[component] if neighbor in subnetwork)
            in_degree[component] = sum(1 for neighbor in reverse_adjacency[component] if neighbor in subnetwork)

        has_branching = any(degree > 1 for degree in out_degree.values())
        has_merging = any(degree > 1 for degree in in_degree.values())

        # Build or consume cached cycle information from DFS back-edge detection.
        if subnetwork_back_edges is None:
            ordered_nodes = tuple(component for component in self._components if component in subnetwork)
            subnetwork_back_edges = self._find_back_edges(ordered_nodes, adjacency)
        has_cycle = len(subnetwork_back_edges) > 0

        if has_cycle or has_branching or has_merging:
            mode = "coupled"
        else:
            mode = "sequential"

        return {
            "mode": mode,
            "components": tuple(subnetwork),
            "in_degree": in_degree,
            "out_degree": out_degree,
            "has_cycle": has_cycle,
            "has_branching": has_branching,
            "has_merging": has_merging,
            "back_edges": tuple(subnetwork_back_edges),
        }

    def _build_network_analysis(self):
        """
        >> Internal method - shouldn't be called by the user directly. <<

        Analyze the current model topology and develop a plan for solving each subnetwork.
        """
        if not self.is_initialized:
            return []

        # Build lookup tables that map Input/Output objects back to their owning component.
        self._output_owner_by_id = {}
        self._input_owner_by_id = {}

        for component in self._components:
            for output in component.get_outputs():
                self._output_owner_by_id[id(output)] = component
            for input_item in component.get_inputs():
                self._input_owner_by_id[id(input_item)] = component

        # Build the component connection graph and identify connected subnetworks.
        edges = []
        adjacency = {component: set() for component in self._components}
        reverse_adjacency = {component: set() for component in self._components}

        for dst_component in self._components:
            for input_item in dst_component.get_inputs(connected_only=True):
                connection = input_item.connection
                src_component = self._output_owner_by_id.get(id(connection.source))
                if src_component is None:
                    continue

                adjacency[src_component].add(dst_component)
                reverse_adjacency[dst_component].add(src_component)
                edges.append((src_component, dst_component, connection.source, input_item))

        # Group components into weakly connected subnetworks.
        unvisited = set(self._components)
        subnetworks = []

        # loop until all components are visited
        while unvisited:  
            start_component = unvisited.pop()
            queue = deque([start_component])
            subnetwork = {start_component}

            while queue:
                component = queue.popleft()
                neighbors = adjacency[component] | reverse_adjacency[component]
                for neighbor in neighbors:
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        subnetwork.add(neighbor)
                        queue.append(neighbor)

            subnetworks.append(subnetwork)

        # Classify each subnetwork as sequential or coupled, and build a plan for solving it.
        plans = []
        back_edges = set()
        component_index = {component: idx for idx, component in enumerate(self._components)}

        for subnetwork in subnetworks:
            ordered_subnetwork = tuple(sorted(subnetwork, key=component_index.get))
            subnetwork_back_edges = self._find_back_edges(ordered_subnetwork, adjacency)
            back_edges.update(subnetwork_back_edges)
            plans.append(self._classify_subnetwork(
                subnetwork,
                adjacency,
                reverse_adjacency,
                subnetwork_back_edges=subnetwork_back_edges,
            ))

        self._network_analysis = {
            "adjacency": adjacency,
            "reverse_adjacency": reverse_adjacency,
            "edges": edges,
            "subnetworks": subnetworks,
            "plans": plans,
            "back_edges": back_edges,
        }

        return plans

    def _compute_execution_order(self):
        """
        >> Internal method - shouldn't be called by the user directly. <<
        
        Reorder self._components in topological order of the connection graph.

        For subnetworks without cyclic connections this produces ordering that
        enables single-pass convergence.  For coupled subnetworks, back-edges
        that close cycles are identified and removed from the graph first; the
        remaining directed acyclic graph (DAG) is then sorted topologically using
        Kahn's topological sorting algorithm, minimizing the number of unknown 
        inputs at the start of each iteration.
        """
        if self._network_analysis is None:
            return

        adjacency = self._network_analysis["adjacency"]

        # Reuse back-edges from network analysis when available; this avoids
        # running a second DFS traversal over the same graph.
        cached_back_edges = self._network_analysis.get("back_edges", None)
        back_edges = set(cached_back_edges) if cached_back_edges is not None else self._find_back_edges(self._components, adjacency)

        # Build reduced adjacency and in-degree map, omitting back-edges.
        reduced_in_degree = {c: 0 for c in self._components}
        reduced_adjacency = {c: [] for c in self._components}
        for src in self._components:
            for dst in adjacency[src]:
                if (src, dst) not in back_edges:
                    reduced_adjacency[src].append(dst)
                    reduced_in_degree[dst] += 1

        # Kahn's topological sort on the reduced DAG.
        queue = deque(c for c in self._components if reduced_in_degree[c] == 0)
        ordered = []
        while queue:
            component = queue.popleft()
            ordered.append(component)
            for neighbor in reduced_adjacency[component]:
                reduced_in_degree[neighbor] -= 1
                if reduced_in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Append any node not yet emitted as a safety guard (remaining nodes should not occur).
        ordered_set = set(ordered)
        ordered.extend(c for c in self._components if c not in ordered_set)

        # store the order
        self._components = ordered  
        # Print out the execution order
        compstr = " → ".join([component.name for component in self._components])
        print("⏣ Component call order | " + compstr)
            
    def initialize(self):
        """
        Initialize the model before running the simulation. This will automatically call the
        presim_setup() method of each component, and assign names to all inputs and outputs.

        The components are given access to the model under the attributed <component>.model, 
        which can be used to access settings and model flags.
        """
        if self.is_initialized:
            print('Model is already initialized. No further action taken. Calling function is:' + self.__class__.__name__   )
            return

        # Set the initial simulation time and iteration count
        self.time = self.settings.start_time
        self.iteration = -1
        self.is_first_step = True
        

        # Initialize the list of outputs. This is extended based on the component settings
        output_names = ['time','timestep','iterations']

        # Do an initial pass to assign names to all unnamed components based on class name
        current_component_names = []
        for item in dir(self):
            itemobj = getattr(self, item)
            if isinstance(itemobj, Component):
                if itemobj.name == "":
                    cname = type(itemobj).__name__
                    itemobj.name = cname
                else:
                    cname = itemobj.name
                current_component_names.append(cname)

        # Check for duplicate component names and create a list of names that need to be modified
        duplicate_names = set([name for name in current_component_names if current_component_names.count(name) > 1])

        # ------------------------------------------------------------
        # Loop through all attributes of the model to find components, assign names, and call presim_setup
        for item in dir(self):

            itemobj = getattr(self, item)

            # If the object is a component...            
            if isinstance(itemobj, Component):
                # Give all of the component instances access to the model
                itemobj.model = self

                # Ensure name uniqueness for the component instance. If a name is already taken, append a number to make it unique.
                instance_name = itemobj.name
                if instance_name in duplicate_names:
                    # auto assign name
                    i = 1
                    while instance_name in current_component_names:
                        instance_name = f"{itemobj.name}_{i}"
                        i += 1
                    setattr(itemobj, "name", instance_name)
                    current_component_names.append(instance_name)


                # Handle component presim_setup here
                itemobj.presim_setup()
                # Automatically assign names to all inputs and outputs of this component 
                cnames = itemobj.auto_assign_names()
                # Add this component to the model's list of components
                self._components.append(itemobj)
                # Record all output data names for the historian
                output_names += cnames

        # Append-only historian storage keeps timestep changes cheap.
        self.historian = dict([[n, []] for n in output_names])
        self._step_count = 0

        # Mark the model as initialized
        self.is_initialized = True
        return 

    # ----------------------------------------------------------------------- 
    def finalize(self):
        """Call component postsim_calcs() once, in component execution order.

        May be called explicitly after an early stop or with auto_postsim_calcs
        disabled. At least one completed timestep is required. Successful repeated
        calls do nothing. Once finalization starts, further stepping is forbidden.
        Hook exceptions propagate; failed finalization cannot be retried because
        earlier hooks may already have produced side effects.
        """
        if self._finalization_state == "completed":
            return
        if self._finalization_state != "pending":
            raise RuntimeError(
                f"Cannot finalize: finalization is {self._finalization_state}."
            )
        if self._step_count == 0:
            raise RuntimeError("Cannot finalize before a timestep has completed.")

        self._finalization_state = "running"
        try:
            for component in self._components:
                component.postsim_calcs()
        except BaseException:
            self._finalization_state = "failed"
            raise
        self._finalization_state = "completed"

    # -----------------------------------------------------------------------
    def step(self, n_steps = 1):
        """
        Parameters
        ----------
        n_steps : int, optional
            Expected number of steps (Note: Internal logic currently executes a single step per call).
        """
        if self._finalization_state != "pending":
            raise RuntimeError(
                "Cannot step after finalization has started. Create a new model for a new run."
            )
        # Check overall initialization for the model
        if not self.is_initialized:
            self.initialize()

        timestep = self.settings.timestep
        if timestep <= 0:
            raise ValueError("Invalid time settings. Ensure that timestep is positive.")

        if not self._has_started_stepping:
            # Track the clock time at the start of the first step for performance measurement
            self._start_time_wall_clock = time.time()

            assert self.settings.stop_time > self.settings.start_time, \
                "Invalid time settings. Ensure that stop_time is greater than start_time."
            assert self.settings.stop_time >= timestep, \
                "Invalid time settings. Ensure that stop_time is greater than the timestep."

            self._build_network_analysis()
            self._compute_execution_order()
            # connect() forbids new connections from this point on, so freeze I/O caches for the hot loop.
            for component in self._components:
                component._lock_io_cache()
            self._has_started_stepping = True

        # Pre-allocate plotter arrays on the first step, after all plotters have been added
        if not self.plotters_initialized:
            for plotter in self._plotters:
                plotter.preallocate(timestep=timestep)
            self.plotters_initialized = True

        self.iteration = -1  # reset the current iteration
        self.is_first_iteration = True
        self.is_converged = False

        for component in self._components:
            for input in component.get_inputs(connected_only=True):
                input.connection.clear_iteration_log()

        err_rel_history = []
        err_abs_history = []
        # main loop 
        for i in range(self.settings.max_iterations):
            all_converged = True 
            self.iteration += 1

            # -----------------------------------------------------------------
            # -----------------------------------------------------------------
            # Run the network solver here to update all coupled components at the start of each
            # iteration. This allows guess propagation to occur across the entire coupled
            # network, which can improve convergence in cases where simple sequential
            # updates are not sufficient. 
            # 
            # Components contribute equations by checking
            # ``if self.coupled_eqs is not None:`` inside calculate(); the Model sets
            # self.coupled_eqs to the NetworkEquationContext before the call and clears it
            # afterward. 
            # 
            # Coefficients in the invertible matrix are based on values at the start of 
            # the iteration, although components can compute fresher coefficients from
            # their inputs before the context check for a Gauss-Seidel effect.
            # 
            # Coupled subnetworks are solved for components that are part of a solve_group.
            # -----------------------------------------------------------------
            network_updated = False
            # Only run if there are coupled networks and the user has provided equations to solve them.
            if self._network_analysis is not None:
                equations_added_any = False
                coupled_count = 0
                # Loop through the network 'plans' and solve each coupled subnetwork independently.
                for plan in self._network_analysis["plans"]:
                    if plan["mode"] != "coupled":
                        continue
                    coupled_count += 1
                    plan_updated, plan_has_equations = self._solve_coupled_subnetwork(plan)
                    network_updated = plan_updated or network_updated
                    equations_added_any = plan_has_equations or equations_added_any

                if coupled_count > 0 and not equations_added_any and not self._network_solver_warned:
                    print("Network solver: coupled networks detected but no equations were added. "
                        "Mark connections with solve_group and contribute equations inside calculate() "
                        "using 'if self.coupled_eqs is not None:' to enable solving.")
                    self._network_solver_warned = True

            if network_updated:
                all_converged = False
            # -----------------------------------------------------------------
            # -----------------------------------------------------------------


            # -----------------------------------------------------------------
            # Component calculation and connection updates. This is the main loop that 
            # iterates through all components, updating their inputs from connections 
            # and calling their calculate() methods.
            # -----------------------------------------------------------------
            max_abs_err = 0.
            max_rel_err = 0.
            for component in self._components:
                # Update connections first
                for input in component.get_inputs(connected_only=True):
                    all_converged = all_converged & input.update_from_connection()
                    max_abs_err = max(max_abs_err, np.max(input.connection.err_abs))
                    max_rel_err = max(max_rel_err, np.max(input.connection.err_rel))

                # Calculate
                component.calculate()
            
            if all_converged and not self.is_first_iteration:
                break
            # Handle the case where the maximum number of iterations is reached without convergence.
            if i == self.settings.max_iterations-1:
                for component in self._components:
                    for input in component.get_inputs(connected_only=True):
                        if not input.connection.is_converged:
                            ps = f'{self.time:<10.5f} | {input.connection.source.name} not converged after {input.connection.n_iter} iterations.'
                            if input.connection.log_n_iter > 0:
                                ps += ' Iter log: ' + ' '.join(str(value) for value in input.connection.iter_log[:,0])
                            print(ps)

            err_rel_history.append(max_rel_err)
            err_abs_history.append(max_abs_err)
            self.is_first_iteration = False

        # Count non-converged connections to derive a fractional indicator for the plotter.
        # Must be done before reset_for_step() clears connection.is_converged.
        n_total = 0
        n_not_converged = 0
        for _comp in self._components:
            for _inp in _comp.get_inputs(connected_only=True):
                n_total += 1
                if not _inp.connection.is_converged:
                    n_not_converged += 1
        conv_fraction = n_not_converged / n_total if n_total > 0 else 0.0

        # Convergence complete for this step, now do post-convergence calculations and logging
        self.is_converged = True
        current_step = self._step_count
        self.historian['time'].append(self.time)
        self.historian['timestep'].append(timestep)
        self.historian['iterations'].append(self.iteration)
        for component in self._components:
            component.calculate()
            for input in component.get_inputs(connected_only=True):
                input.connection.reset_for_step()

            for output in component.get_outputs():
                value = output.v
                self.historian[output.name].append(value.copy() if isinstance(value, np.ndarray) else value)

        self._step_count += 1
        
        # Plotters
        iter_fraction = (self.iteration + 1) / self.settings.max_iterations
        for plotter in self._plotters:
            plotter.log_step(self.time, conv_fraction, iter_fraction, timestep=timestep)

        # Keep Qt tabs responsive without letting GUI activity dominate step time.
        OnlinePlotter.process_ui_events()

        self.time += timestep
        self.is_first_step = False

        # Terminal update
        elapsed = self.time - self.settings.start_time
        duration = self.settings.stop_time - self.settings.start_time
        percent = np.clip((elapsed / duration) * 100 if duration > 0 else 100.0, 0.0, 100.0)
        progress_step = self._step_count
        now_clock = time.time()
        should_print = (
            progress_step != self._last_progress_step and
            ((now_clock - self._last_progress_wall_clock) >= self.settings.progress_update / 1000.0 or percent >= 100.0)
        )
        if should_print:
            bar_width = 30
            n_fill = min(bar_width, max(0, int((percent / 100.0) * bar_width)))
            bar = '█' * n_fill + '-' * (bar_width - n_fill)
            line = f'[{bar}] {percent:5.1f}% | Sim {self.time:.1f}s | Clock {now_clock - self._start_time_wall_clock:.3f}s'
            pad = max(0, self._last_progress_line_len - len(line))
            sys.stdout.write('\r' + line + (' ' * pad))
            sys.stdout.flush()
            self._last_progress_line_len = len(line)
            self._last_progress_step = progress_step
            self._last_progress_wall_clock = now_clock

        if self.settings.auto_postsim_calcs and self.time >= self.settings.stop_time:
            self.finalize()

        return 

