from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esclab.simulate import Component, Model

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "../../../data/residence_hourly.csv"
LOAD_REFERENCE_KW = 1.0
# We choose a denominator offset of 50 W/m² to limit the influence of fluctuations
# during low-irradiance periods, when their contribution to PV energy production 
# is small. This is an engineering modeling choice that balances retaining solar 
# variability against amplifying fluctuations of limited practical significance.
SOLAR_DENOMINATOR_W_M2 = 50.0
# Limit the forecast solar irradiance to no more than this value times the clear-sky irradiance
SOLAR_ENVELOPE_FACTOR = 1.5
SCENARIO_COUNT = 100
FORECAST_HOURS = 24*3
RANDOM_SEED = 42
INNOVATION_MULTIPLIER = 1.0         # Scaling factor for the sampled innovations. 1=default
USE_CORRELATED_INNOVATIONS = True   # False removes only same-hour covariance.
ORIGIN_INDEX = 5                    # Index of the origin to use for the forecast. Choose 0-12


def prepare_forecaster(data_file=DATA_FILE):
    """Fit once and return the data, calendar baselines, and VAR parameters."""
    # Read in the raw data
    data = pd.read_csv(data_file, parse_dates=["timestamp_local_start"])
    # Mark which rows are used for training: 3 of every 4 weeks are training, 1 is held out.
    train_mask = (np.arange(len(data)) // 168) % 4 != 3
    
    times = pd.DatetimeIndex(data["timestamp_local_start"])
    
    # Transform physical GHI and load to signals with more uniform variation.
    # Normalize solar by clear sky, and offset to avoid divide by zero
    solar = data.ghi_w_m2.to_numpy() / ( data.ghi_clear_w_m2.to_numpy() + SOLAR_DENOMINATOR_W_M2)
    # Normalize load by a reference value, and log-transform to reduce skew
    demand = np.log1p(data.load_non_hvac_kw.to_numpy() / LOAD_REFERENCE_KW)
    # Use temperature as a raw signal
    temp = data.temperature_c.to_numpy()
    signals = np.column_stack([solar, demand, temp])

    # -----------------------------------------------------------
    # Compute and remove useful baselines
    # -----------------------------------------------------------
    # Remove the predictable calendar means.
    # For load, solve design @ coefficients ~= log(1 + load / reference).
    # The solar mean is a separate average for each (month, hour) combination.
    # The temperature mean is a separate average for each (month, hour) combination.
    
    # 48 separate weekday/weekend hourly levels + annual sine/cosine.
    # The annual terms describe a smooth repeating seasonal mean, not weather.
    # No intercept is added: the 48 indicators already include an intercept.
    is_weekend = times.dayofweek.to_numpy() >= 5
    group = times.hour.to_numpy() + 24 * is_weekend
    indicators = np.zeros((len(times), 48))
    indicators[np.arange(len(times)), group] = 1.0
    phase = 2 * np.pi * (times.dayofyear.to_numpy() - 1) / 365.0
    # Indicators: 48 columns, 0-23 for weekday hours, 24-47 for weekend hours. 
    # Annual sine/cosine: 2 columns, sin(2π * day_of_year / 365), cos(2π * day_of_year / 365)
    design = np.column_stack([indicators, np.sin(phase), np.cos(phase)])
    
    # Fit to compute the baseline for load using only training rows
    load_coefficients = np.linalg.lstsq( design[train_mask], signals[train_mask, 1], rcond=None )[0]
    
    # Construct solar and temperature baselines by just looking at monthly averages by hour of the day
    solar_table = np.zeros((12, 24))
    temp_table = np.zeros((12, 24))
    
    for month in range(1, 13):
        for hour in range(24):
            rows = train_mask & (times.month == month) & (times.hour == hour)
            solar_table[month - 1, hour] = signals[rows, 0].mean()
            temp_table[month - 1, hour] = signals[rows, 2].mean()
    
    # The baseline mapped back to a year of data
    baseline = np.column_stack([
        solar_table[times.month - 1, times.hour],    # lookup average
        design @ load_coefficients,                  # fitted baseline
        temp_table[times.month - 1, times.hour]      # lookup average
    ])

    # Compute the difference between the observed signals and the baseline to get departures
    departures = signals - baseline

    # The fitted baselines already give zero-mean training departures:
    # month/hour averages for solar and temperature, and load least squares
    # with constant levels. Only scale the departures; no extra centering.
    # Columns are [solar, load, temperature], all in standard-deviation units.
    scale = departures[train_mask].std(axis=0, ddof=1)
    xi = departures / scale

    # get a list of all candidate forecast origins
    candidate_origins = np.flatnonzero(train_mask[:-1] & ~train_mask[1:])
    # choose an origin for the forecast (index 0--12)
    origin = candidate_origins[ORIGIN_INDEX]

    rows = np.arange(1, len(data))
    # train only on rows where both the current and previous hour are in the training set.
    rows = rows[train_mask[rows] & train_mask[rows - 1]]
    # Use previous-hour departures as predictors for the current hour's departures. 
    predictors = np.column_stack([
        np.ones(len(rows)), xi[rows - 1, 0], xi[rows - 1, 1], xi[rows - 1, 2]
    ])
    # The target is the current hour's departures.
    targets = xi[rows]

    # Find THETA0 and THETA1 coefficients 
    coefficients = np.linalg.lstsq(predictors, targets, rcond=None)[0]  
    # Arrange the fitted coefficients in the lecture equation convention.
    # xi[n] = Theta0 + Theta1 @ xi[n-1] + nu[n] (column-vector notation).
    Theta0 = coefficients[0]
    Theta1 = coefficients[1:].T
    # The fitted values are the predicted departures for each of the training rows.
    fitted = predictors @ coefficients

    innovations = targets - fitted  

    # estimate innovation covariance, with regression degrees of freedom.
    dof = len(rows) - predictors.shape[1]
    C_nu = innovations.T @ innovations / dof

    return dict(data=data, baseline=baseline, scale=scale, xi=xi, origin=origin,
                solar_table=solar_table, temp_table=temp_table,
                load_coefficients=load_coefficients,
                Theta0=Theta0, Theta1=Theta1, C_nu=C_nu)


def generate_paths(forecaster, baseline, clear, scenario_count, random_seed,
                   initial_departures, use_correlated_innovations=True,
                   innovation_multiplier=INNOVATION_MULTIPLIER):
    """Advance all realizations hourly; return [hour, realization, signal].

    Signal order is [GHI W/m2, non-HVAC demand kW, ambient temperature degC].
    Accepted bounded departures feed the next hour, with no observation resets.
    The same seed uses the same base normal draws in either covariance case.
    """
    covariance = forecaster["C_nu"]
    if not use_correlated_innovations:
        covariance = np.diag(np.diag(covariance))
    L = np.linalg.cholesky(covariance)
    rng = np.random.default_rng(random_seed)
    previous = np.broadcast_to(initial_departures, (scenario_count, 3)).copy()
    paths = np.empty((len(baseline), scenario_count, 3))
    for n in range(len(baseline)):
        nu = innovation_multiplier * (rng.standard_normal((scenario_count, 3)) @ L.T)
        trial = forecaster["Theta0"] + previous @ forecaster["Theta1"].T + nu
        ghi, load, temp, previous = reconstruct(
            trial, baseline[n], clear[n], forecaster["scale"])
        paths[n] = np.column_stack((ghi, load, temp))
    return paths


def generate_annual_paths(forecaster, scenario_count=SCENARIO_COUNT,
                          random_seed=RANDOM_SEED,
                          use_correlated_innovations=USE_CORRELATED_INNOVATIONS,
                          innovation_multiplier=INNOVATION_MULTIPLIER):
    """Generate exactly the supplied calendar year, starting January 1 at 00:00.

    Initial standardized departures are zero in the hour preceding January 1;
    the first innovation supplies the first hour of the year. Later observations
    never reset the forecast. Use the fitted calendar baselines and clear-sky
    profile for this complete non-leap year in local standard time, without DST.
    """
    data = forecaster["data"]
    times = pd.DatetimeIndex(data.timestamp_local_start)
    clear = data.ghi_clear_w_m2.to_numpy()
    paths = generate_paths(forecaster, forecaster["baseline"], clear, scenario_count,
                           random_seed, np.zeros(3), use_correlated_innovations,
                           innovation_multiplier)
    return times, paths


def reconstruct(standardized, baseline, clear, scale):
    """
    Convert forecast departures to irradiance, electrical demand, and temperature.

    The last axis holds [solar, load, temperature] departures.
    
    Solar is limited to [0, 1.5 * clear-sky irradiance]
    Demand cannot be negative.

    Returns irradiance [W/m²], demand [kW], temperature [°C], and adjusted departures
    to use as the starting values for the next forecast step.
    """

    # Undo departure scaling and restore each signal's calendar baseline.
    transformed = baseline + standardized * scale
    bounded = transformed.copy()

    # Bound the solar index so reconstructed GHI lies in [0, 1.5 * clear].
    # clear=0 gives exactly zero nighttime GHI while retaining hourly spacing.
    upper_solar_index = SOLAR_ENVELOPE_FACTOR * clear / (clear + SOLAR_DENOMINATOR_W_M2)
    bounded[..., 0] = np.clip(transformed[..., 0], 0, upper_solar_index)
    bounded[..., 1] = np.maximum(transformed[..., 1], 0)

    # Undo the physical transforms; expm1(a) means exp(a) - 1.
    ghi = bounded[..., 0] * (clear + SOLAR_DENOMINATOR_W_M2)
    load = LOAD_REFERENCE_KW * np.expm1(bounded[..., 1])
    temperatures = bounded[..., 2]
    accepted_xi = (bounded - baseline) / scale
    return ghi, load, temperatures, accepted_xi


def plot_preprocessing(frame, xi, baseline, scale, origin):
    """
    Plot the preprocessed departures and the calendar baseline the week specified 
    by the origin (zero-based index hour of the year).

    frame : pandas.DataFrame
        The original data frame containing the physical signals.
    xi : numpy.ndarray
        The standardized departures in [solar, load, temperature] order.
    baseline : numpy.ndarray
        The calendar baselines in solar-index, log-load, and °C units.
    scale : numpy.ndarray
        The training standard deviations of the departures from those baselines.
    origin : int
        The zero-based index of the last hour of the week to plot. 
    """
    clear = frame.ghi_clear_w_m2.to_numpy()
    baseline_ghi, baseline_load, _, _ = reconstruct(
        np.zeros_like(xi), baseline, clear, scale)

    # Last training week: physical signals above, standardized departures below.
    week = slice(origin - 167, origin + 1)
    times = frame.timestamp_local_start.iloc[week]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for column, measured, calendar, label in [
        (0, frame.ghi_w_m2.to_numpy(), baseline_ghi, "GHI [W/m²]"),
        (1, frame.load_non_hvac_kw.to_numpy(), baseline_load, "Demand [kW]"),
    ]:
        axes[0, column].plot(times, measured[week], label="Historical")
        axes[0, column].plot(times, calendar[week], label="Calendar baseline")
        axes[0, column].set_ylabel(label)
        axes[0, column].legend(fontsize=8)
        axes[1, column].plot(times, xi[week, column])
        axes[1, column].axhline(0, color="gray", lw=0.7)
        axes[1, column].set_ylabel("Standardized departure xi")
    
    
    for ax in axes[1]:
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    # Close this window to continue to the fitting tasks.
    plt.show(block=True)



def plot_forecast(frame, origin, load, ghi, temp):
    """Plot conditional futures and held-out observations; no fitting occurs."""
    future = slice(origin + 1, origin + 1 + load.shape[1])
    times = frame.timestamp_local_start.iloc[future]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, paths, measured, label in [
        (axes[0], ghi, frame.ghi_w_m2.iloc[future], "GHI [W/m²]"),
        (axes[1], load, frame.load_non_hvac_kw.iloc[future], "Demand [kW]"),
        (axes[2], temp, frame.temperature_c.iloc[future], "Temperature [°C]"),
    ]:
        lower, upper = np.quantile(paths, [0.05, 0.95], axis=0)
        ax.fill_between(times, lower, upper, color="#b4d9e9",
                        label="Pointwise 90% range")
        lines = ax.plot(times, paths[:30].T, color="gray", lw=0.4, alpha=0.3)
        lines[0].set_label("Realizations")
        ax.plot(times, paths.mean(axis=0), color="#126782",
                label="Estimated conditional mean")
        ax.plot(times, measured, color="#d4822b", label="Held-out observations")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    plt.show(block=True)


"""One annual residence simulation with one or N simultaneous realizations.

Configure components in create_model(), then run/debug this file in VS Code.
Parameters belong to their components; ports are vectors over realizations.
Simulation time starts at zero on January 1. Inputs are hourly; the building
and thermostat advance at a shorter integration timestep.
"""


class ForecastSource(Component):
    data_file = Component.Parameter(DATA_FILE)
    realization_count = Component.Parameter(64)
    random_seed = Component.Parameter(964)
    correlated_innovations = Component.Parameter(True)
    innovation_multiplier = Component.Parameter(1.)

    ghi = Component.Output()                    # W/m2
    ambient = Component.Output()                # degC
    non_hvac = Component.Output()               # kW

    def presim_setup(self):
        # Fit and generate once in setup, never during convergence iterations.
        self.forecaster = prepare_forecaster(self.data_file.v)
        self.times, self.paths = generate_annual_paths(self.forecaster,
            scenario_count=int(self.realization_count),
            random_seed=int(self.random_seed),
            use_correlated_innovations=bool(self.correlated_innovations.v),
            innovation_multiplier=float(self.innovation_multiplier))
        self.ghi = self.paths[0, :, 0]
        self.non_hvac = self.paths[0, :, 1]
        self.ambient = self.paths[0, :, 2]

    def calculate(self):
        if self.model.is_converged or not self.model.is_first_iteration:
            return
        # ESCLab labels interval starts in seconds. Hold each hourly input over
        # the shorter integration steps. The first array row is January 1, 00:00.
        hour = int(self.model.time//3600)
        self.ghi = self.paths[hour, :, 0]
        self.non_hvac = self.paths[hour, :, 1]
        self.ambient = self.paths[hour, :, 2]


class Building(Component):
    # Illustrative house and replacement heat pump, not fitted ResStock properties.
    C_b = Component.Parameter(12., units="kWh/K")
    UA = Component.Parameter(0.25, units="kW/K")
    solar_area = Component.Parameter(5., units="m2")   # effective gain aperture
    internal_fraction = Component.Parameter(0.8)
    T_min = Component.Parameter(20., units="degC")
    T_set = Component.Parameter(21., units="degC")
    T_max = Component.Parameter(22., units="degC")
    T_initial = Component.Parameter(21., units="degC")
    heat_capacity = Component.Parameter(16., units="kW")  # thermal
    cool_capacity = Component.Parameter(12., units="kW")
    carnot_fraction = Component.Parameter(0.4)
    hot_approach = Component.Parameter(10., units="K")
    cold_approach = Component.Parameter(7., units="K")
    minimum_lift = Component.Parameter(10., units="K")
    minimum_cop = Component.Parameter(1.2)
    maximum_cop = Component.Parameter(6.)

    ghi = Component.Input()
    ambient = Component.Input()
    non_hvac = Component.Input()
    demand = Component.Output()                 # kW, before PV/battery
    indoor = Component.Output()                 # degC, trial end-of-step state
    heatpump = Component.Output()               # kW electric in either mode
    mode = Component.Output()                   # HEAT=1, OFF=0, COOL=-1
    cop = Component.Output()

    def presim_setup(self):
        # Only read the shared dimension parameter, not another component's
        # setup results: ESCLab initializes components in attribute-name order.
        count = int(self.model.source.realization_count)
        self.temperature_state = np.full(count, float(self.T_initial))
        self.mode_state = np.zeros(count, dtype=int)
        self.indoor = self.temperature_state.copy()
        self.mode = self.mode_state.copy()
        self.heatpump = np.zeros(count)
        self.demand = np.zeros(count)

    def calculate(self):
        if self.model.is_converged:
            self.temperature_state = self.indoor.v.copy()
            self.mode_state = self.mode.v.copy()
            return
        dt_h = self.model.settings.timestep/3600
        if self.model.is_first_iteration:
            old = self.temperature_state
            y = self.mode_state
            self.mode = np.where(y == 1, np.where(old >= self.T_set, 0, 1),
                        np.where(y == -1, np.where(old <= self.T_set, 0, -1),
                        np.where(old <= self.T_min, 1, np.where(old >= self.T_max, -1, 0))))
        self.thermal = np.where(self.mode.v == 1, float(self.heat_capacity),
                               np.where(self.mode.v == -1, -float(self.cool_capacity), 0.))
        self.indoor = ((self.C_b/dt_h)*self.temperature_state + self.UA*self.ambient.v +
            self.solar_area*self.ghi.v/1000 + self.internal_fraction*self.non_hvac.v +
            self.thermal)/(self.C_b/dt_h+self.UA)
        # Teaching COP approximation, not a manufacturer curve. Beginning indoor
        # temperature fixes the exchanger boundary throughout this step.
        # No defrost, backup heat, cycling penalty, or part-load correction.
        heating = self.mode.v == 1
        hot = np.where(heating, self.temperature_state, self.ambient.v) + self.hot_approach + 273.15
        cold = np.where(heating, self.ambient.v, self.temperature_state) - self.cold_approach + 273.15
        lift = np.maximum(hot-cold, float(self.minimum_lift))
        self.cop = np.clip(self.carnot_fraction*np.where(heating, hot, cold)/lift,
                           float(self.minimum_cop), float(self.maximum_cop))
        self.heatpump = abs(self.thermal)/self.cop.v
        self.demand = self.non_hvac.v+self.heatpump.v


class PVBattery(Component):
    capacity = Component.Parameter(10., units="kWh")    # usable capacity
    initial_charge = Component.Parameter(0.5)           # fraction of capacity
    pv_area = Component.Parameter(40., units="m2")
    pv_efficiency = Component.Parameter(0.20)
    pv_rated = Component.Parameter(8., units="kW")
    charge_max = Component.Parameter(5., units="kW")    # electrical-bus power
    discharge_max = Component.Parameter(5., units="kW") # fixed with capacity
    eta_charge = Component.Parameter(0.95)
    eta_discharge = Component.Parameter(0.95)

    ghi = Component.Input()
    demand = Component.Input()
    pv = Component.Output()
    energy = Component.Output()
    soc = Component.Output()
    grid = Component.Output()                  # kW: import positive, export negative
    charge = Component.Output()
    discharge = Component.Output()

    def presim_setup(self):
        count = int(self.model.source.realization_count)
        self.energy_state = np.full(count, self.capacity*self.initial_charge)
        self.energy = self.energy_state.copy()
        self.soc = np.full(count, float(self.initial_charge) if self.capacity > 0 else 0.)
        self.pv = np.zeros(count)
        self.grid = np.zeros(count)
        self.charge = np.zeros(count)
        self.discharge = np.zeros(count)

    def calculate(self):
        if self.model.is_converged:
            self.energy_state = self.energy.v.copy()
            return
        dt_h = self.model.settings.timestep/3600
        self.pv = np.minimum(float(self.pv_rated), self.pv_efficiency*self.pv_area*self.ghi.v/1000)
        surplus = np.maximum(self.pv.v-self.demand.v, 0)
        deficit = np.maximum(self.demand.v-self.pv.v, 0)
        self.charge = np.minimum(np.minimum(surplus, float(self.charge_max)),
            np.maximum(self.capacity-self.energy_state, 0)/(self.eta_charge*dt_h))
        self.discharge = np.minimum(np.minimum(deficit, float(self.discharge_max)),
            self.eta_discharge*np.maximum(self.energy_state, 0)/dt_h)
        self.energy = self.energy_state+(self.eta_charge*self.charge.v-self.discharge.v/self.eta_discharge)*dt_h
        self.grid = self.demand.v-self.pv.v+self.charge.v-self.discharge.v
        self.soc = np.divide(self.energy.v, float(self.capacity),
                             out=np.zeros_like(self.energy.v), where=self.capacity > 0)


class EconomicsMetrics(Component):
    price_buy = Component.Parameter(0.20, units="$/kWh")
    price_sell = Component.Parameter(0.05, units="$/kWh")
    battery_cost = Component.Parameter(300., units="$/kWh")
    study_years = Component.Parameter(15)
    discount_rate = Component.Parameter(0.05)
    # Shared design parameters are assigned from their owners before initialize.
    capacity = Component.Parameter(units="kWh")
    T_min = Component.Parameter(units="degC")
    T_max = Component.Parameter(units="degC")

    grid = Component.Input()
    indoor = Component.Input()
    import_kwh = Component.Output()
    export_kwh = Component.Output()
    purchase_cost = Component.Output()
    export_revenue = Component.Output()
    npc = Component.Output()

    def presim_setup(self):
        count = int(self.model.source.realization_count)
        self.import_kwh = np.zeros(count)
        self.export_kwh = np.zeros(count)
        self.purchase_cost = np.zeros(count)
        self.export_revenue = np.zeros(count)
        self.npc = np.full(count, self.battery_cost*self.capacity)
        self.outside_hours = np.zeros(count)
        self.degree_hours = np.zeros(count)
        self.minimum = np.full(count, np.inf)
        self.maximum = np.full(count, -np.inf)
        self.present_factor = sum((1+float(self.discount_rate))**(-j)
                                  for j in range(1, int(self.study_years)+1))

    def calculate(self):
        if not self.model.is_converged:
            return
        dt_h = self.model.settings.timestep/3600
        self.import_kwh += np.maximum(self.grid.v, 0)*dt_h
        self.export_kwh += np.maximum(-self.grid.v, 0)*dt_h
        excursion = np.maximum(self.T_min-self.indoor.v, 0)+np.maximum(self.indoor.v-self.T_max, 0)
        # Endpoint quadrature, consistent with the backward-Euler scheme.
        self.outside_hours += (excursion > 0)*dt_h
        self.degree_hours += excursion*dt_h
        self.minimum = np.minimum(self.minimum, self.indoor.v)
        self.maximum = np.maximum(self.maximum, self.indoor.v)

    def postsim_calcs(self):
        self.purchase_cost = self.price_buy*self.import_kwh.v
        self.export_revenue = self.price_sell*self.export_kwh.v
        self.operating_cost = self.purchase_cost.v-self.export_revenue.v
        # Repeat this annual operating cost over the study, with upfront capital.
        self.npc = self.battery_cost*self.capacity+self.present_factor*self.operating_cost
        self.results = pd.DataFrame(dict(
            import_kwh=self.import_kwh.v, export_kwh=self.export_kwh.v,
            operating_cost=self.operating_cost, npc=self.npc.v,
            minimum_c=self.minimum, maximum_c=self.maximum,
            outside_hours=self.outside_hours, degree_hours=self.degree_hours))
        self.results.index = pd.RangeIndex(1, len(self.operating_cost)+1, name="realization")
        print("\nAnnual results (mean, minimum, maximum across realizations):")
        print(self.results.agg(["mean", "min", "max"]).round(2).to_string())


def create_model(realization_count=None, show_plots=True, plot_realization=0):
    model = Model()
    model.source = ForecastSource()
    model.building = Building()
    model.battery = PVBattery()
    model.metrics = EconomicsMetrics()

    # Editable run setup. Other physical/economic defaults live on their owners.
    if realization_count is not None:
        model.source.realization_count = realization_count  # 1 or N
    # Example design change: model.battery.capacity = 0.0
    # Weather/load stay hourly. A shorter step resolves indoor temperature and
    # thermostat switching; this does not generate additional weather samples.
    model.settings.timestep = 30*60.                 # seconds
    model.settings.start_time = 0.
    model.settings.progress_update = 100             # milliseconds
    model.settings.tol_rel_global = 1e-8
    model.settings.tol_abs_global = 1e-8
    model.metrics.capacity = model.battery.capacity
    model.metrics.T_min = model.building.T_min
    model.metrics.T_max = model.building.T_max

    model.initialize()  # Also fits and prepares the source's hourly paths.
    model.settings.stop_time = len(model.source.times)*3600

    for source, destination in [
        (model.source.ghi, model.building.ghi),
        (model.source.ambient, model.building.ambient),
        (model.source.non_hvac, model.building.non_hvac),
        (model.source.ghi, model.battery.ghi),
        (model.building.demand, model.battery.demand),
        (model.battery.grid, model.metrics.grid),
        (model.building.indoor, model.metrics.indoor)]:
        model.connect(source, destination, learn_rate=1.)

    if show_plots:
        model.settings.global_plotter_args = dict(show_live=False, time_units="days", grid=False)
        # Percentile envelope and mean across all realizations.
        model.add_plotter([model.building.indoor, model.source.ambient],
                          y1label="Temperature (degC)",
                          array_config=dict(mode="fill_between", fill_alpha=0.2, percentiles=(5, 95)))
        
        # Faint realization paths with a prominent mean on each signal.
        model.add_plotter([output for output in
            (model.building.heatpump, model.building.demand, model.battery.pv, model.battery.grid)],
            y1label="Electrical power (kW)",
            array_config=dict(mode="spaghetti", spaghetti_alpha=0.1, mean_width=2.5))
        
        # Select a zero-based realization without an adapter or copied trace.
        model.add_plotter(model.battery.soc,
                          y1label="Battery SOC",
                          array_config=dict(mode="indices", indices=[plot_realization]))
    return model


def main():
    # started = perf_counter()
    model = create_model()  # Defaults to 64; use realization_count=1 for one path.
    while model.time < model.settings.stop_time:
        model.step()
    # The manual ESCLab loop requires an explicit final-calculation call.
    model.metrics.postsim_calcs()
    # print(f"\n{int(model.source.realization_count)} realizations, one annual simulation: "
    #       f"{perf_counter()-started:.1f} s")
    model.wait_for_plots()
    return model


if __name__ == "__main__":
    main()
