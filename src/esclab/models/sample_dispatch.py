from esclab.simulate import *
import numpy as np
import gurobipy as gp 
from gurobipy import GRB 
from eeslib.functions import convert
import time 

class Design:
    def __init__(self):
        self.producer_rating = 1000  # J/s
        self.storage_capacity = self.producer_rating * 5 * convert('hr','s')  # J
        self.storage_initial_charge = self.storage_capacity * 0.5  # J
        self.c_rate = 0.25  # 1/hr  Fraction of total storage capacity that can be charged or discharged per hour
        self.consumer_efficiency = 0.5  
        self.optimization_horizon = 48 * convert('hr','s')  # s 
        self.control_horizon = 24 * convert('hr','s')  # s 


class Storage(Component):
    capacity           = Component.Parameter(1)
    capacity_init      = Component.Parameter(1)
    c_rate             = Component.Parameter(1)  #Max charge/discharge rate, as a fraction of present charge
    loss_rate          = Component.Parameter(1)  #Loss as a fraction of capacity per s

    flow_in            = Component.Input()  #J/s
    flow_out           = Component.Input()  #J/s

    flow_in_actual     = Component.Output()  #J/s
    flow_out_actual    = Component.Output()  #J/s
    charge             = Component.Output()  #J
    losses             = Component.Output()  #J/s
    last_charge = 0.  #state of charge

    def presim_setup(self, **kwargs):
        self.last_charge = self.capacity_init
        pass 

    def calculate(self):
        # Reality checks
        assert self.flow_in >= 0
        assert self.flow_out >= 0

        # Post-convergence
        if self.model.is_converged:
            self.last_charge = self.charge
            return

        # Losses from last time step
        losses = max(self.last_charge * self.loss_rate * self.model.settings.timestep , 0)
        self.losses = losses
        # Energy balance
        self.charge = self.last_charge - losses + (self.flow_in - self.flow_out)*self.model.settings.timestep 
        
        self.flow_out_actual = self.flow_out
        self.flow_in_actual = self.flow_in 
        # Correct flows if resulting charge exceeds min/max capacity
        if self.charge < 0.:
            self.flow_out_actual += self.charge/self.model.settings.timestep 
            self.charge = 0.
        if self.charge >= self.capacity:
            self.flow_in_actual -= (self.charge - self.capacity)/self.model.settings.timestep
            self.charge = self.capacity
        if self.flow_out > self.flow_out_actual:
            pass
        return

class Producer(Component):
    rated_power    = Component.Parameter(1.)
    horizon        = Component.Parameter(24)

    power          = Component.Output()
    power_forecast = Component.Output()

    def presim_setup(self, **kwargs):
        # initialize the power forecast array
        fctimes = np.arange(self.model.settings.start_time, self.model.settings.start_time + self.horizon, self.model.settings.timestep)
        self.power_forecast = self.__generate_forecast(fctimes)

    def calculate(self):

        # Post-convergence 
        if self.model.is_converged:
            self.power_forecast = np.roll(self.power_forecast, -1)
            self.power_forecast[-1] = self.new_fc_value[0]
            return

        # Iteration calculations
        self.power = self.power_forecast[0] 
        # Only update the new forecast value on the first iteration
        if self.model.iteration == 0:
            self.new_fc_value = self.__generate_forecast(np.array([self.model.time + self.horizon]))

        return
    
    def __generate_forecast(self, time_values):
        tm_adj = (time_values/self.model.settings.timestep % 24.)*(np.pi)/(24.)
        y = np.sin(tm_adj)*self.rated_power #* 3/2 - self.rated_power*1/2
        y[y<0] = 0.
        lmask = np.abs(np.random.uniform(0,2,len(time_values))) - 0.5
        lmask[lmask<0] = 0.
        lmask[lmask>1] = 1.
        y = lmask*y
        return y
        
class Consumer(Component):
    
    horizon    = Component.Parameter(1)
    efficiency = Component.Parameter(1)
    
    flow_in    = Component.Input()

    price      = Component.Output()
    revenue    = Component.Output()
    price_forecast = Component.Output()


    def presim_setup(self, **kwargs):
        
        fctimes = np.arange(self.model.settings.start_time, self.model.settings.start_time+self.horizon, self.model.settings.timestep)
        self.price_forecast = self.__generate_forecast(fctimes)

    def calculate(self):

        # Post-convergence
        if self.model.is_converged:
            self.price_forecast = np.roll(self.price_forecast, -1)
            self.price_forecast[-1] = self.new_fc_value[0]
            return

        self.price = self.price_forecast[0]
        # Only update the new forecast value on the first iteration
        if self.model.iteration == 0:
            self.new_fc_value = self.__generate_forecast(np.array([self.model.time + self.horizon]))

        self.revenue = self.flow_in * self.model.settings.timestep * self.efficiency * self.price

    def __generate_forecast(self, time_values):
        tm_adj = (time_values % 24.)*(2*np.pi)/(24.)
        y = (np.cos(tm_adj)*np.random.normal(0,.1,len(tm_adj))+1)*1
        return y
    
class Scheduler(Component):
    optimization_horizon    = Component.Parameter(1.)
    control_horizon         = Component.Parameter(1.)
    storage_initial_charge  = Component.Parameter(1.)
    storage_capacity        = Component.Parameter(1.)
    c_rate                  = Component.Parameter(1.)
    consumer_efficiency     = Component.Parameter(1.)

    # initialize schedules with dummy arrays of the right length
    charge_schedule    = Component.Input(np.array([]))
    price_schedule     = Component.Input(np.array([]))
    storage_charge     = Component.Input()

    flow_to_consumer   = Component.Output()
    flow_from_producer = Component.Output()
    price_now          = Component.Output()
    charge_avail_now   = Component.Output()
    num_iter           = Component.Output()

    def presim_setup(self, **kwargs):
        # initialize schedules with dummy arrays of the right length
        da = np.ones(int(self.optimization_horizon/self.model.settings.timestep))
        self.charge_schedule = da
        self.price_schedule = da

        # Initialize storage charge state
        self.last_charge_state = self.storage_initial_charge

    def calculate(self):
        
        # post convergence 
        if self.model.is_converged:
            # Only update the initial charge state for the next step after the current step has converged
            self.last_charge_state = self.storage_charge
            self.num_iter = self.model.iteration
            return

        t_rel = int((self.model.time % self.control_horizon)*convert('s','hr'))
        if t_rel == 0:
            # Check for missing input data
            if np.isnan(self.storage_charge):
                # provide temporary data
                self.active_schedule = [{'flow_from_producer':0, 'flow_to_consumer':0}]
            else:
                if self.model.iteration < 2:
                    self.active_schedule = self.__run_opt_model()
        self.price_now = self.price_schedule[t_rel]
        self.charge_avail_now = self.active_schedule[t_rel]['charge']
        self.flow_to_consumer = self.active_schedule[t_rel]['flow_to_consumer']
        self.flow_from_producer = self.active_schedule[t_rel]['flow_from_producer']

    def __run_opt_model(self):
        # md = self.model.design

        nt = len(self.price_schedule)
        T = range(nt)
        # Pass numeric scalars to Gurobi, not iterable ESCLab I/O wrappers.
        smax = float(self.storage_capacity)
        qmax = smax * float(self.c_rate) / convert('hr','s')  # W
        S0 = float(self.last_charge_state)
        
        # Optimization Model
        om = gp.Model('storage_optimization')
        om.setParam('OutputFlag',0)
        
        # Variables
        q_out = om.addVars(T, lb=0., ub=qmax, name='q_out')  # flow to consumer
        q_in =  om.addVars(T, lb=0., ub=qmax, name='q_in')   # flow in used
        s = om.addVars(T, lb=0., ub=smax, name='s')  # storage inventory

        # Objective function
        eta = float(self.consumer_efficiency)
        om.setObjective(gp.quicksum(q_out[t]*eta*self.price_schedule[t]*self.model.settings.timestep for t in T), GRB.MAXIMIZE)
        
        # ------------ Constraints
        # flow in is no greater than available flow in 
        om.addConstrs((q_in[t] <= self.charge_schedule[t] for t in T), 'flow_utilized')

        # energy balance on storage based on power consumed
        om.addConstrs((s[t] == (s[t-1] if t>0 else S0) - (q_out[t] - q_in[t])*self.model.settings.timestep for t in T), 'store_balance')

        om.optimize()

        if om.status == GRB.OPTIMAL:
            res = []
            for t in T:
                res.append({
                    'flow_from_producer':q_in[t].X,
                    'flow_to_consumer':q_out[t].X,
                    'charge':s[t].X,
                })
            return res 
            # print("Optimal objective value:", model.objVal)
        else:
            raise RuntimeError("Optimization model did not converge")


if __name__ == "__main__":
    model = Model()
    model.settings.stop_time = 30*convert('day','s')
    model.settings.timestep = 3600

    model.producer = Producer()
    model.storage = Storage()
    model.consumer = Consumer()
    model.scheduler = Scheduler()

    # ---------------------- Set parameters and initial values

    design = Design()

    # storage
    model.storage.capacity       = design.storage_capacity
    model.storage.c_rate         = design.c_rate  #Max charge/discharge rate, as a fraction of present charge
    model.storage.capacity_init  = design.storage_initial_charge
    model.storage.loss_rate      = 0.001 / convert('h','s')  #Loss as a fraction of capacity per s
    model.storage.flow_in        = 0  #J/s
    model.storage.flow_out       = 0  #J/s

    # Producer
    model.producer.rated_power    = design.producer_rating
    model.producer.horizon        = design.optimization_horizon

    # Consumer
    model.consumer.horizon    = design.optimization_horizon
    model.consumer.efficiency = design.consumer_efficiency
    model.consumer.flow_in    = 0.

    # Scheduler
    model.scheduler.optimization_horizon    = design.optimization_horizon
    model.scheduler.control_horizon         = design.control_horizon
    model.scheduler.storage_initial_charge  = design.storage_initial_charge
    model.scheduler.storage_capacity        = design.storage_capacity
    model.scheduler.c_rate                  = design.c_rate
    model.scheduler.consumer_efficiency     = design.consumer_efficiency

    # ------------- from ----------------------------- to ----------------------
    model.connect(model.scheduler.flow_from_producer , model.storage.flow_in           ) 
    model.connect(model.scheduler.flow_to_consumer   , model.storage.flow_out          )
    model.connect(model.storage.flow_out_actual      , model.consumer.flow_in          ) #, log_n_iter=model.settings.max_iterations)
    model.connect(model.producer.power_forecast      , model.scheduler.charge_schedule )
    model.connect(model.consumer.price_forecast      , model.scheduler.price_schedule  )
    model.connect(model.storage.charge               , model.scheduler.storage_charge  )

    model.add_plotter([model.storage.flow_in, model.storage.flow_out], [model.storage.charge, model.scheduler.charge_avail_now], update_every=24, nmax_points=24*7)
    model.add_plotter([model.consumer.revenue], [model.consumer.price], update_every=12, nmax_points=24*7)
    model.add_plotter([model.scheduler.num_iter], update_every=24, tab_title="sim info")

    model.add_network_graph()

    start = time.time()
    while model.time < model.settings.stop_time:
        model.step()
    print(f'time: {time.time()-start}')
    model.wait_for_plots()
