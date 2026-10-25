"""
This non-running file serves as a starting template for construction new models. 

Refer to esclab.components.template_component for a corresponding template for building new component libraries.

Models have several required calls to create, initialize, and configure the simulation.
"""
# Import necessary simulation tools
from esclab.simulate import *
# Import components used in this model. 
# >>> replace the following line with the library components you need
import esclab.components.template_component as comp


# ############################################################################
#                   Set up the model 
# ############################################################################

# Instantiate the model. The model is the container for all components, connections, and simulation settings.
model = Model()

# ----------------------------------------------------------
# Configure simulation settings. See Model.Settings in simulate.py for all available settings and defaults.
# >>> configure model settings here as needed. 
# >>> Settings: timestep, start_time, stop_time, tol_rel_global, tol_abs_global, max_iterations, learn_rate, progress_update
# For example:
model.settings.stop_time = 10  # seconds

# ----------------------------------------------------------
# Add components to the model by instantiating them as attributes with a name that you choose. 
# These models should be imported from the component libraries, or can be defined at the top of this file.
# >>> add components here as needed. Use any member naming scheme that you like. 
model.ca_1 = comp.FirstComponent()
model.ca_2 = comp.FirstComponent()
model.cb = comp.SecondComponent()

# ----------------------------------------------------------
# Initialize the model. This step is required to set up the internal data structures for the simulation.
model.initialize()


# ----------------------------------------------------------
# Do any additional parameter setting or configuration of the components here. 
# >>> For example:
model.ca_1.a = 1
model.ca_1.b = 0
model.ca_2.a = .5
model.ca_2.b = 2

# ----------------------------------------------------------
# Make connections between components
# For example:
model.connect( model.ca_1.y_out, model.ca_2.x_in, tol_abs=1e-6, tol_rel=1e-6, learn_rate=0.8)

# If using the simultaneous equation solving feature, connections must be assigned to a solve group. 
# For example: 
model.connect( model.ca_1.y_out, model.ca_2.x_in,  solve_group="group1")
model.connect( model.ca_2.x_out, model.cb.x_in,  solve_group="group1")

# ----------------------------------------------------------
# Set up the plotters

model.add_plotter([model.ca_1.y_out, model.ca_2.y_out, model.cb.y_out],
              #   [...],                    # Items to plot on the secondary y-axis
                  y1label="Temperature",    # label for the y-axis of the plot
              #   y2label="Flow",           # label for the secondary y-axis of the plot (if plotting two variables with different units)
                  update_every=1,           # render the plot every n time steps (set to 1 to update every time step)
                  nmax_points=1000,         # maximum number of points to show on the plot (older points will be dropped off as new points are added)
                  show_live=True,           # whether to show the plot live during the simulation (set to False to only show the plot at the end of the simulation
                  )
# Add multiple plotters as desired.
# model.add_plotter([model.ca_1.x_in, model.ca_2.x_in, model.cb.x_in], y1label="Current (A)", update_every=5, nmax_points=1000, show_live=True)

# [optional] Add a network graph to visualize the model structure. This is available for models that have coupled_eqn solving.
# model.add_network_graph()

# ----------------------------------------------------------
# Run the simulation
# The final step calls each component's postsim_calcs() once in execution order.
# To extend stop_time later, set model.settings.auto_postsim_calcs = False before
# stepping, then call model.finalize() explicitly when the entire run is finished.
while model.time < model.settings.stop_time:
    model.step()

# force the plots to stay open after the simulation finishes
model.wait_for_plots()






