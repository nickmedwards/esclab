# ---------------------------------------------------------------------
# modeling two independent fluid loops using fluid_components.py
# probably need cooler
# 
#    |------------[turbine]------|
#    |      |                    |
#    |      |                    |   flows CW
#    |      ---[tank]------------|  (turbine-side -> _t)
#    |                           |
#    |                           |
# [pump]-------[    heat     ]---|
# [pump]-------[  exchanger  ]---|
#    |      |                    |
#    |      |                    |
#    |      ---[tank]------------|
#    |                           |   flows CW
#    |                           |  (heater-side -> _h)
#    |----------[heater]---------|
# 
# expand the matrix by adding pipes?
# 
#    |--[pipe]-------[turbine]------|
#    |          |                   |
#    |          |                   |
#    |          ---[tank]-----------|
#    |                              |
#    |                            [pipe]
#    |                              |
# [pump]-[pipe]---[    heat     ]---|
# [pump]-[pipe]---[  exchanger  ]---|
#    |          |                   |
#    |          |                   |
#    |          ---[tank]-----------|
#    |                              |
#    |                            [pipe]
#    |                              |
#    |--[pipe]--[heater]------------|
# 
# models written:
#  - [x] heat exchanger
#  - [x] turbine
#  - [x] tank
#  - [x] pump
#  - [x] pipe
#  - [x] heater
#  - [x] tee out
#  - [x] tee return
# ---------------------------------------------------------------------

from esclab.simulate import Model
from esclab.components.fluid_components import *

def create_model():
    model = Model()

    # set model-wide parameters

    # component initialization
    # turbine-side -> _t
    model.TURBINE    = Turbine()
    model.TANK_T     = Tank()
    model.PUMP_T     = Pump()
    model.T_OUT_T    = TeeOut()
    model.T_RETURN_T = TeeReturn()
    # heater-side -> _h
    model.HEATER     = Heater()
    model.TANK_H     = Tank()
    model.PUMP_H     = Pump()
    model.T_OUT_H    = TeeOut()
    model.T_RETURN_H = TeeReturn()
    # heat exchanger
    model.HX = HeatExchanger()

    # Editable run setup. Other physical/economic defaults live on their owners.
    # Use 30 minute timestep - balances mode accuracy and simulation speed. 
    model.settings.timestep = 60.                 # seconds
    model.settings.start_time = 0.
    model.settings.stop_time = 3600.
    model.settings.progress_update = 100
    model.settings.tol_rel_global = 1e-8
    model.settings.tol_abs_global = 1e-8
    model.settings.learn_rate = 1.

    # set component parameters

    # initialize
    model.initialize()

    # connect components
    # turbine-side connections
    model.connect(model.PUMP_T.m_dot_out,     model.T_OUT_T.m_dot_in)
    model.connect(model.PUMP_T.T_out,         model.T_OUT_T.T_in)
    
    model.connect(model.T_OUT_T.m_dot_out_1,  model.TURBINE.m_dot_in)
    model.connect(model.T_OUT_T.T_out,        model.TURBINE.T_in)
    model.connect(model.T_OUT_T.m_dot_out_2,  model.TANK_T.m_dot_in)
    model.connect(model.T_OUT_T.T_out,        model.TANK_T.T_in)
    
    model.connect(model.TURBINE.m_dot_out,    model.T_RETURN_T.m_dot_in_1)
    model.connect(model.TURBINE.T_out,        model.T_RETURN_T.T_in_1)

    # probably need a valve or something to supply m_dot_out to tank
    model.connect(model.TANK_T.m_dot_out,     model.T_RETURN_T.m_dot_in_1)
    model.connect(model.TANK_T.T_out,         model.T_RETURN_T.T_in_2)
    
    model.connect(model.T_RETURN_T.m_dot_out, model.HX.m_dot_tube)
    model.connect(model.T_RETURN_T.T_out,     model.HX.T_in_tube)

    model.connect(model.HX.m_dot_out_tube,    model.PUMP_T.m_dot_in)
    model.connect(model.HX.T_out_tube,        model.PUMP_T.T_in)

    # heater-side connections
    model.connect(model.PUMP_H.m_dot_out,     model.T_OUT_H.m_dot_in)
    model.connect(model.PUMP_H.T_out,         model.T_OUT_H.T_in)

    model.connect(model.T_OUT_H.m_dot_out_1,  model.HX.m_dot_out_shell)
    model.connect(model.T_OUT_H.T_out,        model.HX.T_in_shell)
    model.connect(model.T_OUT_H.m_dot_out_2,  model.TANK_H.m_dot_in)
    model.connect(model.T_OUT_H.T_out,        model.TANK_H.T_in)

    model.connect(model.HX.m_dot_out_shell,   model.T_RETURN_H.m_dot_in_1)
    model.connect(model.HX.T_out_shell,       model.T_RETURN_H.T_in_1)

    # probably need a valve or something to supply m_dot_out to tank
    model.connect(model.TANK_H.m_dot_out,     model.T_RETURN_H.m_dot_in_2)
    model.connect(model.TANK_H.T_out,         model.T_RETURN_H.T_in_2)

    model.connect(model.T_RETURN_H.m_dot_out, model.HEATER.m_dot_in)
    model.connect(model.T_RETURN_H.T_out,     model.HEATER.T_in)

    model.connect(model.HEATER.m_dot_out,     model.PUMP_H.m_dot_in)
    model.connect(model.HEATER.T_out,         model.PUMP_H.T_in)

    return model
