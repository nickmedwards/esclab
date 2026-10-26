# ---------------------------------------------------------------------
# modeling two independent fluid loops using fluid_components.py
# 
#    |------------[turbine]------|
#    |      |                    |
#    |      |                    |
#    |      ---[tank]------------|
#    |                           |
#    |                           |
# [pump]-------[    heat     ]---|
# [pump]-------[  exchanger  ]---|
#    |      |                    |
#    |      |                    |
#    |      ---[tank]------------|
#    |                           |
#    |                           |
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
# models implemented:
#  - [ ] heat exchanger
#  - [ ] turbine
#  - [ ] tank
#  - [x] pump
#  - [x] pipe
#  - [x] heater
#  - [x] tee out
#  - [x] tee return
# ---------------------------------------------------------------------

