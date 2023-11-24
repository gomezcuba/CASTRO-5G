import numpy as np

import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg

model = pg.ThreeGPPMultipathChannelModel(scenario="UMi")

txPos1 = np.array((0, 0, 10))
rxPos1 = np.array((1, 0.5, 10)) 
plinfo1, macro1, clusters1, subpaths1 = model.create_channel(txPos1, rxPos1)

txPos2 = np.array((0, 0, 10))
rxPos2 = np.array((1, 5, 10))  
plinfo2, macro2, clusters2, subpaths2 = model.create_channel(txPos2, rxPos2)


txPos3 = np.array((0, 0, 10))
rxPos3 = np.array((1.5, 0.5, 10)) 
plinfo3, macro3, clusters3, subpaths3 = model.create_channel(txPos3, rxPos3)

txPos4 = np.array((0, 0, 10))
rxPos4 = np.array((1.5, 5.5, 10))  
plinfo4, macro4, clusters4, subpaths4 = model.create_channel(txPos4, rxPos4 )


txPos5 = np.array((0, 1, 10))
rxPos5 = np.array((1.5, 5.5, 10))
plinfo5, macro5, clusters5, subpaths5 = model.create_channel(txPos5, rxPos5 )



# Verificar los resultados
print("Canal 1:")
print("PL Info:", plinfo1)
print("Macro:", macro1)
print("Clusters:", clusters1)
print("Subpaths:", subpaths1)

# Verificar los resultados
print("Canal 2:")
print("PL Info:", plinfo2)
print("Macro:", macro2)
print("Clusters:", clusters2)
print("Subpaths:", subpaths2)


# Verificar los resultados
print("Canal 3:")
print("PL Info:", plinfo3)
print("Macro:", macro3)
print("Clusters:", clusters3)
print("Subpaths:", subpaths3)


# Verificar los resultados
print("Canal 4:")
print("PL Info:", plinfo4)
print("Macro:", macro4)
print("Clusters:", clusters4)
print("Subpaths:", subpaths4)


# Verificar los resultados
print("Canal 5:")
print("PL Info:", plinfo5)
print("Macro:", macro5)
print("Clusters:", clusters5)
print("Subpaths:", subpaths5)
