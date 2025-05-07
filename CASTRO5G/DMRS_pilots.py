#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from numba import jit

#PARA EJECUTAR POR SI SOLO, DESCOMENTAR CODIGO COMENTADO

class DMRSConfig:
    def __init__(self, NumLayers=1, PRBSet=None, MappingType='A', SymbolAllocation=(0, 14), 
                 DMRSConfigurationType=1, DMRSLength=1, DMRSAdditionalPosition=0,
                 DMRSTypeAPosition=2, DMRSPortSet=None, NIDNSCID=10, NSCID=0, 
                 NSizeGrid=1):
        
        if DMRSLength == 2 and DMRSAdditionalPosition > 1:
            raise ValueError("DMRSAdditionalPosition must be 0 or 1 when DMRSLength = 2")
            
        self.NumLayers = NumLayers
        self.PRBSet = PRBSet if PRBSet is not None else np.arange(1)
        self.MappingType = MappingType
        self.SymbolAllocation = SymbolAllocation
        self.DMRS = None
        
        self.DMRSConfigurationType = DMRSConfigurationType
        self.DMRSLength = DMRSLength
        self.DMRSAdditionalPosition = DMRSAdditionalPosition
        self.DMRSTypeAPosition = DMRSTypeAPosition
        self.DMRSPortSet = DMRSPortSet if DMRSPortSet is not None else [0]
        self.NIDNSCID = NIDNSCID
        self.NSCID = NSCID
        
        self.NSizeGrid = NSizeGrid
        self.NSlot = 0
        self.SymbolsPerSlot = 14
        self.SubcarriersPerPRB = 12
        self.NSymbols = self.NSizeGrid * self.SymbolsPerSlot
        self.NSubcarriers = self.NSizeGrid * self.SubcarriersPerPRB
    
    def get_pdsch_dmrs_symbol_indices(self):
        start_symbol, duration = self.SymbolAllocation
        
        if self.MappingType == 'A':
            if duration <= 7:
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition],
                        2: [self.DMRSTypeAPosition],
                        3: [self.DMRSTypeAPosition],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                                    
            if duration in (8, 9):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 7],
                        2: [self.DMRSTypeAPosition, 7],
                        3: [self.DMRSTypeAPosition, 7],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                            
            if duration in (10, 11):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 9],
                        2: [self.DMRSTypeAPosition, 6, 9],
                        3: [self.DMRSTypeAPosition, 6, 9],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1, 8, 9],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                                    
            if duration == 12:
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 9],
                        2: [self.DMRSTypeAPosition, 6, 9],
                        3: [self.DMRSTypeAPosition, 5, 8, 11],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1, 8, 9],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                                    
            if duration in (13, 14):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 11],
                        2: [self.DMRSTypeAPosition, 7, 11],
                        3: [self.DMRSTypeAPosition, 5, 8, 11],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1, 10, 11],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                    
        elif self.MappingType == 'B':
            if duration <= 4:
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol],
                        2: [start_symbol],
                        3: [start_symbol],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [],
                        1: [],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
    
            if duration in (5, 6, 7):
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 4],
                        2: [start_symbol, 4],
                        3: [start_symbol, 4],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                    
            if duration == 8:
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 6],
                        2: [start_symbol, 3, 6],
                        3: [start_symbol, 3, 6],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 5, 6],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                    
            if duration == 9:
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 7],
                        2: [start_symbol, 4, 7],
                        3: [start_symbol, 4, 7],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 5, 6],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                    
            if duration == 10:
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 7],
                        2: [start_symbol, 4, 7],
                        3: [start_symbol, 4, 7],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 7, 8],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
    
            if duration == 11:
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 8],
                        2: [start_symbol, 4, 8],
                        3: [start_symbol, 3, 6, 9],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 7, 8],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
    
            if duration in (12, 13):
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 9],
                        2: [start_symbol, 5, 9],
                        3: [start_symbol, 3, 6, 9],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 8, 9],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                
            if duration == 14:
                if self.DMRSLength == 1:
                    table = {
                        0: [],
                        1: [],
                        2: [],
                        3: [],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [],
                        1: [],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
        else:
            raise ValueError("Unsupported Mapping type (A or B)")
    
        if self.DMRSAdditionalPosition not in table:
            raise ValueError("Unsupported additional position. Use (0,1,2,3) for DMRSLength=1, (0,1) for DMRSLength=2")

        return sorted(table[self.DMRSAdditionalPosition])

    def get_pusch_dmrs_symbol_indices(self):
        start_symbol, duration = self.SymbolAllocation
        
        if self.MappingType == 'A':
            if duration < 4:
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition],
                        2: [self.DMRSTypeAPosition],
                        3: [self.DMRSTypeAPosition],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                    
            if duration in (4, 5, 6, 7):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition],
                        2: [self.DMRSTypeAPosition],
                        3: [self.DMRSTypeAPosition],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                                    
            if duration in (8, 9):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 7],
                        2: [self.DMRSTypeAPosition, 7],
                        3: [self.DMRSTypeAPosition, 7],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                            
            if duration in (10, 11):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 9],
                        2: [self.DMRSTypeAPosition, 6, 9],
                        3: [self.DMRSTypeAPosition, 6, 9],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1, 8, 9],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                                    
            if duration == 12:
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 9],
                        2: [self.DMRSTypeAPosition, 6, 9],
                        3: [self.DMRSTypeAPosition, 5, 8, 11],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1, 8, 9],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                                    
            if duration in (13, 14):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 11],
                        2: [self.DMRSTypeAPosition, 7, 11],
                        3: [self.DMRSTypeAPosition, 5, 8, 11],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1, 10, 11],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                    
        elif self.MappingType == 'B':
            if duration <= 4:
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol],
                        2: [start_symbol],
                        3: [start_symbol],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [],
                        1: [],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
    
            if duration in (5, 6, 7):
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 4],
                        2: [start_symbol, 4],
                        3: [start_symbol, 4],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                    
            if duration in (8, 9):
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 6],
                        2: [start_symbol, 3, 6],
                        3: [start_symbol, 3, 6],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 5, 6],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                    
            if duration in (10, 11):
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 8],
                        2: [start_symbol, 4, 8],
                        3: [start_symbol, 3, 6, 9],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, start_symbol+1, 7, 8],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                    
            if duration in (12, 13, 14):
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 10],
                        2: [start_symbol, 5, 10],
                        3: [start_symbol, 3, 6, 9],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 9, 10],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
        else:
            raise ValueError("Unsupported Mapping type (A or B)")
    
        if self.DMRSAdditionalPosition not in table:
            raise ValueError("Unsupported additional position. Use (0,1,2,3) for DMRSLength=1, (0,1) for DMRSLength=2")
    
        return sorted(table[self.DMRSAdditionalPosition])

    def generate_dmrs_grid(self, pilot):
        grid = np.zeros((self.NSubcarriers, self.NSymbols, self.NumLayers), dtype=complex)
        
        match pilot:
            case "PDSCH":
                dmrs_symbols = self.get_pdsch_dmrs_symbol_indices()
            case "PUSCH":
                dmrs_symbols = self.get_pusch_dmrs_symbol_indices()
            case _:
                print("Invalid option. Use PDSCH or PUSCH")
                return grid
    
        symbol_start, symbol_len = self.SymbolAllocation
        symbol_end = symbol_start + symbol_len
        
        for layer, port in enumerate(self.DMRSPortSet):
            sc_offsets = self.get_dmrs_subcarrier_offsets(port)
            valid_symbols = [s for s in dmrs_symbols if symbol_start <= s < symbol_end]
            
            # Precompute c_init for all symbols and PRBs
            c_init = np.array([
                ((2 ** 17) * (self.SymbolsPerSlot * self.NSlot + s + 1) * (2 * self.NIDNSCID + 1) + 2 * self.NIDNSCID + self.NSCID) % (2 ** 31)
                for s in valid_symbols for prb in self.PRBSet for rep in range(self.NSizeGrid)
            ])
            
            # Generate all sequences at once
            seq_length = len(sc_offsets) * 2
            c = np.array([generate_gold_sequence_numba(c_i, seq_length) for c_i in c_init])
            r = qpsk_modulate(c.flatten()).reshape(-1, len(sc_offsets))
            
            # Assign to grid
            idx = 0
            for s in valid_symbols:
                for prb in self.PRBSet:
                    for rep in range(self.NSizeGrid):
                        for i, offset in enumerate(sc_offsets):
                            k = prb * self.SubcarriersPerPRB + offset
                            l = rep * self.SymbolsPerSlot + s
                            grid[k, l, layer] = self.apply_cdm(r[idx, i], port)
                        idx += 1
        
        # for port_idx in range(self.NumLayers):
        #     print(f"\n signal[:, :, {self.DMRSPortSet[port_idx]-1000}]")
        #     print(grid[:, :, port_idx])
        #     print('----------------------------------------------------------------')
        
        return grid
    def get_dmrs_subcarrier_offsets(self, port):
        port_idx = port - 1000
        k_values = []
        if self.DMRSConfigurationType == 1:
            if (port_idx % 4) in [0, 1]:
                delta = 0
                for n, k_p in product((0, 1, 2), (0, 1)):
                    k=4*n+2*k_p+delta
                    k_values.append(k)
                return np.array(k_values)
            else:
                delta = 1
                for n, k_p in product((0, 1, 2), (0, 1)):
                    k=4*n+2*k_p+delta
                    k_values.append(k)
                return np.array(k_values)
            
        elif self.DMRSConfigurationType == 2:
            if (port_idx % 6) in [0, 1]:
                delta = 0
                for n, k_p in product((0, 1), (0, 1)):
                    k=6*n+k_p+delta
                    k_values.append(k)
                return np.array(k_values)
            elif (port_idx % 6) in [2, 3]:
                delta = 2
                for n, k_p in product((0, 1), (0, 1)):
                    k=6*n+k_p+delta
                    k_values.append(k)
                return np.array(k_values)
            else:
                return np.array([0, 1, 6, 7])
            
        raise ValueError("Unsupported DMRS configuration. Use 1 or 2")
    
    def apply_cdm(self, r, port):
        port_idx = port - 1000
        if self.DMRSConfigurationType == 1:
            if (port_idx % 2 == 1 and port_idx <= 3) or (port_idx % 2 == 0 and port_idx > 3):
                r *= (-1)
        elif self.DMRSConfigurationType == 2:
            if (port_idx % 2 == 1 and port_idx <= 5) or (port_idx % 2 == 0 and port_idx > 5):
                r *= (-1)
        return r
    
    # def plot_dmrs(self, grid):
    #     fig, axes = plt.subplots(1, self.NumLayers, figsize=(15, 5))
    #     if self.NumLayers == 1:
    #         axes = [axes]
        
    #     # Create a custom colormap where 0=white and non-zero=red
    #     from matplotlib.colors import ListedColormap
    #     cmap = ListedColormap(['white', 'red'])
        
    #     for i in range(self.NumLayers):
    #         ax = axes[i]
    #         # Create a binary mask for DMRS presence
    #         dmrs_mask = (np.abs(grid[:, :, i]) > 0).astype(int)
            
    #         # Calculate correct y-axis limits and ticks
    #         y_min, y_max = -0.5, self.NSubcarriers - 0.5
    #         y_ticks = np.arange(0, self.NSubcarriers)  # Show all subcarriers
    #         y_tick_labels = [str(tick) for tick in y_ticks]  # Only label PRB starts
            
    #         ax.imshow(dmrs_mask, aspect='auto', origin='upper', cmap=cmap,
    #                       interpolation='none', vmin=0, vmax=1,
    #                       extent=[-0.5, self.NSymbols - 0.5, y_max, y_min])
            
    #         ax.set_title(f'DMRS Port {self.DMRSPortSet[i]} (Config Type {self.DMRSConfigurationType})')
    #         ax.set_xlabel('OFDM Symbols')
    #         ax.set_ylabel('Subcarriers')
    #         ax.set_xticks(np.arange(self.NSymbols))
    #         ax.set_yticks(y_ticks)
    #         ax.set_yticklabels(y_tick_labels)
    #         ax.set_ylim(y_min, y_max)
            
    #         # Mark k boundaries
    #         for sc in range(0, self.NSubcarriers, self.SubcarriersPerPRB):
    #             ax.axhline(sc - 0.5, color='gray', linestyle='-', linewidth=1.5)
                
    #         # Mark l boundaries
    #         for sc in range(0, self.NSymbols, self.SymbolsPerSlot):
    #             ax.axvline(sc - 0.5, color='gray', linestyle='-', linewidth=1.5)
            
    #         # Optional: thinner lines for each subcarrier
    #         for sc in range(1, self.NSubcarriers):
    #             ax.axhline(sc - 0.5, color='lightgray', linestyle=':', linewidth=0.5)
            
    #         # Mark symbol boundaries
    #         for sym in range(self.NSymbols + 1):
    #             ax.axvline(sym - 0.5, color='gray', linestyle='--', linewidth=0.5)
            
    #     plt.tight_layout()
    #     plt.show()
   
@jit(nopython=True)
def generate_gold_sequence_numba(c_init, length):
    N_c = 1600
    x1 = np.zeros(N_c + length, dtype=np.int32)
    x2 = np.zeros(N_c + length, dtype=np.int32)

    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1

    for n in range(31, N_c + length):
        x1[n] = (x1[n - 3] + x1[n - 31]) % 2
        x2[n] = (x2[n - 3] + x2[n - 2] + x2[n - 1] + x2[n - 31]) % 2

    c = (x1[N_c:] + x2[N_c:]) % 2
    return c

def qpsk_modulate(bits):
    return (1 / np.sqrt(2)) * ((1 - 2 * bits[0::2]) + 1j * (1 - 2 * bits[1::2]))

# def configure_dmrs(pilot, num_layers, config_type=1, symbol_allocation=(0, 14), num_prbs_k=1, num_prbs_l=1, mapping_type='A',
#                      dmrs_length=1, dmrs_add_pos=0, dmrs_type_a_pos=2):
    
#     dmrs = DMRSConfig(NumLayers=num_layers, 
#                         PRBSet=np.arange(num_prbs_k), 
#                         MappingType=mapping_type, 
#                         SymbolAllocation=symbol_allocation,
#                         DMRSConfigurationType=config_type,
#                         DMRSLength=dmrs_length,
#                         DMRSAdditionalPosition=dmrs_add_pos,
#                         DMRSTypeAPosition=dmrs_type_a_pos,
#                         DMRSPortSet=[1000 + i for i in range(num_layers)],
#                         NIDNSCID=10,
#                         NSCID=0,
#                         NSizeGrid=num_prbs_l)
    
#     dmrs.DMRS = dmrs
#     pilots = dmrs.generate_dmrs_grid(pilot)
#     dmrs.plot_dmrs(pilots)
    
#     return pilots

# if __name__ == "__main__":
#     # Example
#     dmrs_pilot="PDSCH"
#     ports = 4
#     configuration_type = 2
#     symbol_allocation = (0, 14)
#     num_prbs_k=1
#     num_prbs_l=1
#     mapping_type='A'
#     pilot_length=2
#     pilot_add_pos=1
#     dmrs_type_a_pos=2

#     Pilots = configure_dmrs(pilot=dmrs_pilot,
#                             num_layers=ports, 
#                             config_type=configuration_type, 
#                             symbol_allocation=symbol_allocation,
#                             num_prbs_k=num_prbs_k,
#                             num_prbs_l=num_prbs_l,
#                             mapping_type=mapping_type, 
#                             dmrs_length=pilot_length, 
#                             dmrs_add_pos=pilot_add_pos,
#                             dmrs_type_a_pos=dmrs_type_a_pos)