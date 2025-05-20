#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import product

class DMRSConfig:
    def __init__(self, NumLayers=1, PRBSet=1, MappingType='A', SymbolAllocation=(0, 14), 
                 DMRSConfigurationType=1, DMRSLength=1, DMRSAdditionalPosition=0,
                 DMRSTypeAPosition=2, DMRSPortSet=None, NIDNSCID=10, NSCID=0, 
                 NSizeSymbol=1, NSlot_list=[0]):
        
        if DMRSLength == 2 and DMRSAdditionalPosition > 1:
            raise ValueError("DMRSAdditionalPosition must be 0 or 1 when DMRSLength = 2")
        
        if not isinstance(NSlot_list, list):
            raise ValueError("NSlot_list must be a list of slot indices")
        
        if len(NSlot_list) != NSizeSymbol:
            raise ValueError(f"Length of NSlot_list ({len(NSlot_list)}) must equal NSizeSymbol ({NSizeSymbol})")
            
        self.NumLayers = NumLayers
        self.PRBSet = PRBSet
        self.MappingType = MappingType
        self.SymbolAllocation = SymbolAllocation
        
        self.DMRSConfigurationType = DMRSConfigurationType
        self.DMRSLength = DMRSLength
        self.DMRSAdditionalPosition = DMRSAdditionalPosition
        self.DMRSTypeAPosition = DMRSTypeAPosition
        self.DMRSPortSet = DMRSPortSet if DMRSPortSet is not None else [0]
        self.NIDNSCID = NIDNSCID
        self.NSCID = NSCID
        
        self.NSizeSymbol = NSizeSymbol
        self.SymbolsPerSlot = 14
        self.SubcarriersPerPRB = 12
        self.NSymbols = self.NSizeSymbol * self.SymbolsPerSlot
        self.NSubcarriers = self.PRBSet * self.SubcarriersPerPRB
        
        self.NSlot_list = NSlot_list
    
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
        grid = np.zeros((self.NSymbols, self.NSubcarriers, self.NumLayers), dtype=complex)
        
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
            for _, symbol in enumerate(dmrs_symbols):
    
                if not (symbol_start <= symbol < symbol_end):
                    continue
                
                for prb, rep in product(np.arange(self.PRBSet), np.arange(self.NSizeSymbol)):
                    c_init = ((2 ** 17) * (14 * self.NSlot_list[rep] + symbol + 1) * (2 * self.NIDNSCID + 1) + 2 * self.NIDNSCID + self.NSCID) % (2 ** 31)
                    c = self.generate_gold_sequence(c_init, len(sc_offsets) * 2)
                    r = self.qpsk_modulate(c)
                    
                    for i, offset in enumerate(sc_offsets):
                        k = prb * self.SubcarriersPerPRB + offset
                        l = rep * self.SymbolsPerSlot + symbol
                        symbol_val = self.apply_cdm(r[i], port)
                        grid[l, k, layer] = symbol_val

        # for port_idx in range(self.NumLayers):
        #     print(f"\n signal[:, :, {self.DMRSPortSet[port_idx]-1000}]")
        #     print(grid[:, :, port_idx].T)
        #     print('----------------------------------------------------------------')
                            
        return grid, dmrs_symbols

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
   
    def generate_gold_sequence(self, c_init, length):
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
    
    def qpsk_modulate(self, bits):
        return (1 / np.sqrt(2)) * ((1 - 2 * bits[0::2]) + 1j * (1 - 2 * bits[1::2]))

    def plot_dmrs(self, grid):
        fig, axes = plt.subplots(1, self.NumLayers, figsize=(15, 5))
        if self.NumLayers == 1:
            axes = [axes]
        
        colors = ['blue', 'red', 'white', 'green', 'yellow']
        cmap = ListedColormap(colors[:len(np.unique(grid))])
        
        for i in range(self.NumLayers):
            ax = axes[i]
    
            unique_vals, indices = np.unique(grid[:, :, i], return_inverse=True)
            indices = indices.reshape(grid[:, :, i].shape).T
            
            y_min, y_max = -0.5, self.NSubcarriers - 0.5
            y_ticks = np.arange(0, self.NSubcarriers)
            y_tick_labels = [str(tick) for tick in y_ticks]
            
            x = np.arange(-0.5, self.NSymbols)
            y = np.arange(-0.5, self.NSubcarriers)
            
            im = ax.pcolormesh(x, y, indices, cmap=cmap, vmin=0, vmax=len(unique_vals)-1,
                              edgecolors='k', linewidths=0.5)
            
            cbar = plt.colorbar(im, ax=ax, ticks=np.arange(len(unique_vals)))
            cbar.ax.set_yticklabels([f"{val:.2f}" for val in unique_vals])
            
            ax.set_title(f'DMRS Port {self.DMRSPortSet[i]} (Config Type {self.DMRSConfigurationType})')
            ax.set_xlabel('OFDM Symbols')
            ax.set_ylabel('Subcarriers')
            ax.set_xticks(np.arange(self.NSymbols))
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels)
            ax.set_ylim(y_min, y_max)
            
            for sc in range(0, self.NSubcarriers, self.SubcarriersPerPRB):
                ax.axhline(sc - 0.5, color='gray', linestyle='-', linewidth=1.5)
                
            for sc in range(0, self.NSymbols, self.SymbolsPerSlot):
                ax.axvline(sc - 0.5, color='gray', linestyle='-', linewidth=1.5)
            
            for sc in range(self.NSubcarriers):
                ax.axhline(sc - 0.5, color='lightgray', linestyle=':', linewidth=0.5)
            
            for sym in range(self.NSymbols + 1):
                ax.axvline(sym - 0.5, color='gray', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
    
def configure_dmrs(pilot, num_layers, config_type=1, symbol_allocation=(0, 14), num_prbs_k=1, num_prbs_l=1, mapping_type='A',
                      dmrs_length=1, dmrs_add_pos=0, dmrs_type_a_pos=2, NSlot_list=[0]):
    
    dmrs = DMRSConfig(NumLayers=num_layers, 
                        PRBSet=num_prbs_k, 
                        MappingType=mapping_type, 
                        SymbolAllocation=symbol_allocation,
                        DMRSConfigurationType=config_type,
                        DMRSLength=dmrs_length,
                        DMRSAdditionalPosition=dmrs_add_pos,
                        DMRSTypeAPosition=dmrs_type_a_pos,
                        DMRSPortSet=[1000 + i for i in range(num_layers)],
                        NIDNSCID=10,
                        NSCID=0,
                        NSizeSymbol=num_prbs_l,
                        NSlot_list=NSlot_list)
    
    pilots, _ = dmrs.generate_dmrs_grid(pilot)
    dmrs.plot_dmrs(pilots)
    
    return pilots

if __name__ == "__main__":
    # Example
    dmrs_pilot="PDSCH"
    ports = 1
    configuration_type = 2
    symbol_allocation = (0, 14)
    num_prbs_k=2
    num_prbs_l=2
    mapping_type='A'
    pilot_length=2
    pilot_add_pos=1
    dmrs_type_a_pos=2

    Pilots = configure_dmrs(pilot=dmrs_pilot,
                            num_layers=ports, 
                            config_type=configuration_type, 
                            symbol_allocation=symbol_allocation,
                            num_prbs_k=num_prbs_k,
                            num_prbs_l=num_prbs_l,
                            mapping_type=mapping_type, 
                            dmrs_length=pilot_length, 
                            dmrs_add_pos=pilot_add_pos,
                            dmrs_type_a_pos=dmrs_type_a_pos,
                            NSlot_list=np.arange(num_prbs_l).tolist())