import numpy as np
from pathlib import Path
from typing import Union, Any, Dict, Optional, List
from matplotlib import pyplot as plt

def zeros_poles_freq_to_positions(
        zeros_poles_freq_dict: Dict[str, List[float]],
        freq: np.ndarray
        ) -> Dict[str, np.ndarray]:
    
    freq_ravel = np.asarray(freq).ravel()
    positions_dict = {}
    for key, values in zeros_poles_freq_dict.items():
        targets = np.asarray(values).ravel()
        
        if targets.size == 0:
            positions_dict[key] = np.array([], dtype=int)
        else:
            # Compute absolute differences and find the closest index in `freq`.
            diff_matrix = np.abs(freq_ravel[:, np.newaxis] - targets)
            positions_dict[key] = np.argmin(diff_matrix, axis=0)

    return positions_dict

def plot_responses(
    plot_config: Dict[str, Any],
    data: np.ndarray,
    title: Optional[str] = None,
    regions: Optional[np.ndarray] = None,
    zeros_poles_positions: Optional[List[int]] = None,
    save_path: Optional[Union[str, Path]] = None
    ) -> None:

    x_samples = np.arange(data.shape[-1])

    data_map = {
        'samples': x_samples,
        'freq': data[0,:],
        'mag' : data[1,:],
        'ph'  : data[2,:]
    }

    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(plot_config['fig_width'],
                 plot_config['fig_height'])
        )
    axs = np.array(axs).flatten()
    for idx, cfg in enumerate(plot_config['plots']):
        ax = axs[idx]
        
        # Plot Main Data.
        x_data = data_map[cfg['arg_key']]
        y_data = data_map[cfg['data_key']]
        
        ax.plot(
            x_data, y_data,
            '.', markersize=plot_config['markersize_data'],
            linestyle='-',
            alpha=0.7)
        
        has_mask = False
        
        # Highlight regions.
        if regions is not None:
            for m_idx, m_cfg in enumerate(plot_config['masks']):
                indices = np.where(regions[m_idx] == 1)[0]
                if len(indices) > 0:
                    has_mask = True
                    ymin, ymax = ax.get_ylim()
                    ax.fill_between(x_data, ymin, ymax, where=regions[m_idx].astype(bool), 
                            color=m_cfg['color'], alpha=0.25, label=m_cfg['label'])

        # Mark masks.
        if zeros_poles_positions is not None:
            for m_idx, m_cfg in enumerate(plot_config['masks']):
                indices = zeros_poles_positions[m_cfg['label']]
                if len(indices) > 0:
                    has_mask = True
                    ax.plot(x_data[indices], y_data[indices], 
                            marker=m_cfg['marker'], markersize=plot_config['markersize_mask'], 
                            linestyle='', color=m_cfg['color'], 
                            label=m_cfg['label']+'_true')
        
        # Set title (on the first plot of the sample group only).
        if title is not None:
            if idx == 0:
                ax.set_title(title, fontsize=plot_config['fontsize'], fontweight='bold')
            
        ax.set_xscale(cfg['xscale'])

        ax.set_xlabel(
            cfg['xlabel'],
            fontsize=plot_config['fontsize'],
            fontdict=plot_config['label_font']
            )
        ax.set_ylabel(
            cfg['ylabel'],
            fontsize=plot_config['fontsize'],
            fontdict=plot_config['label_font']
            )
        ax.grid(True, alpha=plot_config['grid_alpha'], axis='both', linestyle='--')
        
        # Only show legend if masks exist.
        if has_mask:
            ax.legend(
                fontsize=plot_config['fontsize_legend'],
                loc=plot_config['legend_loc'],
                framealpha=plot_config['legend_framealpha']
                )

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)

def plot_multiple_responses(
    plot_config: Dict[str, Any],
    data_list: List[np.ndarray],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None
    ) -> None:

    data_map_list = []
    for data in data_list:
        data_map_list.append({
            'samples': np.arange(data.shape[-1]),
            'freq': data[0,:],
            'mag' : data[1,:],
            'ph'  : data[2,:]
        })

    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(plot_config['fig_width'],
                 plot_config['fig_height'])
        )
    axs = np.array(axs).flatten()
    for idx, cfg in enumerate(plot_config['plots']):
        ax = axs[idx]
        
        for data_map in data_map_list:
            # Plot Main Data.
            x_data = data_map[cfg['arg_key']]
            y_data = data_map[cfg['data_key']]
            
            ax.plot(
                x_data, y_data,
                '.', markersize=plot_config['markersize_data'],
                linestyle='-',
                alpha=0.7)
            
        # Set title (on the first plot of the sample group only).
        if title is not None:
            if idx == 0:
                ax.set_title(title, fontsize=plot_config['fontsize'], fontweight='bold')
            
        ax.set_xscale(cfg['xscale'])

        ax.set_xlabel(
            cfg['xlabel'],
            fontsize=plot_config['fontsize'],
            fontdict=plot_config['label_font']
            )
        ax.set_ylabel(
            cfg['ylabel'],
            fontsize=plot_config['fontsize'],
            fontdict=plot_config['label_font']
            )
        ax.grid(True, alpha=plot_config['grid_alpha'], axis='both', linestyle='--')
        '''
        ax.legend(
            fontsize=plot_config['fontsize_legend'],
            loc=plot_config['legend_loc'],
            framealpha=plot_config['legend_framealpha']
            )
        '''
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)