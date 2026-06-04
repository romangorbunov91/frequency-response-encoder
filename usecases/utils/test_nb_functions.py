import numpy as np
from pathlib import Path
from typing import Union, Any, Dict, Optional, List, Sequence
from matplotlib import pyplot as plt

def rmse_complex(
    response_in: np.ndarray,
    response_ref: np.ndarray,
    Nlim: Optional[Sequence[Optional[int]]] = None,
    eps: float = 1e-12
) -> float:

    if response_ref.shape[0] != 3 or response_in.shape[0] != 3:
        raise ValueError("Both inputs must be of shape (3, N)!!!")
        
    if Nlim is None:
        Nlim = [None, None]
    Nslice = slice(Nlim[0], Nlim[1])

    freq   = response_in[0, Nslice]
    mag_db = response_in[1, Nslice]
    ph_deg = response_in[2, Nslice]
    
    freq_ref   = response_ref[0, Nslice]
    mag_db_ref = response_ref[1, Nslice]
    ph_deg_ref = response_ref[2, Nslice]

    mag_abs     = 10**(mag_db     / 20)
    mag_abs_ref = 10**(mag_db_ref / 20)
    
    complex_in  = mag_abs     * np.exp(1j*np.deg2rad(ph_deg    ))
    complex_ref = mag_abs_ref * np.exp(1j*np.deg2rad(ph_deg_ref))

    # Interpolate real & imag separately to avoid phase-wrapping artifacts.
    complex_interp = np.interp(freq_ref, freq, complex_in.real) + 1j * np.interp(freq_ref, freq, complex_in.imag)

    RMSE = np.sqrt(np.mean(np.abs(complex_ref - complex_interp)**2))

    # Relative and Normalized RMSE.
    RRMSE = RMSE / (np.sqrt(np.mean(mag_abs_ref**2)) + eps)
    NRMSE = RMSE / (np.max(mag_abs_ref) - np.min(mag_abs_ref) + eps)
    return float(RMSE), float(RRMSE), float(NRMSE)


def find_multipliers(fz: float, fd: float, err_rel_ref: float, Nmin: int, Nmax: int):
    R = fd / fz
    
    if R < 2:
        raise ValueError("fz must be < fd/2 for a valid solution.")
    
    err_abs_ref = fz * err_rel_ref
    Mmax = int(R / 2)
    N_vals = np.arange(Nmin, Nmax + 1, dtype=float)
    
    N_arr = np.empty(Mmax, dtype=np.int64)
    M_arr = np.empty(Mmax, dtype=np.int64)
    L_arr = np.empty(Mmax, dtype=np.int64)
    fz_approx_arr = np.empty(Mmax)
    err_abs_arr = np.empty(Mmax)
    err_rel_arr = np.empty(Mmax)
    tol_flag_arr = np.empty(Mmax, dtype=bool)
    
    for mdx, M in enumerate(range(1, Mmax+1)):
        L = np.floor(M * N_vals / R)
        # In-place clamp.
        np.maximum(L, 1, out=L)
        
        R_approx = M * N_vals / L
        fz_approx = fd / R_approx
        err_abs = np.abs(fz - fz_approx)
        
        valid_idx = np.nonzero(err_abs < err_abs_ref)[0]
        
        if valid_idx.size > 0:
            pos = valid_idx[0]
            tol_flag_arr[mdx] = True
        else:
            pos = np.argmin(err_abs)
            tol_flag_arr[mdx] = False

        N_arr[mdx] = int(N_vals[pos])
        M_arr[mdx] = M
        L_arr[mdx] = int(L[pos])
        fz_approx_arr[mdx] = fz_approx[pos]
        err_abs_arr[mdx] = err_abs[pos]
        err_rel_arr[mdx] = err_abs[pos] / fz_approx[pos]
    
    if np.any(tol_flag_arr):
        valid_mdx = np.nonzero(tol_flag_arr)[0]
        best_mdx = valid_mdx[np.argmin(N_arr[valid_mdx])]
    else:
        best_mdx = np.argmin(err_abs_arr)
        
    return (int(N_arr[best_mdx]),
            int(M_arr[best_mdx]),
            int(L_arr[best_mdx])), \
           (float(fz_approx_arr[best_mdx]),
            float(err_abs_arr[best_mdx]),
            float(err_rel_arr[best_mdx]),
            bool(tol_flag_arr[best_mdx]))


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
                            color=m_cfg['color'], alpha=0.25, label=m_cfg['label']+'_mask')

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
                ax.set_title(title, fontsize=plot_config['fontsize'])
            
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
    legend: List[str],
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
        
        for mdx, data_map in enumerate(data_map_list):
            # Plot Main Data.
            x_data = data_map[cfg['arg_key']]
            y_data = data_map[cfg['data_key']]
            
            ax.plot(
                x_data, y_data,
                '.', markersize=plot_config['markersize_data'],
                linestyle=plot_config['line_styles'][mdx],
                alpha=plot_config['line_alphas'][mdx],
                label=legend[mdx])
            
        # Set title (on the first plot of the sample group only).
        if title is not None:
            if idx == 0:
                ax.set_title(title, fontsize=plot_config['fontsize'])
            
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


def transfer_function(
    freq: List[float],
    zero_poles: int,
    poles: List[float],
    zeros: List[float],
    gain: Optional[float] = 1.0,
    delay: Optional[float] = 0.0
    ) -> List[float]:
    
    omega = 2*np.pi*freq
    
    gain_complex = gain / (1j * omega)**zero_poles * np.exp(-1j * omega * delay)
        
    for zero in zeros:
        gain_complex *= 1.0 + 1j*freq/zero
    
    for pole in poles:
        gain_complex /= 1.0 + 1j*freq/pole
    
    return gain_complex


def mask_postprocess(
    mask: np.ndarray,
    keys: List[str],
    M: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters sequences of '1s' to keep only those with length >= M.
    
    Returns:
        - Middle coordinates of the valid sequences (shape: (N, 2)).
        - Filtered mask (same shape as input, dtype float32).
        - Number of regions before filtering per channel (shape: (num_channels,)).
        - Number of regions after filtering per channel (shape: (num_channels,)).
    """
    arr = np.array(mask, dtype=int)
    positions_dict = {}
    
    # Initialize the filtered mask with zeros of the same shape.
    mask_filtered = np.zeros_like(arr)
    
    # Lists to store counts per channel (row)
    regions_before_per_channel = []
    regions_after_per_channel = []
    
    for row_idx, key in enumerate(keys):
        row = arr[row_idx]
        positions = []
        
        # Pad with 0 at both ends to correctly catch sequences
        # that start at index 0 or end at the last index.
        padded = np.pad(row, (1, 1), constant_values=0)
        
        # Converts to 1 or -1.
        diff = np.diff(padded)
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Count regions before filtering for this specific channel
        regions_before_per_channel.append(len(starts))
        
        num_after = 0
        for start, end in zip(starts, ends):
            length = end - start
            if length >= M:
                mid_col = (start + end - 1) // 2
                positions.append(mid_col)
                
                # Keep the valid sequence in the filtered mask.
                mask_filtered[row_idx, start:end] = 1
                num_after += 1
                
        # Count regions after filtering for this specific channel.
        regions_after_per_channel.append(num_after)
        
        positions_dict[key] = np.array(positions, dtype=int)
                
    # Convert the per-channel lists to numpy arrays
    regions_before_arr = np.array(regions_before_per_channel, dtype=int)
    regions_after_arr = np.array(regions_after_per_channel, dtype=int)

        
    return positions_dict, np.array(mask_filtered, dtype=np.float32), regions_before_arr, regions_after_arr