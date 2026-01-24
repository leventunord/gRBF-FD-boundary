import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def plot_convergence(N_vals, err_stat_list, filename='temp', ref_anchor=None):
    fig, ax = plt.subplots(figsize=(9, 6))

    for stat in err_stat_list:
        ax.errorbar(N_vals, stat['mean'], yerr=stat['std'], 
                    capsize=4,
                    **stat['plot_kwargs'])

    ax.set_xscale('log')
    ax.set_yscale('log')

    # convergence ref
    if ref_anchor is  None:
        ref_anchor = (1600, 1e-3)

    N0, E0 = ref_anchor

    slope = -2.0 
    C = E0 / (N0 ** slope)
    
    ref_line_y = C * (N_vals ** slope)
    
    ax.plot(N_vals, ref_line_y, 'k--', linewidth=0.5,)

    ax.annotate(
        r'$O(N^{-2})$', 
        xy=ref_anchor,
        xytext=(50, -40),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=10,
    )

    ax.set_xticks(N_vals)
    ax.set_xticklabels(N_vals)
    ax.minorticks_off()

    ax.legend()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_distribution(params, error_vals, filename='temp'):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sc = ax.scatter(
        params[:, 0], params[:, 1], c=error_vals,
        s=100,
        cmap='viridis', 
        norm=LogNorm(),
        marker='o',
        edgecolors='none',
        alpha=0.8
    )

    # find the point with maximum error
    max_id = np.argmax(error_vals)
    ax.scatter(params[max_id][0], params[max_id][1], s=100, facecolors='none', edgecolors='red', linewidth=2)

    max_err_text = f"{error_vals[max_id]:.3e}"

    ax.annotate(
        max_err_text, 
        xy=(params[max_id][0], params[max_id][1]),
        xytext=(0, 10),
        textcoords='offset points',
        ha='center',
        va='bottom',
        color='red',
        fontsize=10,
        fontweight='bold'
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Absolute Inverse Error', rotation=270, labelpad=20)

    ax.set_xlim(0.0, 2*np.pi)
    ax.set_ylim(0.0, np.pi)

    ax.set_yticks([0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_xticks([0, np.pi/2, np.pi, 3 * np.pi/2, 2*np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\phi$', rotation=0)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_aspect('equal')

    ax.minorticks_off()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()