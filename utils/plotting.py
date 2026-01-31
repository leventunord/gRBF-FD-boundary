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
        ref_anchor = (1600, 1e-2)

    N0, E0 = ref_anchor

    slope = -2.0 
    C = E0 / (N0 ** slope)
    
    ref_line_y = C * (N_vals ** slope)
    
    ax.plot(N_vals, ref_line_y, 'k--', linewidth=0.5,)

    ax.annotate(
        r'$O(N^{-2})$', 
        xy=ref_anchor,
        xytext=(50, -10),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=10,
    )

    ax.set_xticks(N_vals)
    ax.set_xticklabels(N_vals)
    ax.minorticks_off()

    ax.legend()

    plt.title(r"$\partial u/\partial n$ FE (fixed) K=50 with QP")

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_distribution(params, error_vals, pos_ids=[], unstable_ids=[], filename='temp'):
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

    if len(pos_ids) > 0:
        ax.scatter(
            params[pos_ids, 0], params[pos_ids, 1],
            s=100,
            facecolors='none',
            edgecolors='red',
            linewidth=2,
            label=r'$w_0$ >= 0'
        )

    if len(unstable_ids) > 0:
        ax.scatter(
            params[unstable_ids, 0], params[unstable_ids, 1],
            s=100,
            facecolors='none',
            edgecolors='magenta',
            linewidth=2,
            label='ratio <= 1'
        )

    # find the point with maximum error
    # max_id = np.argmax(error_vals)
    # ax.scatter(params[max_id][0], params[max_id][1], s=100, facecolors='none', edgecolors='red', linewidth=2)

    # max_err_text = f"{error_vals[max_id]:.3e}"

    # ax.annotate(
    #     max_err_text, 
    #     xy=(params[max_id][0], params[max_id][1]),
    #     xytext=(0, -20),
    #     textcoords='offset points',
    #     ha='center',
    #     va='bottom',
    #     color='red',
    #     fontsize=10,
    #     fontweight='bold'
    # )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Absolute Forward Error', rotation=270, labelpad=20)

    ax.set_xlim(0.0, 2*np.pi)
    ax.set_ylim(np.pi/9, 8*np.pi/9)

    ax.set_yticks([np.pi/9, np.pi/2, 8*np.pi/9])
    ax.set_yticklabels([r'$\pi/9$', r'$\pi/2$', r'$8\pi/9$'])
    ax.set_xticks([0, np.pi/2, np.pi, 3 * np.pi/2, 2*np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\phi$', rotation=0)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_aspect('equal')

    ax.minorticks_off()

    ax.legend(
        loc='lower right',        
        bbox_to_anchor=(1, -0.3), 
        borderaxespad=0.,           
        frameon=True,
        fontsize=10
    )

    plt.title(r"Laplacian FE K (fixed)=50 N=6400 with QP")

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()