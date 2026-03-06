import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def plot_convergence(N_vals, err_stat_list, title=None, ref_list=None, filename=None):
    N_vals = np.array(N_vals)

    fig, ax = plt.subplots(figsize=(9, 8))

    for stat in err_stat_list:
        mean = stat['mean']
        std = stat['std']

        line, = plt.loglog(N_vals, mean, marker='o', **stat['plot_kwargs'])
        
        color = line.get_color()
        
        plt.fill_between(
            N_vals, 
            mean - std, 
            mean + std, 
            color=color, 
            alpha=0.2,
            edgecolor=None
        )

    ax.set_xscale('log')
    ax.set_yscale('log')

    def plot_ref_line(anchor, slope, label):
        N0, E0 = anchor
        C = E0 / (N0 ** slope)
        ref_line = C * (N_vals ** slope)

        ax.plot(N_vals, ref_line, 'k--', linewidth=0.5,)

        ax.annotate(
            label, 
            xy=anchor,
            xytext=(50, -5),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=16,
        )

    if ref_list is not None:
        for ref in ref_list:
            plot_ref_line(*ref)

    ax.set_xticks(N_vals)
    ax.set_xticklabels(N_vals)
    ax.minorticks_off()

    ax.legend()
    ax.grid(True)

    if title is not None:
        plt.title(title)

    if filename is not None:
        fig.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')

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
        xytext=(0, -20),
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

    plt.title(r"IE (fixed) K=50 N=6400 with QP")

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_point_cloud(points, id_list, filename=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for ids in id_list:
        ax.scatter(points[ids, 0], points[ids, 1], points[ids, 2], s=10)
    
    ax.set_aspect('equal')
    
    if filename is not None:
            fig.savefig(f'./static/{filename}.pdf', dpi=300, bbox_inches='tight')

    plt.show()