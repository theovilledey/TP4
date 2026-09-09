# module for the contour lines - here is usage example
#   levels = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
#   add_labeled_contours(ax, beta_plot, levels, Kx, Gy)

def add_labeled_contours(ax,grid,levels,X, Y, *,colors='black',linewidths=0.6,fontsize=6, fmt='%g'):
    """
    Draw labeled contours on ax.
    ax     : matplotlib axis
    grid   : the 2D array (same one given to pcolormesh)
    levels : list of contour levels - e.g. [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
    X, Y   : coordinate arrays matching pcolormesh(X, Y, grid)
    """
    cs = ax.contour(X, Y, grid, levels=levels,
                    colors=colors, linewidths=linewidths)
    label_positions = []
    for level_segs in cs.allsegs:
        # keep only non-empty segments for this level
        segs = [s for s in level_segs if len(s) > 0]
        if not segs:
            continue
        longest = max(segs, key=len)
        label_positions.append(longest[len(longest) // 2])
    if label_positions:
        ax.clabel(cs, fmt=fmt, fontsize=fontsize, inline=True, manual=label_positions)
    return cs

LITERATURE_EMITTERS = {
    "A": {
        "System 1": (3275, 45),
        "System 2": (320,  5.8),
        "System 3": (1544, 180),
        },
    "B": {
        "System 4": (12.8, 81),
        "System 5": (5.7,  72),
        "System 6": (0.57, 1.3),
        }}

# to add the systems or not?
def add_literature_points(ax, emitter, markers=("o", "s", "*")):
    for (name, (k, g)), marker in zip(LITERATURE_EMITTERS[emitter].items(), markers):
        ax.scatter(k, g, s=80, zorder=5, label=name, marker=marker, color="black")
    ax.legend(fontsize=9, loc="best")
