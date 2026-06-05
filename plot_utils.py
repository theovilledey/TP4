# module for the contour lines
def add_labeled_contours(ax, grid, levels, extent,*,colors='black', linewidths=0.6, fontsize=6, fmt='%g'):
    """
    Draw contours of ax
    ax : write ax, matplotlib axis
    grid : the 2D array
    levels : list of contour levels - for beta I used [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
    extent : [xmin, xmax, ymin, ymax] - just match the imshow - extent=[-2, 6, -2, 6]
    """
    cs = ax.contour(grid, levels=levels, origin='lower',extent=extent, colors=colors, linewidths=linewidths)
    label_positions = []
    for level_segs in cs.allsegs:
        if not level_segs:
            continue
        longest = max(level_segs, key=len)
        label_positions.append(longest[len(longest) // 2])
    ax.clabel(cs, fmt=fmt, fontsize=fontsize, inline=True,manual=label_positions)
    return cs