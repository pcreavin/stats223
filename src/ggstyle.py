"""Minimal matplotlib imitation of ggplot2 theme_bw + facet_grid, used to re-plot DH's figures."""

import matplotlib.pyplot as plt

GREY92, GREY20, GREY85 = "#EBEBEB", "#333333", "#D9D9D9"
# ggplot2 linetypes in units of line width
LINETYPES = {"solid": "-", "dashed": (0, (4, 4)), "longdash": (0, (7, 3)),
             "dotdash": (0, (1, 3, 4, 3)), "dotted": (0, (1, 3))}
PT_PER_MM = 72.27 / 25.4
LW_DEFAULT = 0.5 * PT_PER_MM * 0.75  # ggplot linewidth 0.5 in points (~1.07)


def hue_pal(n: int, h=(15, 375), c=100, l=65) -> list[str]:
    """scales::hue_pal()(n): evenly spaced HCL hues (CIE-LCh(uv)) converted to sRGB."""
    import math
    start, end = h
    if (end - start) % 360 < 1:
        end -= 360 / n
    hues = [start + i * (end - start) / max(n - 1, 1) for i in range(n)] if n > 1 else [start]
    out = []
    for hue in hues:
        hr = math.radians(hue % 360)
        L, U, V = l, c * math.cos(hr), c * math.sin(hr)
        # Luv -> XYZ (D65)
        un, vn = 0.19783000664283, 0.46831999493879
        Y = ((L + 16) / 116) ** 3 if L > 8 else L / 903.3
        up, vp = U / (13 * L) + un, V / (13 * L) + vn
        X = Y * 9 * up / (4 * vp)
        Z = Y * (12 - 3 * up - 20 * vp) / (4 * vp)
        rgb = [3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z,
               -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z,
               0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z]
        rgb = [12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055 for v in rgb]
        out.append("#" + "".join(f"{round(min(max(v, 0), 1) * 255):02X}" for v in rgb))
    return out


def theme_bw(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=GREY92, linewidth=0.8)
    ax.minorticks_on()
    ax.grid(True, which="minor", color=GREY92, linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(colors=GREY20, labelsize=8, length=2.5, color=GREY20)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GREY20)
        s.set_linewidth(0.6)


def strip(ax, text, side="top") -> None:
    kw = dict(transform=ax.transAxes, fontsize=8.5, color="#1A1A1A", ha="center", va="center",
              bbox=dict(boxstyle="square,pad=0.3", fc=GREY85, ec=GREY20, lw=0.6))
    if side == "top":
        ax.text(0.5, 1.0 + 0.06, text, **kw)
    else:
        ax.text(1.0 + 0.06, 0.5, text, rotation=-90, **kw)


def facet_grid(fig, gs, rows, cols, row_labels=None, col_labels=None, sharey=True):
    """Axes grid in a GridSpec cell, with ggplot strips (columns on top, rows on the right)."""
    sub = gs.subgridspec(len(rows), len(cols), wspace=0.08, hspace=0.12)
    axes = [[fig.add_subplot(sub[i, j]) for j in range(len(cols))] for i in range(len(rows))]
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i][j]
            theme_bw(ax)
            if i == 0 and col_labels is not None:
                strip(ax, col_labels[j], "top")
            if j == len(cols) - 1 and row_labels is not None:
                strip(ax, row_labels[i], "right")
            if i < len(rows) - 1:
                ax.tick_params(labelbottom=False)
            if j > 0 and sharey:
                ax.tick_params(labelleft=False)
    return axes


def bottom_legend(fig, handles, labels, title, y=0.01, ncol=None, x=0.5):
    leg = fig.legend(handles, labels, title=title, loc="lower center", bbox_to_anchor=(x, y),
                     ncol=ncol or len(labels), frameon=False, fontsize=8, title_fontsize=9,
                     handlelength=2.5, columnspacing=1.2)
    leg._legend_box.align = "left"
    # ggplot puts the title to the left of horizontal keys
    return leg
