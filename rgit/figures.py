"""Shared figure conventions for the manuscript.

Panel letters are bold lower-case letters without parentheses, placed at the
upper left of each panel outside the axes frame, aligned with the left edge of
the panel's tick and axis labels.
"""
from __future__ import annotations

import matplotlib


def label_panels(fig, axes, letters="abcdefghijklmnop", fontsize=None, pad_pt=3.0):
    """Write a bold letter at the upper left of every axes in ``axes``.

    Call after the layout is final (``tight_layout`` / ``constrained_layout``)
    and before ``savefig``. The letter's left edge is aligned with the left
    edge of the panel's tight bounding box (axis label and tick labels
    included) and its baseline sits ``pad_pt`` points above the axes frame.
    """
    if fontsize is None:
        fontsize = round(1.4 * matplotlib.rcParams["font.size"])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pad = pad_pt / 72.0 / fig.get_size_inches()[1]
    texts = []
    for ax, letter in zip(axes, letters):
        bb = ax.get_tightbbox(renderer)
        x0 = inv.transform((bb.x0, bb.y0))[0]
        y1 = ax.get_position().y1
        texts.append(fig.text(x0, y1 + pad, letter, fontsize=fontsize,
                              fontweight="bold", va="bottom", ha="left"))
    return texts
