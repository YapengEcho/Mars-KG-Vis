"""在后端用 Matplotlib 把编目统计画成 PNG，供网站统计页显示。"""

from collections import Counter
from functools import lru_cache
from io import BytesIO
from threading import Lock

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from .catalog_query import filter_records
from .schemas import Catalog, Slide, SlideType


TYPE_COLORS: dict[SlideType, str] = {
    "SL": "#ec9c79",
    "SP": "#78bed4",
    "FL": "#b9a2e8",
    "CL": "#e1c574",
}
LATITUDE_EDGES = ("90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N")
_SURFACE = "#1a1713"
_LINE = "#544a3e"
_LABEL = "#c6beb3"
_TEXT = "#f2ede5"
_DRAW_LOCK = Lock()


def count_latitude_bands(records: list[Slide]) -> tuple[int, ...]:
    counts = [0] * (len(LATITUDE_EDGES) - 1)
    for record in records:
        # 纬度加 90 后每 30° 分一组。北极也计入最北的一组。
        band = min(len(counts) - 1, int((record.latitude + 90) / 30))
        counts[band] += 1
    return tuple(counts)


def _png(figure: Figure) -> bytes:
    """图直接写到内存，接口拿到的是 PNG 字节。"""
    output = BytesIO()
    FigureCanvasAgg(figure).print_png(output)
    return output.getvalue()


@lru_cache(maxsize=20)
def draw_type_chart(counts: tuple[int, ...], selected: SlideType | None) -> bytes:
    """保留全目录的类型占比，点选类别时压暗其余扇区。"""
    with _DRAW_LOCK:
        figure = Figure(figsize=(2.6, 2.6), dpi=150, facecolor="none")
        axes = figure.add_axes((0, 0, 1, 1))
        background_red, background_green, background_blue = to_rgb(_SURFACE)
        colors: list[str | tuple[float, float, float]] = []
        for kind, color in TYPE_COLORS.items():
            if selected and kind != selected:
                red, green, blue = to_rgb(color)
                colors.append((
                    red * 0.35 + background_red * 0.65,
                    green * 0.35 + background_green * 0.65,
                    blue * 0.35 + background_blue * 0.65,
                ))
            else:
                colors.append(color)
        axes.pie(
            counts,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.33, "linewidth": 0},
        )
        axes.set(xlim=(-1, 1), ylim=(-1, 1), frame_on=False)
        return _png(figure)


@lru_cache(maxsize=20)
def draw_latitude_chart(counts: tuple[int, ...], selected: SlideType | None) -> bytes:
    """每个纬度带画一根柱子，柱顶标出记录数。"""
    with _DRAW_LOCK:
        figure = Figure(figsize=(5.0, 2.65), dpi=160, facecolor="none")
        axes = figure.add_axes((0.12, 0.18, 0.86, 0.78), facecolor="none")
        positions = range(len(counts))
        labels = [
            f"{start}\n–{end}" for start, end in zip(LATITUDE_EDGES, LATITUDE_EDGES[1:])
        ]
        color = TYPE_COLORS[selected] if selected else "#dfa477"
        bars = axes.bar(positions, counts, width=0.55, color=color)
        axes.bar_label(
            bars, labels=[f"{count:,}" for count in counts], color=_TEXT, padding=4, fontsize=10.5
        )
        axes.set(
            xlim=(-0.55, len(counts) - 0.45),
            ylim=(0, max(4, max(counts) * 1.23)),
            xticks=positions,
            xticklabels=labels,
        )
        axes.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        axes.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axes.set_axisbelow(True)
        axes.grid(axis="y", color=_LINE, linestyle="--", linewidth=0.6)
        axes.tick_params(axis="both", length=0, labelcolor=_LABEL, labelsize=10, pad=5)
        axes.spines[["top", "right", "left"]].set_visible(False)
        axes.spines["bottom"].set_color(_LINE)
        return _png(figure)


def render_catalog_chart(catalog: Catalog, kind: str, selected: SlideType | None) -> bytes:
    if kind == "types":
        totals = Counter(record.type for record in catalog.records)
        return draw_type_chart(tuple(totals[code] for code in TYPE_COLORS), selected)
    # 点选类型后，柱形图只数这一类滑坡。
    matching_records = filter_records(catalog, slide_type=selected)
    return draw_latitude_chart(count_latitude_bands(matching_records), selected)
