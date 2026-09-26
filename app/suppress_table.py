"""观测计数交叉表的阈值隐藏：低于阈值的格必须隐藏，并以最少补充隐藏阻断合计反推。

直接公开逐格精确计数会泄露小型交叉单元（例如某保护站某物种的罕见观测）。
本模块把 2～4 行、2～4 列的正整数计数矩阵中所有低于阈值的格隐藏（必须
隐藏格），同时公开原始行合计、列合计与隐藏后的表格。但若只隐藏这些格，
结合行列合计往往可以逐一反推出被隐藏的值：某行（列）只剩一个隐藏格时，
该格即等于合计减去其余可见格之和。

因此需要在其余格中再选择若干补充隐藏格，使每个必须隐藏格都落在“全部
隐藏格构成的行×列二部图”的某个环上：环上的边无法被行列合计唯一确定
（沿环交替加减同一个未知量，得到另一组同样满足全部合计的解），从而
阻断上述反推。补充隐藏格的数量取所有可行方案中的最小值；数量相同时，
按补充格行优先（先按行、再按列）坐标序列的字典序取最小方案。

矩阵至多为 4×4，因此直接按“数量从小到大、同数量按行优先字典序”枚举
补充格组合并逐一检验，第一个可行组合即为所求。检验方式：对每条必须
隐藏边，在删去该边后的二部图中判断其行、列端点是否仍连通（连通即该
边在某个环上）。矩阵至少为 2×2，全部格都隐藏时二部图为完全二部图，
每条边都在环上，因此枚举必然在穷尽前找到可行解。
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .anonymizer import unprocessable

# 矩阵行数与列数的范围。
MIN_DIMENSION = 2
MAX_DIMENSION = 4


def prepare_suppression_counts(raw_counts: Any) -> list[list[int]]:
    """校验计数矩阵，返回正整数矩阵（2～4 行、2～4 列的矩形）。

    每格必须是正整数（拒绝布尔、浮点、字符串与 0）；所有行必须等长。
    """
    if not isinstance(raw_counts, list):
        raise unprocessable("counts 必须是数组")
    if len(raw_counts) < MIN_DIMENSION:
        raise unprocessable(f"counts 至少需要包含 {MIN_DIMENSION} 行")
    if len(raw_counts) > MAX_DIMENSION:
        raise unprocessable(f"counts 最多包含 {MAX_DIMENSION} 行")

    matrix: list[list[int]] = []
    width: int | None = None
    for index, raw_row in enumerate(raw_counts):
        label = f"counts 第 {index + 1} 行"
        if not isinstance(raw_row, list):
            raise unprocessable(f"{label}必须是数组")
        if len(raw_row) < MIN_DIMENSION:
            raise unprocessable(f"{label}至少需要包含 {MIN_DIMENSION} 列")
        if len(raw_row) > MAX_DIMENSION:
            raise unprocessable(f"{label}最多包含 {MAX_DIMENSION} 列")
        if width is None:
            width = len(raw_row)
        elif len(raw_row) != width:
            raise unprocessable(
                f"{label}的列数 {len(raw_row)} 与首行的 {width} 列不一致"
            )
        row: list[int] = []
        for value in raw_row:
            if isinstance(value, bool) or not isinstance(value, int):
                raise unprocessable(f"{label}的每格计数必须是整数")
            if value < 1:
                raise unprocessable(f"{label}的每格计数必须是正整数（>= 1）")
            row.append(value)
        matrix.append(row)
    return matrix


def solve_suppression(
    counts: list[list[int]], threshold: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """返回 (必须隐藏格, 补充隐藏格)，均为 (行, 列) 零起始坐标、按行优先排序。

    必须隐藏格为所有计数低于阈值的格；补充隐藏格是在其余格中选出的、
    使每个必须隐藏格都落在全部隐藏格构成的行列二部图的环上的最小集合
    （同数量时按行优先坐标序列字典序最小）。
    """
    rows = len(counts)
    cols = len(counts[0])
    mandatory = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if counts[i][j] < threshold
    ]
    if not mandatory:
        return [], []

    candidates = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if counts[i][j] >= threshold
    ]
    # 候选格按行优先排列，combinations 依此顺序生成的组合本身即按
    # 行优先坐标序列的字典序枚举；按数量从小到大遍历，第一个可行组合
    # 就是“数量最少、同数量字典序最小”的方案。
    for size in range(len(candidates) + 1):
        for subset in combinations(candidates, size):
            if _all_mandatory_on_cycles(mandatory, subset, rows, cols):
                return mandatory, list(subset)
    # 矩阵至少 2×2，全部格都隐藏时是完全二部图，每条边都在环上，
    # 因此上面的枚举必然已经返回。
    raise AssertionError("不可达：全部格都隐藏时必然可行")


def _all_mandatory_on_cycles(
    mandatory: list[tuple[int, int]],
    supplementary: tuple[tuple[int, int], ...],
    rows: int,
    cols: int,
) -> bool:
    """判断每个必须隐藏格是否都在全部隐藏格构成的二部图的某个环上。

    二部图的行顶点编号 0..rows-1，列顶点编号 rows..rows+cols-1，每个
    隐藏格是一条行-列边。一条边在某个环上，当且仅当删去该边后其两个
    端点仍然连通。
    """
    adjacency: list[list[int]] = [[] for _ in range(rows + cols)]
    for i, j in mandatory:
        adjacency[i].append(rows + j)
        adjacency[rows + j].append(i)
    for i, j in supplementary:
        adjacency[i].append(rows + j)
        adjacency[rows + j].append(i)
    return all(
        _connected_without_edge(adjacency, i, rows + j) for i, j in mandatory
    )


def _connected_without_edge(
    adjacency: list[list[int]], source: int, target: int
) -> bool:
    """在禁止使用 source-target 这条边的前提下，判断两点是否连通（DFS）。"""
    visited = {source}
    stack = [source]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if (node == source and neighbor == target) or (
                node == target and neighbor == source
            ):
                continue
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return target in visited
