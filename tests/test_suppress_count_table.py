"""观测计数交叉表阈值隐藏接口 POST /api/v1/suppress-count-table 的端到端测试。"""

from __future__ import annotations

import random
from itertools import combinations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ENDPOINT = "/api/v1/suppress-count-table"

# 基础样例：3×3 矩阵，threshold=5，(1,0)=3 与 (2,2)=4 必须隐藏。
COUNTS = [[12, 7, 5], [3, 9, 8], [6, 11, 4]]

# payload() 的 counts 缺省哨兵（None 本身是合法待测输入）。
_DEFAULT = object()


def payload(counts=_DEFAULT, threshold: int = 5, **extra) -> dict:
    body = {
        "counts": COUNTS if counts is _DEFAULT else counts,
        "threshold": threshold,
    }
    body.update(extra)
    return body


def mandatory_of(counts: list[list[int]], threshold: int) -> list[tuple[int, int]]:
    rows = len(counts)
    cols = len(counts[0])
    return [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if counts[i][j] < threshold
    ]


def on_cycle(
    edge: tuple[int, int], hidden: set[tuple[int, int]], rows: int, cols: int
) -> bool:
    """并查集判断 edge 的行、列端点在删去该边后是否连通（独立参考实现）。"""
    parent = list(range(rows + cols))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for cell in hidden:
        if cell == edge:
            continue
        root_a, root_b = find(cell[0]), find(rows + cell[1])
        if root_a != root_b:
            parent[root_a] = root_b
    return find(edge[0]) == find(rows + edge[1])


def reference_supplementary(
    counts: list[list[int]], threshold: int
) -> list[tuple[int, int]]:
    """暴力枚举（数量升序、同数量按行优先字典序）求补充隐藏格的参考解。"""
    rows = len(counts)
    cols = len(counts[0])
    mandatory = mandatory_of(counts, threshold)
    candidates = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if counts[i][j] >= threshold
    ]
    for size in range(len(candidates) + 1):
        for subset in combinations(candidates, size):
            hidden = set(mandatory) | set(subset)
            if all(on_cycle(m, hidden, rows, cols) for m in mandatory):
                return list(subset)
    raise AssertionError("全部格都隐藏时必然可行")


def assert_response_valid(body: dict, counts: list[list[int]], threshold: int):
    """校验响应结构：表格、合计、两类坐标相互一致，且满足环性质与最优性。"""
    rows = len(counts)
    cols = len(counts[0])
    assert set(body) == {
        "suppressed_counts",
        "row_totals",
        "column_totals",
        "threshold",
        "mandatory_cells",
        "supplementary_cells",
    }
    assert body["threshold"] == threshold
    assert body["row_totals"] == [sum(row) for row in counts]
    assert body["column_totals"] == [
        sum(counts[i][j] for i in range(rows)) for j in range(cols)
    ]

    mandatory = [(c["row"], c["column"]) for c in body["mandatory_cells"]]
    supplementary = [
        (c["row"], c["column"]) for c in body["supplementary_cells"]
    ]
    # 必须隐藏格恰为所有低于阈值的格；两类坐标均按行优先排序且不重叠。
    assert mandatory == mandatory_of(counts, threshold)
    assert supplementary == sorted(supplementary)
    assert not set(mandatory) & set(supplementary)

    # 发布表格：隐藏格恰为 null，可见格与原始值一致。
    hidden = set(mandatory) | set(supplementary)
    table = body["suppressed_counts"]
    assert len(table) == rows
    for i in range(rows):
        assert len(table[i]) == cols
        for j in range(cols):
            if (i, j) in hidden:
                assert table[i][j] is None
            else:
                assert table[i][j] == counts[i][j]

    # 每个必须隐藏格都在全部隐藏格构成的行列二部图的环上。
    for m in mandatory:
        assert on_cycle(m, hidden, rows, cols)

    # 补充格数量最少，且同数量时为行优先坐标序列字典序最小方案。
    assert supplementary == reference_supplementary(counts, threshold)


# ---------------------------------------------------------------- 基本功能

def test_basic_suppression():
    response = client.post(ENDPOINT, json=payload())
    assert response.status_code == 200
    body = response.json()

    # (1,0) 与 (2,2) 必须隐藏；唯一的 4 环由补充格 (1,2)、(2,0) 补全。
    assert body["suppressed_counts"] == [
        [12, 7, 5],
        [None, 9, None],
        [None, 11, None],
    ]
    assert body["row_totals"] == [24, 20, 21]
    assert body["column_totals"] == [21, 27, 17]
    assert body["threshold"] == 5
    assert body["mandatory_cells"] == [
        {"row": 1, "column": 0},
        {"row": 2, "column": 2},
    ]
    assert body["supplementary_cells"] == [
        {"row": 1, "column": 2},
        {"row": 2, "column": 0},
    ]
    assert_response_valid(body, COUNTS, 5)


def test_no_cell_below_threshold_hides_nothing():
    # 计数等于阈值时不算“低于阈值”，无需隐藏。
    response = client.post(
        ENDPOINT, json=payload(counts=[[5, 6], [7, 8]], threshold=5)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suppressed_counts"] == [[5, 6], [7, 8]]
    assert body["mandatory_cells"] == []
    assert body["supplementary_cells"] == []
    assert body["row_totals"] == [11, 15]
    assert body["column_totals"] == [12, 14]


def test_threshold_one_hides_nothing():
    # 所有格都是正整数，threshold=1 时没有低于阈值的格。
    response = client.post(
        ENDPOINT, json=payload(counts=[[1, 2], [3, 4]], threshold=1)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suppressed_counts"] == [[1, 2], [3, 4]]
    assert body["mandatory_cells"] == []
    assert body["supplementary_cells"] == []


def test_all_cells_below_threshold_need_no_supplement():
    # 全部格必须隐藏：完全二部图的每条边本就在环上，无需补充。
    response = client.post(
        ENDPOINT, json=payload(counts=[[1, 2], [3, 4]], threshold=5)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suppressed_counts"] == [[None, None], [None, None]]
    assert body["mandatory_cells"] == [
        {"row": 0, "column": 0},
        {"row": 0, "column": 1},
        {"row": 1, "column": 0},
        {"row": 1, "column": 1},
    ]
    assert body["supplementary_cells"] == []
    assert body["row_totals"] == [3, 7]
    assert body["column_totals"] == [4, 6]


def test_single_low_cell_in_2x2_hides_everything():
    # 2×2 中唯一的隐藏格必然可被行列合计反推，只能全部隐藏。
    response = client.post(
        ENDPOINT, json=payload(counts=[[12, 7], [3, 9]], threshold=5)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mandatory_cells"] == [{"row": 1, "column": 0}]
    assert body["supplementary_cells"] == [
        {"row": 0, "column": 0},
        {"row": 0, "column": 1},
        {"row": 1, "column": 1},
    ]
    assert body["suppressed_counts"] == [[None, None], [None, None]]


def test_single_low_cell_supplements_lexicographically_smallest_block():
    # 3×3 中只有 (0,0) 低于阈值：最小补充为 3 格，可行方案是任一含 (0,0)
    # 的 2×2 角块的其余三格，行优先字典序最小为 (0,1),(1,0),(1,1)。
    counts = [[1, 5, 5], [5, 5, 5], [5, 5, 5]]
    response = client.post(ENDPOINT, json=payload(counts=counts, threshold=5))
    assert response.status_code == 200
    body = response.json()
    assert body["mandatory_cells"] == [{"row": 0, "column": 0}]
    assert body["supplementary_cells"] == [
        {"row": 0, "column": 1},
        {"row": 1, "column": 0},
        {"row": 1, "column": 1},
    ]
    assert_response_valid(body, counts, 5)


def test_two_low_cells_sharing_a_row():
    # 同一行的两个必须隐藏格可由同一 4 环覆盖，补充格为对应列的另一行两格。
    counts = [[1, 1, 9], [9, 9, 9]]
    response = client.post(ENDPOINT, json=payload(counts=counts, threshold=5))
    assert response.status_code == 200
    body = response.json()
    assert body["mandatory_cells"] == [
        {"row": 0, "column": 0},
        {"row": 0, "column": 1},
    ]
    assert body["supplementary_cells"] == [
        {"row": 1, "column": 0},
        {"row": 1, "column": 1},
    ]
    assert body["suppressed_counts"] == [[None, None, 9], [None, None, 9]]
    assert_response_valid(body, counts, 5)


def test_supports_4x4_matrix():
    counts = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    response = client.post(ENDPOINT, json=payload(counts=counts, threshold=5))
    assert response.status_code == 200
    assert_response_valid(response.json(), counts, 5)


def test_supports_large_values():
    large = 10**12
    counts = [[large, large], [large, 1]]
    response = client.post(ENDPOINT, json=payload(counts=counts, threshold=2))
    assert response.status_code == 200
    body = response.json()
    assert body["row_totals"] == [2 * large, large + 1]
    assert body["column_totals"] == [2 * large, large + 1]
    assert_response_valid(body, counts, 2)


def test_matches_brute_force_reference_on_random_matrices():
    # 固定种子的随机矩阵：响应必须通过结构校验并等于暴力参考解。
    rng = random.Random(20260926)
    for _ in range(60):
        rows = rng.randint(2, 4)
        cols = rng.randint(2, 4)
        counts = [[rng.randint(1, 12) for _ in range(cols)] for _ in range(rows)]
        threshold = rng.randint(1, 14)
        response = client.post(
            ENDPOINT, json=payload(counts=counts, threshold=threshold)
        )
        assert response.status_code == 200, (counts, threshold, response.json())
        assert_response_valid(response.json(), counts, threshold)


# ---------------------------------------------------------------- 422：counts 格式

def test_422_on_wrong_row_count():
    assert client.post(ENDPOINT, json=payload(counts=[])).status_code == 422
    assert client.post(ENDPOINT, json=payload(counts=[[1, 2]])).status_code == 422
    counts = [[1, 2]] * 5
    assert client.post(ENDPOINT, json=payload(counts=counts)).status_code == 422


def test_422_on_wrong_column_count():
    assert client.post(ENDPOINT, json=payload(counts=[[1], [2]])).status_code == 422
    assert client.post(ENDPOINT, json=payload(counts=[[], []])).status_code == 422
    counts = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    assert client.post(ENDPOINT, json=payload(counts=counts)).status_code == 422


def test_422_on_counts_not_a_matrix():
    for bad in ("x", 5, None, [1, 2]):
        response = client.post(ENDPOINT, json=payload(counts=bad))
        assert response.status_code == 422, bad


def test_422_on_ragged_matrix():
    counts = [[1, 2], [3, 4, 5]]
    assert client.post(ENDPOINT, json=payload(counts=counts)).status_code == 422


def test_422_on_non_positive_count():
    assert (
        client.post(ENDPOINT, json=payload(counts=[[1, 2], [3, 0]])).status_code
        == 422
    )
    assert (
        client.post(ENDPOINT, json=payload(counts=[[1, 2], [3, -4]])).status_code
        == 422
    )


def test_422_on_non_integer_count():
    for bad in (True, 1.5, "3", None):
        response = client.post(ENDPOINT, json=payload(counts=[[bad, 2], [3, 4]]))
        assert response.status_code == 422, bad


# ---------------------------------------------------------------- 422：threshold 格式

def test_422_on_non_positive_threshold():
    for bad in (0, -5):
        response = client.post(ENDPOINT, json=payload(threshold=bad))
        assert response.status_code == 422, bad


def test_422_on_non_integer_threshold():
    for bad in (True, 2.5, "5"):
        response = client.post(ENDPOINT, json=payload(threshold=bad))
        assert response.status_code == 422, bad


# ---------------------------------------------------------------- 422：请求体结构

def test_422_on_missing_or_extra_body_field():
    body = payload()
    for key in ("counts", "threshold"):
        incomplete = {k: v for k, v in body.items() if k != key}
        assert client.post(ENDPOINT, json=incomplete).status_code == 422, key
    assert client.post(ENDPOINT, json=payload(extra=1)).status_code == 422


def test_422_detail_mentions_reason():
    response = client.post(ENDPOINT, json=payload(counts=[[1, 2], [3, 0]]))
    assert response.status_code == 422
    assert "正整数" in response.json()["detail"]


# ---------------------------------------------------------------- 既有服务不受影响

def test_existing_endpoints_still_available():
    assert client.get("/api/v1/health").status_code == 200
    root = client.get("/").json()
    assert root["suppress_count_table"] == "/api/v1/suppress-count-table"
    assert root["round_count_table"] == "/api/v1/round-count-table"
