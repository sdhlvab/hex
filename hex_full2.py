# hex_full.py
# Hex 11x11 na easyAI: Negamax + ulepszone logi, memoizacja i szybsza heurystyka.
# P1 ('X') łączy GÓRĘ–DÓŁ. P2 ('O') łączy LEWO–PRAWO.

from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax
import string, sys, time, heapq
from collections import deque
from functools import lru_cache

BOARD_SIZE = 11
ROWS = COLS = BOARD_SIZE
LETTERS = list(string.ascii_uppercase[:COLS])

EMPTY, P1, P2 = 0, 1, 2
SYMBOL = {EMPTY: ".", P1: "X", P2: "O"}

# ------- utils: coords & IO --------
def parse_move(move_str):
    s = move_str.strip().replace(",", " ").replace(";", " ")
    parts = s.split()
    if not parts:
        raise ValueError("Pusty ruch.")
    if len(parts) == 1:
        token = parts[0]
        if len(token) >= 2 and token[0].isalpha() and token[1:].isdigit():
            col_letter = token[0].upper()
            if col_letter in LETTERS:
                c = LETTERS.index(col_letter)
                r = int(token[1:]) - 1
                return (r, c)
    else:
        a, b = parts[0], parts[1]
        if a.isalpha() and b.isdigit():
            col_letter = a.upper()
            if col_letter in LETTERS:
                c = LETTERS.index(col_letter)
                r = int(b) - 1
                return (r, c)
        if a.isdigit() and b.isdigit():
            r = int(a) - 1
            c = int(b) - 1
            return (r, c)
    raise ValueError(f"Niepoprawny ruch: '{move_str}'. Użyj np. A1, B7, 'A 1' albo '7 8'.")

def move_to_str(rc):
    r, c = rc
    return f"{LETTERS[c]}{r+1}"

# ------- neighbors on hex -------
NEIGH_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
def neighbors(r, c):
    for dr, dc in NEIGH_STEPS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < ROWS and 0 <= cc < COLS:
            yield rr, cc

# ------- victory detection -------
def path_exists(board, player):
    visited = [[False]*COLS for _ in range(ROWS)]
    dq = deque()
    if player == P1:
        for c in range(COLS):
            if board[0][c] == P1:
                dq.append((0, c)); visited[0][c] = True
        goal_row = ROWS - 1
        while dq:
            r, c = dq.popleft()
            if r == goal_row: return True
            for rr, cc in neighbors(r, c):
                if not visited[rr][cc] and board[rr][cc] == P1:
                    visited[rr][cc] = True; dq.append((rr, cc))
        return False
    else:
        for r in range(ROWS):
            if board[r][0] == P2:
                dq.append((r, 0)); visited[r][0] = True
        goal_col = COLS - 1
        while dq:
            r, c = dq.popleft()
            if c == goal_col: return True
            for rr, cc in neighbors(r, c):
                if not visited[rr][cc] and board[rr][cc] == P2:
                    visited[rr][cc] = True; dq.append((rr, cc))
        return False

# ------- heuristic: faster shortest-path + caches -------
def board_key(board):
    # hashable representation for caching
    return tuple(tuple(row) for row in board)

@lru_cache(maxsize=200_000)
def _shortest_path_cost_cached(key, player):
    # reconstruct board (tiny overhead vs huge savings on reuse)
    board = [list(row) for row in key]
    INF = 10**9

    def cell_cost(r, c):
        v = board[r][c]
        if v == player: return 0
        if v == EMPTY:  return 1
        return 3  # opponent

    dist = [[INF]*COLS for _ in range(ROWS)]
    pq = []

    if player == P1:
        for c in range(COLS):
            d = cell_cost(0, c)
            dist[0][c] = d
            heapq.heappush(pq, (d, 0, c))
        target_row = ROWS - 1
        while pq:
            d, r, c = heapq.heappop(pq)
            if d != dist[r][c]: continue
            if r == target_row: return d
            for rr, cc in neighbors(r, c):
                nd = d + cell_cost(rr, cc)
                if nd < dist[rr][cc]:
                    dist[rr][cc] = nd; heapq.heappush(pq, (nd, rr, cc))
        return INF
    else:
        for r in range(ROWS):
            d = cell_cost(r, 0)
            dist[r][0] = d
            heapq.heappush(pq, (d, r, 0))
        target_col = COLS - 1
        while pq:
            d, r, c = heapq.heappop(pq)
            if d != dist[r][c]: continue
            if c == target_col: return d
            for rr, cc in neighbors(r, c):
                nd = d + cell_cost(rr, cc)
                if nd < dist[rr][cc]:
                    dist[rr][cc] = nd; heapq.heappush(pq, (nd, rr, cc))
        return INF

def shortest_path_cost(board, player):
    return _shortest_path_cost_cached(board_key(board), player)

@lru_cache(maxsize=200_000)
def _path_exists_cached(key, player):
    board = [list(row) for row in key]
    return path_exists(board, player)

def path_exists_cached(board, player):
    return _path_exists_cached(board_key(board), player)

def heuristic_score(board, current_player):
    key = board_key(board)
    me  = current_player
    opp = P1 if me == P2 else P2

    if path_exists_cached(board, me):  return 10000
    if path_exists_cached(board, opp): return -10000

    my_cost  = _shortest_path_cost_cached(key, me)
    opp_cost = _shortest_path_cost_cached(key, opp)
    diff = opp_cost - my_cost
    # lekka preferencja kontaktu z brzegiem (ważne dla P2 na kolumnie A, dla P1 na wierszu 1)
    edge_bonus = 0
    if me == P1:
        if any(board[0][c] == P1 for c in range(COLS)): edge_bonus += 3
        if any(board[ROWS-1][c] == P1 for c in range(COLS)): edge_bonus += 3
    else:
        if any(board[r][0] == P2 for r in range(ROWS)): edge_bonus += 3
        if any(board[r][COLS-1] == P2 for r in range(ROWS)): edge_bonus += 3

    return diff * 50 + edge_bonus

# ------- move ordering: center-first (cheap & helps pruning) -------
CENTER_R, CENTER_C = (ROWS-1)/2.0, (COLS-1)/2.0
def center_priority_key(rc):
    r, c = rc
    # mniejsza odległość do środka -> wyżej na liście
    return (abs(r - CENTER_R) + abs(c - CENTER_C), r, c)

# ------- Game class -------
class HexGame(TwoPlayerGame):
    def __init__(self, players):
        self.players = players
        self.board = [[EMPTY]*COLS for _ in range(ROWS)]
        self.current_player = 1
        self.history = []

    def possible_moves(self):
        empties = [(r, c) for r in range(ROWS) for c in range(COLS) if self.board[r][c] == EMPTY]
        empties.sort(key=center_priority_key)
        return [move_to_str(rc) for rc in empties]

    def make_move(self, move):
        r, c = parse_move(move)
        if not (0 <= r < ROWS and 0 <= c < COLS):
            raise ValueError("Poza planszą.")
        if self.board[r][c] != EMPTY:
            raise ValueError("Pole zajęte. Spróbuj inne.")
        self.board[r][c] = self.current_player
        self.history.append((r, c))

    def unmake_move(self, move):
        r, c = self.history.pop()
        self.board[r][c] = EMPTY

    def show(self):
        header = "   " + " ".join(LETTERS)
        print(header)
        for r in range(ROWS):
            indent = "  " * r
            row_str = " ".join(SYMBOL[self.board[r][c]] for c in range(COLS))
            print(f"{r+1:>2} {indent}{row_str}")
        print("\nLegenda: X (P1) łączy GÓRA–DÓŁ,  O (P2) łączy LEWO–PRAWO.")
        print(f"Teraz ruch: {'TY (X)' if self.current_player == P1 else 'AI (O)'}")
        print("Podaj ruch np. A1, B7, 'A 1' albo '7 8'.")

    def win(self):
        return path_exists_cached(self.board, P1) or path_exists_cached(self.board, P2)

    def is_over(self):
        if self.win(): return True
        return all(self.board[r][c] != EMPTY for r in range(ROWS) for c in range(COLS))

    def scoring(self):
        return heuristic_score(self.board, self.current_player)

# ------- Verbose AI wrapper (czytelne komunikaty + timing) -------
class VerboseAI(AI_Player):
    def ask_move(self, game):
        print("\n[AI] Myślę nad ruchem…")
        t0 = time.perf_counter()
        move = super().ask_move(game)
        dt = time.perf_counter() - t0
        print(f"[AI] Ruch: {move}  (czas: {dt:.2f}s)\n")
        return move

# ------- CLI helpers -------
def ask_depth():
    while True:
        try:
            val = input("Głębokość AI (2–3 dla 11x11; domyślnie 2): ").strip()
            if val == "": return 2
            d = int(val)
            if d < 1: print("Podaj liczbę ≥ 1."); continue
            return d
        except Exception:
            print("Wpisz liczbę całkowitą, np. 2.")

def ask_who_starts():
    while True:
        s = input("Kto zaczyna? (1 = Ty/X, 2 = AI/X) [domyślnie: 1]: ").strip()
        if s == "": return 1
        if s in ("1","2"): return int(s)
        print("Wpisz 1 albo 2.")

if __name__ == "__main__":
    print("=== HEX 11x11 (easyAI, Negamax, fast) ===")
    depth = ask_depth()
    who = ask_who_starts()

    ai_algo = Negamax(depth)
    game = HexGame([Human_Player(), VerboseAI(ai_algo)]) if who == 1 else HexGame([VerboseAI(ai_algo), Human_Player()])

    print("\nWskazówki:")
    print("- P1 (X) łączy GÓRA–DÓŁ. P2 (O) łączy LEWO–PRAWO.")
    print("- Staraj się dotknąć **swoich krawędzi** możliwie wcześnie (to ważne!).")
    print("- Ruch wpisuj jak 'A1', 'B7', 'A 1' albo '7 8'.\n")

    game.play()

    if path_exists_cached(game.board, P1):
        print("\nWygrał gracz X (P1) – połączenie GÓRA–DÓŁ!")
    elif path_exists_cached(game.board, P2):
        print("\nWygrała AI O (P2) – połączenie LEWO–PRAWO!")
    else:
        print("\nKoniec gry.")
