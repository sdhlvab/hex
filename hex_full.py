# hex_easyai.py
# Gra Hex 11x11 na easyAI: Negamax + heurystyka najkrótszej ścieżki
# P1 ('X') łączy GÓRĘ z DOŁEM. P2 ('O') łączy LEWO z PRAWO.
# Autor: przygotowane dla Kamilka :)

from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax
import string
import sys
import heapq
from collections import deque

BOARD_SIZE = 11  # 11x11
ROWS = BOARD_SIZE
COLS = BOARD_SIZE
LETTERS = list(string.ascii_uppercase[:COLS])  # ['A'..'K']

EMPTY = 0
P1 = 1  # X: top-bottom
P2 = 2  # O: left-right

SYMBOL = {
    EMPTY: ".",
    P1: "X",
    P2: "O",
}

# --- narzędzia do współrzędnych ---
def parse_move(move_str):
    """
    Akceptuje formy: 'A1', 'a1', 'A 1', '1,1' (wtedy to wiersz,kolumna 1-index)
    Zwraca (r, c) 0-index.
    """
    s = move_str.strip().replace(",", " ").replace(";", " ")
    parts = s.split()
    if len(parts) == 1:
        token = parts[0]
        # próba formatu 'A1'
        if len(token) >= 2 and token[0].isalpha():
            col_letter = token[0].upper()
            num = token[1:]
            if col_letter in LETTERS and num.isdigit():
                c = LETTERS.index(col_letter)
                r = int(num) - 1
                return (r, c)
        # próba formatu '1,1' bez spacji (np. '11,3' nie przejdzie tu)
        # zostawiamy tylko 'A1' powyżej
    else:
        # próba formatu rozdzielonego spacją: 'A 1' lub '1 1'
        a, b = parts[0], parts[1]
        if a.isalpha() and b.isdigit():
            col_letter = a.upper()
            if col_letter in LETTERS:
                c = LETTERS.index(col_letter)
                r = int(b) - 1
                return (r, c)
        if a.isdigit() and b.isdigit():
            # traktuj jako (wiersz, kolumna) oba 1-index
            r = int(a) - 1
            c = int(b) - 1
            return (r, c)
    raise ValueError(f"Niepoprawny ruch: '{move_str}'. Użyj np. A1, B7, 'A 1' lub '7 8'.")

def move_to_str(rc):
    r, c = rc
    return f"{LETTERS[c]}{r+1}"

# --- sąsiedzi w heksie ---
def neighbors(r, c):
    """Sąsiedzi sześciokątni."""
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < ROWS and 0 <= cc < COLS:
            yield rr, cc

# --- wykrywanie zwycięstwa (DFS/ BFS per gracz) ---
def path_exists(board, player):
    """
    True jeśli istnieje ścieżka od jednej krawędzi do przeciwnej dla gracza.
    P1: góra->dół, P2: lewo->prawo.
    """
    visited = [[False]*COLS for _ in range(ROWS)]
    dq = deque()

    if player == P1:
        # start: górny rząd
        for c in range(COLS):
            if board[0][c] == P1:
                dq.append((0, c))
                visited[0][c] = True
        # cel: dowolny w dolnym rzędzie
        goal_row = ROWS - 1
        while dq:
            r, c = dq.popleft()
            if r == goal_row:
                return True
            for rr, cc in neighbors(r, c):
                if not visited[rr][cc] and board[rr][cc] == P1:
                    visited[rr][cc] = True
                    dq.append((rr, cc))
        return False

    else:  # P2
        # start: lewa kolumna
        for r in range(ROWS):
            if board[r][0] == P2:
                dq.append((r, 0))
                visited[r][0] = True
        # cel: dowolna w prawej kolumnie
        goal_col = COLS - 1
        while dq:
            r, c = dq.popleft()
            if c == goal_col:
                return True
            for rr, cc in neighbors(r, c):
                if not visited[rr][cc] and board[rr][cc] == P2:
                    visited[rr][cc] = True
                    dq.append((rr, cc))
        return False

# --- heurystyka: “najkrótsza ścieżka” (Dijkstra) ---
def shortest_path_cost(board, player):
    """
    Przybliżenie: liczymy minimalny koszt przejścia z boku do boku.
    Wejście na:
      - własne pole: koszt 0
      - puste pole: koszt 1
      - pole przeciwnika: koszt 3 (utrudnienie)
    Im mniejszy koszt, tym lepiej dla danego gracza.
    """
    INF = 10**9
    opp = P1 if player == P2 else P2

    # odległości dla wszystkich pól
    dist = [[INF]*COLS for _ in range(ROWS)]
    pq = []

    def cell_cost(r, c):
        v = board[r][c]
        if v == player:
            return 0
        elif v == EMPTY:
            return 1
        else:  # przeciwnik
            return 3

    if player == P1:
        # źródło: górny rząd -> zainicjalizuj ich koszty
        for c in range(COLS):
            d = cell_cost(0, c)
            dist[0][c] = d
            heapq.heappush(pq, (d, 0, c))

        target_row = ROWS - 1
        while pq:
            d, r, c = heapq.heappop(pq)
            if d != dist[r][c]:
                continue
            if r == target_row:
                return d
            for rr, cc in neighbors(r, c):
                nd = d + cell_cost(rr, cc)
                if nd < dist[rr][cc]:
                    dist[rr][cc] = nd
                    heapq.heappush(pq, (nd, rr, cc))
        return INF

    else:  # P2
        # źródło: lewa kolumna
        for r in range(ROWS):
            d = cell_cost(r, 0)
            dist[r][0] = d
            heapq.heappush(pq, (d, r, 0))

        target_col = COLS - 1
        while pq:
            d, r, c = heapq.heappop(pq)
            if d != dist[r][c]:
                continue
            if c == target_col:
                return d
            for rr, cc in neighbors(r, c):
                nd = d + cell_cost(rr, cc)
                if nd < dist[rr][cc]:
                    dist[rr][cc] = nd
                    heapq.heappush(pq, (nd, rr, cc))
        return INF

def heuristic_score(board, current_player):
    """
    Ocena pozycji z perspektywy bieżącego gracza (Negamax).
    Mniejszy koszt najkrótszej ścieżki = lepiej. Bierzemy różnicę kosztów:
    score ≈ (cost_opp - cost_me) * waga
    """
    me = current_player
    opp = P1 if me == P2 else P2

    my_cost = shortest_path_cost(board, me)
    opp_cost = shortest_path_cost(board, opp)

    # szybkie zwycięstwa/porazki:
    if path_exists(board, me):
        return 10000
    if path_exists(board, opp):
        return -10000

    # surowa różnica kosztów (im większa na plus, tym lepiej dla "me")
    diff = opp_cost - my_cost
    return diff * 50  # wzmocnienie, by AI bardziej to czuło

# --- Klasa gry easyAI ---
class HexGame(TwoPlayerGame):
    def __init__(self, players):
        self.players = players
        self.board = [[EMPTY]*COLS for _ in range(ROWS)]
        self.current_player = 1  # P1 zaczyna (X)
        self.history = []  # do unmake_move

    def possible_moves(self):
        # Zwracamy listę tekstowych koordynatów, np. "A1", "B3", ...
        moves = []
        for r in range(ROWS):
            for c in range(COLS):
                if self.board[r][c] == EMPTY:
                    moves.append(move_to_str((r, c)))
        return moves

    def make_move(self, move):
        # parsuj i wykonaj
        r, c = parse_move(move)
        if self.board[r][c] != EMPTY:
            # w easyAI gracz może wpisać zły ruch – pozwólmy na błąd słowny
            raise ValueError("Pole zajęte. Spróbuj inne.")
        self.board[r][c] = self.current_player
        self.history.append((r, c))

    def unmake_move(self, move):
        # Cofnij ostatni ruch (dla wydajności AI)
        r, c = self.history.pop()
        self.board[r][c] = EMPTY

    def show(self):
        # Ładny wydruk hex (lekko ukośnie)
        header = "   " + " ".join(LETTERS)
        print(header)
        for r in range(ROWS):
            indent = "  " * r
            row_str = " ".join(SYMBOL[self.board[r][c]] for c in range(COLS))
            print(f"{r+1:>2} {indent}{row_str}")

        # podpowiedź kto łączy które krawędzie
        print("\nLegenda: X (P1) łączy GÓRA–DÓŁ,  O (P2) łączy LEWO–PRAWO.")
        print(f"Ruch gracza: {'X' if self.current_player == P1 else 'O'}")
        print("Podaj ruch np. A1, B7, 'A 1' albo '7 8'.")

    # Logika końca gry
    def win(self):
        # easyAI odpyta o zwycięstwo dla aktualnego układu; umownie sprawdzamy obie strony
        return path_exists(self.board, P1) or path_exists(self.board, P2)

    def is_over(self):
        if path_exists(self.board, P1) or path_exists(self.board, P2):
            return True
        # Teoretycznie Hex nie kończy się remisem; ale jak brak ruchów, to koniec
        return len(self.possible_moves()) == 0

    def scoring(self):
        # Negamax oczekuje oceny z perspektywy current_player
        me = self.current_player
        return heuristic_score(self.board, me)

    # (opcjonalnie) czy obecny gracz przegrał — niekonieczne, bo mamy scoring()
    def loss(self):
        opp = P1 if self.current_player == P2 else P2
        return path_exists(self.board, opp)

# --- main ---
def ask_depth():
    while True:
        try:
            val = input("Podaj głębokość przeszukiwania AI (2–4 dla 11x11, domyślnie 3): ").strip()
            if val == "":
                return 3
            d = int(val)
            if d < 1:
                print("Podaj liczbę dodatnią.")
                continue
            return d
        except Exception:
            print("Wpisz liczbę całkowitą, np. 3.")

def ask_who_starts():
    while True:
        s = input("Kto zaczyna? (1 = Ty jako X, 2 = AI jako X) [domyślnie: 1]: ").strip()
        if s == "":
            return 1
        if s in ("1", "2"):
            return int(s)
        print("Wpisz 1 lub 2.")

if __name__ == "__main__":
    print("=== HEX 11x11 (easyAI, Negamax) ===")
    depth = ask_depth()
    who = ask_who_starts()

    ai_algo = Negamax(depth)

    if who == 1:
        game = HexGame([Human_Player(), AI_Player(ai_algo)])
    else:
        game = HexGame([AI_Player(ai_algo), Human_Player()])

    # mała podpowiedź
    print("\nWskazówki:")
    print("- P1 (X) łączy GÓRĘ z DOŁEM. P2 (O) łączy LEWO z PRAWO.")
    print("- Ruch wpisuj jak 'A1', 'B7', 'A 1' albo '7 8'.\n")

    game.play()

    # ogłoszenie zwycięzcy
    if path_exists(game.board, P1):
        print("\nWygrał gracz X (P1) – połączenie GÓRA–DÓŁ!")
    elif path_exists(game.board, P2):
        print("\nWygrał gracz O (P2) – połączenie LEWO–PRAWO!")
    else:
        print("\nKoniec gry.")
