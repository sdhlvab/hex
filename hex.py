"""
HEX 11x11 — gra turowa, deterministyczna, dwuosobowa o sumie zerowej.

Opis:
    Implementacja gry Hex (plansza 11×11) z AI wykorzystującą easyAI i algorytm
    Negamax. P1 ('X') łączy GÓRĘ–DÓŁ, P2 ('O') łączy LEWO–PRAWO.
    Zawiera:
      - parser i formatowanie ruchów,
      - sąsiadów heksowych (6 kierunków),
      - detekcję zwycięstwa (BFS),
      - heurystykę opartą o najkrótszą ścieżkę (Dijkstra) + memoizację,
      - porządkowanie ruchów (preferencja środka),
      - interfejs CLI i opakowanie AI z logami/timingiem.

Autorzy:
    Kamil Koniak s26766
    Kamil Suchomski s21974

Zasady gry:
    - Gracze stawiają na przemian po jednym kamieniu na pustym polu.
    - P1 ('X') wygrywa, jeśli połączy górną i dolną krawędź.
    - P2 ('O') wygrywa, jeśli połączy lewą i prawą krawędź.
    - Brak bicia/usuwania kamieni.

Wymagania środowiskowe:
    - Python 3.9–3.12 (testowane na 3.12).
    - easyAI: `pip install easyAI`

Uruchomienie:
    $ python hex_full.py
    → podaj głębokość (2 zalecane dla 11×11) oraz kto zaczyna (Ty/AI).
"""

from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax
import string
import sys
import time
import heapq
from collections import deque
from functools import lru_cache

# --- parametry planszy i symbole ------------------------------------------------

BOARD_SIZE = 11
ROWS = COLS = BOARD_SIZE
LETTERS = list(string.ascii_uppercase[:COLS])

EMPTY, P1, P2 = 0, 1, 2
SYMBOL = {EMPTY: ".", P1: "X", P2: "O"}


# ------- utils: coords & IO -----------------------------------------------------

def parse_move(move_str):
    """Zamienia tekst ruchu na współrzędne (wiersz, kolumna) w indeksowaniu 0-based.

    Argumenty:
        move_str (str): Tekst ruchu. Akceptowane formy:
            - "A1", "B7", "K11" (litera kolumny + numer wiersza, 1-indeksowane),
            - "A 1" (litera i liczba rozdzielone spacją),
            - "7 8" (wiersz i kolumna jako liczby, 1-indeksowane).

    Zwraca:
        tuple[int, int]: Para (r, c) w indeksowaniu 0-based.

    Wyjątki:
        ValueError: Gdy format jest niepoprawny lub wejście jest puste.
    """
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
    raise ValueError(
        f"Niepoprawny ruch: '{move_str}'. Użyj np. A1, B7, 'A 1' albo '7 8'."
    )


def move_to_str(rc):
    """Formatuje współrzędne (0-based) do postaci przyjaznej użytkownikowi, np. 'A1'.

    Argumenty:
        rc (tuple[int, int]): Współrzędne 0-based (r, c).

    Zwraca:
        str: Np. "A1", "F6".
    """
    r, c = rc
    return f"{LETTERS[c]}{r+1}"


# ------- neighbors on hex -------------------------------------------------------

NEIGH_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]


def neighbors(r, c):
    """Generator sąsiadów heksowych (6 kierunków) dla danego pola.

    Argumenty:
        r (int): Wiersz (0..ROWS-1).
        c (int): Kolumna (0..COLS-1).

    Zwraca:
        iterator[tuple[int, int]]: Współrzędne sąsiadów (rr, cc) w granicach planszy.
    """
    for dr, dc in NEIGH_STEPS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < ROWS and 0 <= cc < COLS:
            yield rr, cc


# ------- victory detection ------------------------------------------------------

def path_exists(board, player):
    """Sprawdza, czy gracz ułożył już zwycięską ścieżkę krawędź-krawędź.

    Dla P1 weryfikuje połączenie GÓRA (r=0) ↔ DÓŁ (r=ROWS-1).
    Dla P2 weryfikuje połączenie LEWO (c=0) ↔ PRAWO (c=COLS-1).
    Implementacja oparta na BFS.

    Argumenty:
        board (list[list[int]]): Aktualny stan planszy.
        player (int): P1 (1) lub P2 (2).

    Zwraca:
        bool: True, jeśli istnieje połączenie krawędź-krawędź; w p.p. False.
    """
    visited = [[False] * COLS for _ in range(ROWS)]
    dq = deque()
    if player == P1:
        for c in range(COLS):
            if board[0][c] == P1:
                dq.append((0, c))
                visited[0][c] = True
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
    else:
        for r in range(ROWS):
            if board[r][0] == P2:
                dq.append((r, 0))
                visited[r][0] = True
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


# ------- heuristic: faster shortest-path + caches -------------------------------

def board_key(board):
    """Buduje niezmienniczy (hashowalny) klucz planszy do cache’owania.

    Argumenty:
        board (list[list[int]]): Plansza jako listy list.

    Zwraca:
        tuple[tuple[int, ...], ...]: Niezmiennicza reprezentacja planszy.
    """
    return tuple(tuple(row) for row in board)


@lru_cache(maxsize=200_000)
def _shortest_path_cost_cached(key, player):
    """Szacuje koszt najkrótszej ścieżki łączącej krawędzie dla danego gracza (cache).

    Metoda: algorytm Dijkstry od krawędzi startowej do docelowej.
    Koszt pola:
        - własny kamień: 0
        - puste pole:    1
        - kamień przeciwnika: 3

    Argumenty:
        key (tuple[tuple[int, ...], ...]): Hashowalny klucz planszy (patrz `board_key`).
        player (int): P1 (1) lub P2 (2).

    Zwraca:
        int: Najmniejszy znaleziony koszt przejścia; INF jeśli nieosiągalne.

    Uwagi:
        Funkcja jest cache’owana (`functools.lru_cache`), co radykalnie redukuje
        liczbę powtórek obliczeń w drzewie przeszukiwania.
    """
    board = [list(row) for row in key]
    INF = 10**9

    def cell_cost(r, c):
        v = board[r][c]
        if v == player:
            return 0
        if v == EMPTY:
            return 1
        return 3  # przeciwnik

    dist = [[INF] * COLS for _ in range(ROWS)]
    pq = []

    if player == P1:
        # start: górna krawędź, cel: dolna krawędź
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
    else:
        # start: lewa krawędź, cel: prawa krawędź
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


def shortest_path_cost(board, player):
    """Publiczny wrapper (niecache’owany) do wyznaczania kosztu najkrótszej ścieżki.

    Argumenty:
        board (list[list[int]]): Aktualny stan planszy.
        player (int): P1 (1) lub P2 (2).

    Zwraca:
        int: Szacowany koszt najkrótszej ścieżki gracza.
    """
    return _shortest_path_cost_cached(board_key(board), player)


@lru_cache(maxsize=200_000)
def _path_exists_cached(key, player):
    """Cache’owana wersja sprawdzania zwycięskiej ścieżki.

    Argumenty:
        key (tuple[tuple[int, ...], ...]): Hashowalny klucz planszy.
        player (int): P1 (1) lub P2 (2).

    Zwraca:
        bool: True, jeśli gracz ma połączenie krawędź-krawędź.
    """
    board = [list(row) for row in key]
    return path_exists(board, player)


def path_exists_cached(board, player):
    """Sprawdza zwycięstwo z wykorzystaniem cache.

    Argumenty:
        board (list[list[int]]): Aktualny stan planszy.
        player (int): P1 (1) lub P2 (2).

    Zwraca:
        bool: True, jeśli gracz ma już połączenie krawędź-krawędź.
    """
    return _path_exists_cached(board_key(board), player)


def heuristic_score(board, current_player):
    """Wartościuje pozycję z punktu widzenia wskazanego gracza.

    Heurystyka:
        - natychmiastowe ±10000 przy wykryciu wygranej/przegranej,
        - w przeciwnym razie: (koszt_opp − koszt_mój) × 50,
        - lekki bonus za kontakt z własnymi krawędziami (otwarcie gry).

    Argumenty:
        board (list[list[int]]): Aktualny stan planszy.
        current_player (int): Gracz, dla którego liczona jest ocena (P1 lub P2).

    Zwraca:
        int: Im większa, tym lepsza pozycja dla `current_player`.
    """
    key = board_key(board)
    me = current_player
    opp = P1 if me == P2 else P2

    if path_exists_cached(board, me):
        return 10000
    if path_exists_cached(board, opp):
        return -10000

    my_cost = _shortest_path_cost_cached(key, me)
    opp_cost = _shortest_path_cost_cached(key, opp)
    diff = opp_cost - my_cost

    # Preferencja kontaktu z brzegiem (pomaga w otwarciu i ukierunkowaniu gry).
    edge_bonus = 0
    if me == P1:
        if any(board[0][c] == P1 for c in range(COLS)):
            edge_bonus += 3
        if any(board[ROWS - 1][c] == P1 for c in range(COLS)):
            edge_bonus += 3
    else:
        if any(board[r][0] == P2 for r in range(ROWS)):
            edge_bonus += 3
        if any(board[r][COLS - 1] == P2 for r in range(ROWS)):
            edge_bonus += 3

    return diff * 50 + edge_bonus


# ------- move ordering: center-first (cheap & helps pruning) --------------------

CENTER_R, CENTER_C = (ROWS - 1) / 2.0, (COLS - 1) / 2.0


def center_priority_key(rc):
    """Funkcja klucza sortowania preferująca ruchy bliższe środkowi planszy.

    Argumenty:
        rc (tuple[int, int]): Pozycja pola (r, c) 0-based.

    Zwraca:
        tuple[float, int, int]: Klucz sortowania (mniejszy = „lepszy”).
    """
    r, c = rc
    return (abs(r - CENTER_R) + abs(c - CENTER_C), r, c)


# ------- Game class -------------------------------------------------------------

class HexGame(TwoPlayerGame):
    """Implementacja easyAI `TwoPlayerGame` dla gry Hex 11×11.

    Atrybuty:
        players (list): Dwuelementowa lista graczy (Human/AI) w kolejności tur.
        board (list[list[int]]): Plansza 11×11 (0=puste, 1=P1, 2=P2).
        current_player (int): Numer aktualnego gracza wg easyAI (1 lub 2).
        history (list[tuple[int, int]]): Stos zagranych posunięć (r, c).

    Uwagi:
        - `scoring()` powinno zwracać ocenę z perspektywy gracza będącego na ruchu
          (wymóg frameworka easyAI).
        - `make_move`/`unmake_move` wspierają przeszukiwanie drzewa gry.
    """

    def __init__(self, players):
        """Inicjuje pustą planszę i ustawia pierwszego gracza.

        Argumenty:
            players (list[AI_Player|Human_Player]): Dwóch graczy zgodnie z kolejnością tur.
        """
        self.players = players
        self.board = [[EMPTY] * COLS for _ in range(ROWS)]
        self.current_player = 1
        self.history = []

    def possible_moves(self):
        """Zwraca wszystkie legalne ruchy jako napisy, posortowane preferencją środka.

        Zwraca:
            list[str]: Ruchy w formacie "A1", "B7", ...
        """
        empties = [(r, c) for r in range(ROWS) for c in range(COLS) if self.board[r][c] == EMPTY]
        empties.sort(key=center_priority_key)
        return [move_to_str(rc) for rc in empties]

    def make_move(self, move):
        """Wykonuje ruch na planszy.

        Argumenty:
            move (str): Ruch w formacie akceptowanym przez `parse_move`.

        Wyjątki:
            ValueError: Gdy ruch wychodzi poza planszę lub pole jest zajęte.
        """
        r, c = parse_move(move)
        if not (0 <= r < ROWS and 0 <= c < COLS):
            raise ValueError("Poza planszą.")
        if self.board[r][c] != EMPTY:
            raise ValueError("Pole zajęte. Spróbuj inne.")
        self.board[r][c] = self.current_player
        self.history.append((r, c))

    def unmake_move(self, move):
        """Cofa ostatni ruch (używane podczas przeszukiwania).

        Argumenty:
            move (str): Ignorowane — zgodność z interfejsem easyAI.
        """
        r, c = self.history.pop()
        self.board[r][c] = EMPTY

    def show(self):
        """Wyświetla aktualny stan planszy w układzie ASCII przypominającym heksy."""
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
        """Informuje easyAI, czy gra została wygrana przez któregokolwiek gracza.

        Zwraca:
            bool: True, jeśli istnieje zwycięska ścieżka P1 lub P2.
        """
        return path_exists_cached(self.board, P1) or path_exists_cached(self.board, P2)

    def is_over(self):
        """Informuje easyAI o zakończeniu gry (wygrana lub pełna plansza).

        Zwraca:
            bool: True, jeśli gra dobiegła końca.
        """
        if self.win():
            return True
        return all(self.board[r][c] != EMPTY for r in range(ROWS) for c in range(COLS))

    def scoring(self):
        """Zwraca ocenę pozycji dla gracza będącego na ruchu (wymóg easyAI).

        Zwraca:
            int: Ocena pozycji dla `self.current_player`.
        """
        return heuristic_score(self.board, self.current_player)


# ------- Verbose AI wrapper (czytelne komunikaty + timing) ----------------------

class VerboseAI(AI_Player):
    """Opakowanie AI_Player dodające logi o myśleniu i czasie obliczeń."""

    def ask_move(self, game):
        """Prosi AI o ruch, wypisując postęp oraz czas obliczeń.

        Argumenty:
            game (TwoPlayerGame): Aktualny stan gry.

        Zwraca:
            str: Wybrany ruch (np. "F6").
        """
        print("\n[AI] Myślę nad ruchem…")
        t0 = time.perf_counter()
        move = super().ask_move(game)
        dt = time.perf_counter() - t0
        print(f"[AI] Ruch: {move}  (czas: {dt:.2f}s)\n")
        return move


# ------- CLI helpers ------------------------------------------------------------

def ask_depth():
    """Pyta użytkownika o głębokość przeszukiwania Negamax.

    Zwraca:
        int: Głębokość przeszukiwania (domyślnie 2; minimalnie 1).
    """
    while True:
        try:
            val = input("Głębokość AI (2–3 dla 11x11; domyślnie 2): ").strip()
            if val == "":
                return 2
            d = int(val)
            if d < 1:
                print("Podaj liczbę ≥ 1.")
                continue
            return d
        except Exception:
            print("Wpisz liczbę całkowitą, np. 2.")


def ask_who_starts():
    """Pyta, kto zaczyna partię (1 = Człowiek/X, 2 = AI/X).

    Zwraca:
        int: 1 lub 2 (domyślnie 1).
    """
    while True:
        s = input("Kto zaczyna? (1 = Ty/X, 2 = AI/X) [domyślnie: 1]: ").strip()
        if s == "":
            return 1
        if s in ("1", "2"):
            return int(s)
        print("Wpisz 1 albo 2.")


# ------- main -------------------------------------------------------------------

if __name__ == "__main__":
    print("=== HEX 11x11 (easyAI, Negamax, fast) ===")
    depth = ask_depth()
    who = ask_who_starts()

    ai_algo = Negamax(depth)
    game = (
        HexGame([Human_Player(), VerboseAI(ai_algo)])
        if who == 1
        else HexGame([VerboseAI(ai_algo), Human_Player()])
    )

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
