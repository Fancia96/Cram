import numpy as np
from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax

"""
    Cram to gra, ktora polega na kladzeniu kosci Domino na planszy
    Przegrywa osoba ktora nie moze wiecej polozyc kosci na planszy
    Zasady gry: https://en.wikipedia.org/wiki/Cram_(game)
     - Ruchy wykonuje sie poprzez podanie koordynatow na podstawie wystwietlonej planszy w konsoli
     - Podane koordynaty musza ze soba sasiadowac
     - Przykladowymi ruchami jest "A1 A2", "A3 B3"


    Jak przygotwac srodowisko do gry:
     - Zainstalowac numpy - pip install numpy
     - Zainstalowac easyAI - pip install easyai

    Autorzy:
     - Kamil Rominski(s20599)
     - Wanda Bojanowska(s18425)
"""

"""
Zmienne sluzace do zamiany ruchu na koordynaty i odwrotnie
"""

str2poz = lambda s: ["ABCDEFGH".index(s[0]), int(s[1]) - 1]
poz2str = lambda a: "ABCDEFGH"[a[0]] + str(a[1] + 1)

ruch2str = lambda m: poz2str((m[0], m[1])) + " " + poz2str((m[2], m[3]))


def string2ruch(s):
    """Funkcja do zamiany ruchu w formie tekstowej na adresy listy
    Parametry:
     -  s(string): ruch podany przez gracza w formie tekstowej
    Zwraca: zamieniony ruch podany przez gracza w formie zrozumialej przez pozostale funkcje
    """
    pozycja = [str2poz(p) for p in s.split(" ")]
    return pozycja[0] + pozycja[1]


class Cram(TwoPlayerGame):
    """Glowna klasa gry
    link: https://en.wikipedia.org/wiki/Cram_(game)

    Argumenty:
        - TwoPlayerGame: klasa przyjmująca za argumenty rodzaje graczy, na przykład człowiek vs człowiek, człowiek vs ai itd
    """

    def __init__(self, players):
        """Funkcja inicjujaca gre oraz zmienne i jej parametry

        Argumenty:
         - players: parametr pobierajacy graczy z klasy
        """
        self.players = players
        self.board_size = (6, 6)
        self.board = np.zeros(self.board_size, dtype=int)
        self.current_player = 1  # player 1 starts.

    def possible_moves(self):
        """Funkcja ktora wylicza mozliwe ruchy do wykonania przez gracza

        Zwraca: lista mozliwych ruchow do wykonania
        """
        ruchy = []
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                if self.board[i, j] == 0:
                    if (i + 1) < self.board_size[0] and self.board[i + 1, j] == 0:
                        ruchy.append([i, j, i + 1, j])
                    if (j + 1) < self.board_size[1] and self.board[i, j + 1] == 0:
                        ruchy.append([i, j, i, j + 1])
        return list(map(ruch2str, ruchy))

    def make_move(self, ruch):
        """Funkcja na podstawie danych z argumentu ruch oznacza koordynaty jako zajete

        Argumenty
         - ruch(list): lista z koordynatami wybranego ruchu przez gracza
        """
        ruch = string2ruch(ruch)
        self.board[ruch[0], ruch[1]] = 1
        self.board[ruch[2], ruch[3]] = 1

    def unmake_move(self, ruch):
        """"Funkcja na podstawie danych z argumentu ruch oznacza koordynaty jako wolne, uzywane przez AI do szybszych dzialan

        Argumenty
         - ruch(list): lista z koordynatami wybranego ruchu przez gracza

        """
        ruch = string2ruch(ruch)
        self.board[ruch[0], ruch[1]] = 0
        self.board[ruch[2], ruch[3]] = 0

    def show(self):
        """Funkcja Wypisuje tablice gry z wykonanymi i dostepnymi ruchami w postaci tekstowej w konsoli

        Wolne pola sa oznaczone "."

        Zajete pola sa oznaczone "*"

        ABCDEF, 123456 - koordynaty do wskazania ruchu przez gracza
        """
        print(
            "\n"
            + "\n".join(
                ["  1 2 3 4 5 6"]
                + [
                    "ABCDEF"[k]
                    + " "
                    + " ".join(
                        [".*"[self.board[k, i]] for i in range(self.board_size[0])]
                    )
                    for k in range(self.board_size[1])
                ]
                + [""]

            )
        )

    def lose(self):
        """Funkcja opisuje warunki przegrania gry

        Zwraca:
        boolean: wartosc true jezeli nie ma wiecej ruchow, w przeciwnym wypadku false

        """
        return self.possible_moves() == []

    def scoring(self):
        """Funkcja sluzaca oceny wygranej lub przegranej. Zwraca -100 dla przegranego gracza, 0 dla wygranego

        Zwraca:
         - int: Wynik gry

        """

        return -100 if self.lose() else 0

    def is_over(self):
        """Funkcja sprawdzajaca warunki zakonczenia gry

        Zwraca:
         - boolean: wartosc funkcji lose()

        """
        return self.lose()


ai = Negamax(6)  # wybor algorytmu do AI
game = Cram([Human_Player(), AI_Player(ai)])  # Inicjalizacja gry
game.play()  # rozpoczecie gry
print("Gracz %d przegrywa" % game.current_player)  # drukuje napis koncowy gry
