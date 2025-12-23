import numpy as np

def build_reward_matrix():
    """
    Beispiel: 12 Orte (0..11).
    - 0 = blockiert (kein direkter Pfad), 1 = erlaubte Aktion, 1000 = Zielzustand
    Du passt dieses Raster an dein Warehouse an.
    """
    n = 12
    R = np.zeros((n, n), dtype=float)

    # Beispielhafte Kanten (ungerichtet) – passe an deine Warehouse-Topologie an
    edges = [
        (0, 1), (1, 2), (2, 3),
        (3, 7), (7, 11),           # Pfad Richtung Ziel 11
        (1, 5), (5, 6), (6, 10),   # alternative Routen
        (2, 6), (3, 4), (4, 8), (8, 9), (9, 10), (10, 11)
    ]

    for a, b in edges:
        R[a, b] = 1.0
        R[b, a] = 1.0

    goal = 11
    # Zielprämien: Bewegung DIREKT ins Ziel mit hoher Belohnung
    for s in range(n):
        if R[s, goal] > 0:
            R[s, goal] = 1000.0

    return R
