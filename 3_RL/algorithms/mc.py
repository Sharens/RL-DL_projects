"""
Monte Carlo (model-free) dla środowisk dyskretnych.
- prediction: V(s) z pełnych epizodów
- control: first-visit MC control (ε-greedy)
"""

from __future__ import annotations
from typing import Callable, Tuple, Optional
import numpy as np
from .utils import rollout_episode, returns_from_trajectory, epsilon_greedy


def mc_prediction(env, policy: Callable[[int], int], nS: int, gamma: float = 0.99, episodes: int = 10_000, max_steps: int = 10_000):
    V = np.zeros(nS, dtype=np.float64)
    N = np.zeros(nS, dtype=np.int64)

    for _ in range(episodes):
        traj = rollout_episode(env, policy, max_steps=max_steps)

        # TODO: policz zwroty G_t dla każdego kroku (użyj helpera returns_from_trajectory z utils)
        Gs = returns_from_trajectory(traj, gamma)
        visited = set()
        for (t, (s, a, r)) in enumerate(traj):
            if s in visited: continue
            visited.add(s)
            G = Gs[t]
            N[s] += 1
            V[s] += (G - V[s]) / N[s]

        visited = ...  # TODO: stwórz zbiór visited do first-visit (żeby aktualizować stan tylko raz na epizod)

        for (t, (s, a, r)) in enumerate(traj):
            # TODO: first-visit MC: jeśli stan s już był w tym epizodzie, pomiń aktualizację
            if ...:
                continue  # first-visit (aktualizujemy tylko przy pierwszym wystąpieniu stanu w epizodzie)

            visited.add(...)  # TODO: dodaj bieżący stan s do visited (oznaczamy, że już był)

            G = ...  # TODO: wybierz zwrot dla kroku t (to jest target do aktualizacji V[s])

            N[s] += ...  # TODO: zwiększ licznik wizyt stanu s (potrzebny do średniej inkrementalnej)

            # TODO: aktualizacja średniej inkrementalnej V[s] zgodnie ze wzorem: V <- V + (G - V)/N
            V[s] += ...  # aktualizujemy estymatę wartości stanu s na podstawie zwrotu G

    return V, N

def mc_control_epsilon_greedy(
    env, nS, nA, gamma=0.99, episodes=10_000, epsilon=0.1, seed=0, max_steps=10_000
):
    rng = np.random.default_rng(seed)
    Q = np.zeros((nS, nA), dtype=np.float64)
    N = np.zeros((nS, nA), dtype=np.int64)

    # Naprawiona funkcja policy
    def policy(s: int) -> int:
        return epsilon_greedy(Q[s], epsilon, rng)

    for _ in range(episodes):
        # 1. Generujemy trajektorię (rollout)
        traj = rollout_episode(env, policy, max_steps=max_steps)

        # 2. Liczymy zwroty G_t dla każdego kroku
        Gs = returns_from_trajectory(traj, gamma)

        # 3. First-visit MC control
        visited = set()
        for (t, (s, a, r)) in enumerate(traj):
            key = (s, a)
            if key in visited:
                continue  # Pomijamy kolejne wizyty w tej samej parze (s, a)
            
            visited.add(key)
            G = Gs[t]
            
            # Inkrementalna średnia dla Q(s, a)
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / N[s, a]

    # Wyciągamy optymalną politykę (zawsze wybierz najlepszą akcję)
    pi_greedy = np.argmax(Q, axis=1)
    return Q, pi_greedy, N
