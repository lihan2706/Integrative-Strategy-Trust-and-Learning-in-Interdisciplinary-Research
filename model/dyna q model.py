"""
Second-Order Dyna-Q Agent Experiment Suite
============================================================
Paper: Computational Modeling of Cross-disciplinary Scientific Collaboration
       (Second-order agent extension)

Fully self-contained - no external rl_agent.py required.

Experiments
-----------
Exp A  Multi-seed robustness check (8 seeds)
Exp B  Individual vs team reward (8 seeds + t-test)
Exp C  Initial bias sweep (bold_prior 0% to 100%)
Exp D  Dyna-Q vs first-order fixed strategies (significance tests)
Exp E  State coverage growth analysis

Usage
-----
python dyna_q_fixed.py                # run all experiments
python dyna_q_fixed.py --exp A        # run experiment A only
python dyna_q_fixed.py --exp B        # run experiment B only

Dependencies
------------
numpy, matplotlib, scipy  (no rl_agent.py needed)

Bug-fix changelog (vs dyna_q_experiments_full.py)
--------------------------------------------------
Fix 1 - DynaQAgent.search(): `cur` changed from max(in_hist) to self._prev.
    max(in_hist) returns the trust-group best, which is effectively the team
    maximum.  Using it as the individual reward baseline collapses the gradient
    to (own_score - team_best) <= 0 for all rounds after the first, destroying
    the learning signal.  self._prev is the agent's own last score.

Fix 2 - count_unique_visited() helper replaces the raw summing loop used in
    both run_one() and run_exp_E().  The old loop counted each (agent, state)
    pair separately, so one state visited by k agents was counted k times and
    coverage percentages could exceed 100 %.  The helper deduplicates via a set.

Fix 3 - Agent.search(): documented the `findings = own_hist` alias.  The alias
    means the `.update()` call in RLTeam.tournament() is a no-op.  Added clear
    comments to prevent future developers from "fixing" the alias and silently
    breaking the shared-memory mechanism.

Fix 4 - greedy_ratio(): rewritten to count votes per *unique state* (majority /
    mean-Q across agents) instead of per (agent, state) pair.  The old version
    let k agents visiting the same state cast k votes, inflating whichever
    label they agreed on and double-counting disagreements.

Fix 5 - DynaQAgent.model: stores a running-mean reward instead of overwriting
    with the latest observation.  With sigma=10, a single noisy sample can
    corrupt the replay buffer; incremental averaging gives stable Dyna targets.
    The tuple format changes from (reward, next_state) to
    (mean_reward, next_state, count); all consumers updated accordingly.
"""

# -----------------------------------------------------------------------------
# 0. Imports
# -----------------------------------------------------------------------------
import argparse
import json
import math
import os
import random
import sys
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from scipy import stats

# -- Plot style: matches the PPT white/clean academic theme --------------
def _apply_ax_style(ax):
    """Apply publication-ready style to a single Axes object."""
    ax.set_facecolor('white')
    ax.grid(alpha=0.30, ls='--', color='#CCCCCC', zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#444444')

SUMMARY_BOX = dict(boxstyle='round,pad=0.4',
                        facecolor='#F5F5F5',
                        edgecolor='#BBBBBB', lw=1.0)
# -----------------------------------------------------------------------------

# Output directory: folder of this file, or cwd when run from Jupyter
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()

# -----------------------------------------------------------------------------
# 1. Landscape classes  (from original notebook)
# -----------------------------------------------------------------------------
class LandscapeComplex:
    """Rugged landscape with many local optima (smoothness ~ 4)."""
    def __init__(self, smoothness=4, length=2000, seed=None):
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        self.s = smoothness
        total_segments = round(length / self.s)
        points = [random.choice(range(1, 101))]
        heights = []
        for _ in range(total_segments - 1):
            a = points[-1]
            b = random.choice(range(1, 101))
            points.append(b)
            seg_len = random.choice(range(1, 2 * self.s))
            step = round((b - a) / seg_len, 2)
            for i in range(1, seg_len + 1):
                val = a + step * i
                heights.append(100 if val > 100 else val)
        self.heights = heights
        self.length  = len(self.heights)


class LandscapeSimple:
    """Smooth landscape with few broad peaks (smoothness ~ 30)."""
    def __init__(self, smoothness=30, length=2000, seed=None):
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        self.s = smoothness
        total_segments = round(length / self.s)
        points = [random.choice(range(1, 101))]
        heights = []
        for _ in range(total_segments - 1):
            a = points[-1]
            b = random.choice(range(1, 101))
            points.append(b)
            seg_len = random.choice(range(1, 2 * self.s))
            step = round((b - a) / seg_len, 2)
            for i in range(1, seg_len + 1):
                val = a + step * i
                heights.append(100 if val > 100 else val)
        self.heights = heights
        self.length  = len(self.heights)


# -----------------------------------------------------------------------------
# 2. Base Agent  (dict-based in_hist; returns findings, rounds)
# -----------------------------------------------------------------------------
class Agent:
    """
    First-order agent with a fixed step-size heuristic.

    Parameters
    ----------
    no        : agent id
    h         : step-size tuple, e.g. (1, 2, 3)
    landscape : LandscapeComplex or LandscapeSimple
    sigma     : observation noise std
    """
    def __init__(self, no, h, landscape, sigma=0):
        self.no        = no
        self.h         = h
        self.landscape = landscape
        self.sigma     = sigma

    def data(self, loc):
        """Noisy observation at position loc."""
        return max(np.random.normal(self.landscape.heights[loc], self.sigma),
                   0.001)

    def search(self, start, own_hist, in_hist):
        """
        Hill-climb from `start`.

        Parameters
        ----------
        start    : integer start position
        own_hist : dict {pos: score}  - agent's own memory (updated in-place)
        in_hist  : dict {pos: score}  - shared knowledge from trusted teammates

        Returns
        -------
        findings : dict {pos: score}
        rounds   : int  (number of steps taken)
        """
        loc      = start
        # NOTE (Fix 3): `findings` is intentionally an alias for `own_hist`.
        # Agent.search() updates `own_hist` in-place as it explores, so the
        # caller's dict is mutated directly.  The `indiv_hists[agent].update(findings)`
        # call in RLTeam.tournament() is therefore a no-op - it exists only for
        # readability.  Do NOT replace this alias with a copy unless you also
        # remove the in-place mutations below, or you will silently break the
        # information-sharing mechanism.
        findings = own_hist
        # Initialise current position
        maxi = in_hist.get(loc) or self.data(loc)
        findings[loc] = maxi

        count   = 0
        n_total = 0
        while count < len(self.h):
            nxt = (loc + self.h[n_total % len(self.h)]) % self.landscape.length
            if nxt not in in_hist:
                value = self.data(nxt)
                findings[nxt]  = value
                in_hist[nxt]   = value
            else:
                value = in_hist[nxt]

            if maxi < value:
                loc, maxi, count = nxt, value, 0
            else:
                count += 1
            n_total += 1

        return findings, n_total


# -----------------------------------------------------------------------------
# 3. Action space: Bold / Conservative only (Balanced removed)
# -----------------------------------------------------------------------------
ACTION_SPACE = [
    # Conservative (avg step <= 5)
    (1, 2, 3),
    (2, 3, 4),
    (3, 4, 5),
    (4, 5, 6),
    # Bold (avg step >= 9)
    (8, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
]
N_ACTIONS = len(ACTION_SPACE)
BOLD_IDX  = [4, 5, 6]
CONS_IDX  = [0, 1, 2, 3]

def _classify(h):
    return "Bold" if np.mean(h) >= 9 else "Conservative"

ACTION_LABELS = [_classify(h) for h in ACTION_SPACE]

# Fixed-strategy mapping (used in Exp D)
FIXED_HEURISTICS = {
    'AllConservative': [(1,2,3),(2,3,4),(3,4,5),(4,5,6)],
    'AllBold':         [(8,9,10),(9,10,11),(10,11,12)],
    'Mix78Bold':       [(8,9,10)]*7 + [(1,2,3)]*2,   # paper-optimal mix
}

# Color scheme
COLORS      = {'Bold': '#E67E22', 'Conservative': '#1ABC9C'}
PK_COLOR    = {'complex': '#C0392B', 'simple': '#2471A3'}
MODE_COLOR  = {'individual': '#E67E22', 'team': '#1ABC9C'}


# -----------------------------------------------------------------------------
# 4. State encoding: 10x8x3x5 = 1200 states
# -----------------------------------------------------------------------------
N_STATES = 1200

def encode_state(cur, prev, pp, team_best):
    """
    Discretize continuous observations into Q-table indices.

    Dimensions
    ----------
    s_bin  : score bucket (10)           0-100 -> 0-9
    d_bin  : step improvement bucket (8) [-20,+20] -> 0-7
    t_bin  : trend (3)                   accelerating/flat/decelerating
    b_bin  : gap to team best (5)        [0,40+] -> 0-4
    """
    s_bin = min(int(cur / 100 * 10), 9)
    d_bin = min(int((cur - prev + 20) / 40 * 8), 7)
    d_bin = max(d_bin, 0)
    trend = (cur - prev) - (prev - pp)
    t_bin = 2 if trend > 2 else (0 if trend < -2 else 1)
    b_bin = min(int(max(team_best - cur, 0) / 40 * 5), 4)
    return (s_bin, d_bin, t_bin, b_bin)


# -----------------------------------------------------------------------------
# 5. Dyna-Q second-order agent
# -----------------------------------------------------------------------------
class DynaQAgent(Agent):
    """
    Dyna-Q second-order agent.

    Second-order features
    ---------------------
    * Observe score change trend -> encode as state (1200-dim)
    * Epsilon-greedy choice of Bold/Conservative action
    * After each real search, replay dyna_k transitions (accelerates learning)
    * Supports two reward functions:
        'individual' -- own score improvement
        'team'       -- team-best improvement (socially oriented objective)

    Parameters
    ----------
    action_prior : probability vector over actions, sampled during exploration
    reward_mode  : 'individual' or 'team'
    """
    def __init__(self, no, landscape, sigma=0,
                 alpha=0.1, gamma=0.9,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9999,
                 dyna_k=15, action_prior=None, reward_mode='individual'):
        super().__init__(no, ACTION_SPACE[0], landscape, sigma)
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.eps_end     = epsilon_end
        self.eps_decay   = epsilon_decay
        self.dyna_k      = dyna_k
        self.reward_mode = reward_mode

        if action_prior is not None:
            self.action_prior = np.array(action_prior, dtype=float)
            self.action_prior /= self.action_prior.sum()
        else:
            self.action_prior = np.ones(N_ACTIONS) / N_ACTIONS

        self.Q      = defaultdict(lambda: np.zeros(N_ACTIONS))
        self.model  = {}        # (state, action) -> (mean_reward, next_state, count)
        # FIX 5: We store a running mean of observed rewards for each (s,a) pair
        # instead of overwriting with the latest sample.  With sigma=10, a single
        # noisy observation can completely corrupt the model; averaging across
        # visits gives a much more stable Dyna replay target.

        self._prev           = 0.
        self._pp             = 0.
        self._team_best      = 0.
        self._prev_team_best = 0.

        self.score_history  = []
        self.action_history = []

    def _q_update(self, s, a, r, ns):
        td = r + self.gamma * np.max(self.Q[ns]) - self.Q[s][a]
        self.Q[s][a] += self.alpha * td

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return int(np.random.choice(N_ACTIONS, p=self.action_prior))
        return int(np.argmax(self.Q[state]))

    def search(self, start, own_hist, in_hist, team_best=None):
        if team_best is not None:
            self._team_best = team_best

        # FIX 1: Use the agent's own previous score as the baseline for computing
        # the reward signal, not max(in_hist). in_hist contains the entire trust
        # group's memory, so its maximum is the team best - using it as `cur`
        # collapses individual reward to (own_new - team_best) which is almost
        # always <= 0, effectively killing the Q-learning gradient after round 1.
        cur   = self._prev
        state = encode_state(cur, self._prev, self._pp, self._team_best)
        ai    = self.choose_action(state)
        self.h = ACTION_SPACE[ai]

        findings, rounds = super().search(start, own_hist, in_hist)
        new_score = max(findings.values()) if findings else cur

        # reward function
        if self.reward_mode == 'individual':
            reward = new_score - cur
        else:
            new_team = max(new_score, self._team_best)
            reward   = new_team - self._prev_team_best
            self._prev_team_best = new_team

        ns = encode_state(new_score, cur, self._prev, self._team_best)
        self._q_update(state, ai, reward, ns)

        # FIX 5: Update the world model using a running mean reward so that
        # repeated visits to the same (state, action) pair are averaged rather
        # than overwritten.  This makes Dyna replay much more stable under
        # high observation noise (sigma=10).
        if (state, ai) in self.model:
            r_old, ns_old, cnt = self.model[(state, ai)]
            # Use incremental mean: mean_new = mean_old + (r - mean_old) / (cnt+1)
            new_cnt = cnt + 1
            r_mean  = r_old + (reward - r_old) / new_cnt
            # next_state: keep the most recent (next-state transitions are deterministic
            # given the landscape, so overwriting is fine here)
            self.model[(state, ai)] = (r_mean, ns, new_cnt)
        else:
            self.model[(state, ai)] = (reward, ns, 1)

        # Dyna-Q replay
        if len(self.model) >= 2:
            for (s, a) in random.sample(list(self.model.keys()),
                                         min(self.dyna_k, len(self.model))):
                r2, ns2, _ = self.model[(s, a)]   # unpack running-mean tuple
                self._q_update(s, a, r2, ns2)

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        self._pp     = self._prev
        self._prev   = new_score
        self.score_history.append(new_score)
        self.action_history.append(ai)
        return findings, rounds

    def reset_memory(self):
        self._prev = self._pp = self._team_best = self._prev_team_best = 0.


# -----------------------------------------------------------------------------
# 6. Extended RLTeam (supports DynaQAgent)
# -----------------------------------------------------------------------------
class RLTeam:
    """Team supporting mixed RL/fixed-strategy agents with tournament search."""
    def __init__(self, members, landscape, trust_level=0.7):
        self.members     = members
        self.landscape   = landscape
        self.trust_level = trust_level
        self._build_trust_groups()

    def _build_trust_groups(self):
        self.trust_map = {a: [a] for a in self.members}
        if self.trust_level > 0:
            M = self.members[:]
            random.shuffle(M)
            k = math.ceil(len(self.members) * self.trust_level)
            while M:
                grp = M[:k]; M = M[k:]
                for a in grp:
                    self.trust_map[a] = grp

    def tournament(self, start):
        shared_hist  = {}
        indiv_hists  = {a: {} for a in self.members}
        current_best = 0.
        converged    = False
        total_rounds = 0

        while not converged:
            new_findings = {}
            for agent in self.members:
                in_hist = {}
                for trusted in self.trust_map[agent]:
                    in_hist.update(indiv_hists[trusted])

                if isinstance(agent, DynaQAgent):
                    findings, _ = agent.search(
                        start, indiv_hists[agent], in_hist,
                        team_best=current_best)
                else:
                    findings, _ = agent.search(
                        start, indiv_hists[agent], in_hist)

                indiv_hists[agent].update(findings)  # no-op: findings IS indiv_hists[agent] (alias)
                new_findings.update(findings)

            shared_hist.update(new_findings)
            new_best = max(shared_hist.values()) if shared_hist else 0.
            if new_best <= current_best:
                converged = True
            current_best = new_best
            total_rounds += 1
            if total_rounds > 200:
                break

        return current_best, total_rounds

    def reset_rl_memory(self):
        for a in self.members:
            if hasattr(a, 'reset_memory'):
                a.reset_memory()


# -----------------------------------------------------------------------------
# 7. Global parameters
# -----------------------------------------------------------------------------
SIGMA    = 10       # observation noise
TRUST    = 0.7      # team trust level
N_STARTS = 4000     # search starts per seed
DYNA_K   = 15       # Dyna-Q replay steps
SEEDS    = [2029, 1234, 5678, 9999, 3141, 2718, 1111, 4242]
SEEDS_C  = SEEDS[:4]   # Exp C uses 4 seeds (saves time)

PROBLEMS = [
    ('complex', 4,  'Complex Problem (smoothness=4)'),
    ('simple',  30, 'Simple Problem (smoothness=30)'),
]

OUTPUT_DIR = _here   # figure output directory


# -----------------------------------------------------------------------------
# 8. Utility functions
# -----------------------------------------------------------------------------
def make_prior(bold_prob):
    """Generate N_ACTIONS-dim prior probability vector (allocated by bold_prob)."""
    p = np.zeros(N_ACTIONS)
    for i in BOLD_IDX:
        p[i] = bold_prob / len(BOLD_IDX)
    for i in CONS_IDX:
        p[i] = (1 - bold_prob) / len(CONS_IDX)
    return p

def count_unique_visited(agents):
    """
    Count how many distinct Q-table states have been visited across all agents.

    FIX 2 / FIX 3: The original code iterated over all agents and summed every
    non-zero entry, letting one state be counted up to 9 times (once per agent).
    This helper uses a set to deduplicate across agents so the result is in
    [0, N_STATES] and directly comparable to N_STATES.
    """
    visited = set()
    for ag in agents:
        for state, q in ag.Q.items():
            if not np.allclose(q, 0):
                visited.add(state)
    return len(visited)


def greedy_ratio(agents):
    """
    Compute the greedy-action distribution over all *uniquely* visited states.

    FIX 4: The original code counted (agent x state) pairs, so one state visited
    by k agents contributed k votes - inflating whichever label those k agents
    agreed on and double-counting disagreements.  We now take one vote per
    unique state: if agents disagree on a state, we use majority vote; ties
    go to the action with the highest mean Q-value across agents.
    """
    # Aggregate Q-values for each unique state across agents
    state_q_sum   = defaultdict(lambda: np.zeros(N_ACTIONS))
    state_q_count = defaultdict(int)
    for ag in agents:
        for state, q in ag.Q.items():
            if not np.allclose(q, 0):
                state_q_sum[state]   += q
                state_q_count[state] += 1

    counts = defaultdict(int)
    for state, q_sum in state_q_sum.items():
        mean_q = q_sum / state_q_count[state]
        counts[ACTION_LABELS[int(np.argmax(mean_q))]] += 1

    total = sum(counts.values())
    if total == 0:
        return {k: 0. for k in set(ACTION_LABELS)}
    return {k: counts[k] / total for k in set(ACTION_LABELS)}

def make_landscape(pk, sm, seed):
    return (LandscapeComplex(smoothness=sm, seed=seed)
            if pk == 'complex'
            else LandscapeSimple(smoothness=sm, seed=seed))

def run_one(pk, sm, sigma, trust, n_starts, bold_prob,
            dyna_k, seed, reward_mode='individual'):
    """
    Run a single experiment.

    Returns
    -------
    ratio    : {'Bold': float, 'Conservative': float}  greedy policy proportions
    mean_sc  : float  mean score over all starts
    last_sc  : float  mean score over last 1000 starts
    visited  : int    number of visited states
    """
    np.random.seed(seed); random.seed(seed)
    L      = make_landscape(pk, sm, seed)
    prior  = make_prior(bold_prob)
    agents = [DynaQAgent(i, L, sigma, dyna_k=dyna_k,
                          action_prior=prior,
                          reward_mode=reward_mode) for i in range(9)]
    team   = RLTeam(agents, L, trust_level=trust)
    rng    = np.random.default_rng(seed)
    starts = rng.integers(0, L.length, size=n_starts).tolist()

    scores = []
    for s in starts:
        team.reset_rl_memory()
        val, _ = team.tournament(s)
        scores.append(val)

    ratio   = greedy_ratio(agents)
    # FIX 2a: Count unique visited states across all agents (not agentxstate pairs).
    # The old code summed over all agents separately, allowing one state to be
    # counted 9 times - making coverage appear to exceed 100%.
    visited = count_unique_visited(agents)
    return ratio, float(np.mean(scores)), float(np.mean(scores[-1000:])), visited

def run_fixed(heuristics, pk, sm, sigma, n_starts, seed):
    """Run a first-order fixed-strategy team; return mean score over last 1000 starts."""
    np.random.seed(seed); random.seed(seed)
    L       = make_landscape(pk, sm, seed)
    members = [Agent(i, heuristics[i % len(heuristics)], L, sigma)
               for i in range(9)]

    class _FixedTeam:
        def __init__(self):
            self.members   = members
            self.landscape = L
            trust_level    = TRUST
            k = math.ceil(len(members) * trust_level)
            M = members[:]
            random.shuffle(M)
            self.trust = {a: [a] for a in members}
            while M:
                grp = M[:k]; M = M[k:]
                for a in grp:
                    self.trust[a] = grp

        def tournament(self, start):
            shared = {}; best = 0.; rounds = 0; converged = False
            indiv  = {a: {} for a in self.members}
            while not converged:
                new = {}
                for a in self.members:
                    ih = {}
                    for t in self.trust[a]:
                        ih.update(indiv[t])
                    f, _ = a.search(start, indiv[a], ih)
                    indiv[a].update(f); new.update(f)
                shared.update(new)
                nb = max(shared.values()) if shared else 0.
                if nb <= best:
                    converged = True
                best = nb; rounds += 1
                if rounds > 200: break
            return best, rounds

        def reset_rl_memory(self): pass

    team   = _FixedTeam()
    rng    = np.random.default_rng(seed)
    starts = rng.integers(0, L.length, size=n_starts).tolist()
    scores = []
    for s in starts:
        val, _ = team.tournament(s)
        scores.append(val)
    return float(np.mean(scores[-1000:]))

def sig_str(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=180, bbox_inches='tight')
    print(f'  [saved] {path}')
    plt.close()


# -----------------------------------------------------------------------------
# 9. Experiment A: Multi-seed robustness
# -----------------------------------------------------------------------------
def run_exp_A():
    print('\n' + '='*60)
    print('Experiment A: Multi-seed Robustness (bold_prior=0.78)')
    print('='*60)

    res = {}
    for pk, sm, label in PROBLEMS:
        bold_list = []; score_list = []
        for seed in SEEDS:
            rb, _, last_sc, vis = run_one(pk, sm, SIGMA, TRUST,
                                           N_STARTS, 0.78, DYNA_K, seed)
            bold_list.append(rb['Bold'])
            score_list.append(last_sc)
            print(f'  [{pk}] seed={seed}  bold={rb["Bold"]:.1%}  '
                  f'score={last_sc:.2f}  vis={vis}')
        res[pk] = dict(ba=bold_list, sa=score_list,
                       bm=float(np.mean(bold_list)),
                       bs=float(np.std(bold_list)),
                       sm=float(np.mean(score_list)),
                       ss=float(np.std(score_list)))
        print(f'  [{pk}] SUMMARY  bold={res[pk]["bm"]:.1%}+/-{res[pk]["bs"]:.1%}  '
              f'score={res[pk]["sm"]:.2f}+/-{res[pk]["ss"]:.2f}\n')

    _plot_A(res)
    return res

def _plot_A(res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle('Experiment A: Multi-seed Robustness\n'
                 f'Dyna-Q (Bold/Conservative only, {N_STATES} states, '
                 f'n_starts={N_STARTS}, Dyna-K={DYNA_K})',
                 fontsize=13, fontweight='bold', color='#222222')

    for ax, (pk, sm, label) in zip(axes, PROBLEMS):
        d     = res[pk]
        bolds = d['ba']; cons_list = [1-b for b in bolds]
        x     = np.arange(len(bolds)); w = 0.35

        b1 = ax.bar(x-w/2, bolds,     w, color=COLORS['Bold'],
                    alpha=0.85, edgecolor='white', lw=0.5, label='Bold')
        b2 = ax.bar(x+w/2, cons_list, w, color=COLORS['Conservative'],
                    alpha=0.85, edgecolor='white', lw=0.5, label='Conservative')

        ax.axhline(d['bm'], color=COLORS['Bold'], ls='--', lw=1.8, alpha=0.80,
                   label=f'Bold mean={d["bm"]:.1%}+/-{d["bs"]:.1%}')
        ax.axhline(1-d['bm'], color=COLORS['Conservative'], ls='--', lw=1.8, alpha=0.80,
                   label=f'Cons mean={1-d["bm"]:.1%}')
        ax.axhline(0.78, color='#AAAAAA', ls=':', lw=1.5, alpha=0.7,
                   label='Initial prior 78%')

        for bar, v in zip(b1, bolds):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.008,
                    f'{v:.0%}', ha='center', va='bottom', fontsize=8.5, color='#333333')
        for bar, v in zip(b2, cons_list):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.008,
                    f'{v:.0%}', ha='center', va='bottom', fontsize=8.5, color='#333333')

        ax.set_xlim(-0.6, len(bolds)-0.4); ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Seed\n{s}' for s in SEEDS[:len(bolds)]], fontsize=8.5)
        ax.set_ylabel('Greedy Policy Proportion', fontsize=11)
        ax.set_title(label, fontsize=11, fontweight='bold', color=PK_COLOR[pk])
        ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
        _apply_ax_style(ax)

        summary = (f'Bold:   {d["bm"]:.1%} +/- {d["bs"]:.1%}\n'
                   f'Cons:   {1-d["bm"]:.1%} +/- {np.std(cons_list):.1%}\n'
                   f'Initial prior: 78%\n'
                   f'Drift:  {d["bm"]-0.78:+.1%}\n'
                   f'Score:  {d["sm"]:.2f} +/- {d["ss"]:.2f}')
        ax.text(0.02, 0.38, summary, transform=ax.transAxes,
                fontsize=9, va='top', family='monospace',
                bbox=SUMMARY_BOX)

    plt.tight_layout()
    savefig('expA_plot.png')


# -----------------------------------------------------------------------------
# 10. Experiment B: Individual vs team reward (8 seeds + t-test)
# -----------------------------------------------------------------------------
def run_exp_B():
    print('\n' + '='*60)
    print('Experiment B: Individual vs Team Reward (8 seeds)')
    print('='*60)

    res = {}
    for pk, sm, label in PROBLEMS:
        res[pk] = {m: {'ba': [], 'sa': []} for m in ['individual', 'team']}
        for mode in ['individual', 'team']:
            for seed in SEEDS:
                rb, _, last_sc, _ = run_one(pk, sm, SIGMA, TRUST,
                                             N_STARTS, 0.78, DYNA_K, seed,
                                             reward_mode=mode)
                res[pk][mode]['ba'].append(rb['Bold'])
                res[pk][mode]['sa'].append(last_sc)
            bm = float(np.mean(res[pk][mode]['ba']))
            bs = float(np.std(res[pk][mode]['ba']))
            sm_ = float(np.mean(res[pk][mode]['sa']))
            res[pk][mode].update(bm=bm, bs=bs, sm=sm_)
            print(f'  [{pk}] {mode:12s}  bold={bm:.1%}+/-{bs:.1%}  '
                  f'score={sm_:.2f}')

        # t-tests
        t_b, p_b = stats.ttest_ind(res[pk]['individual']['ba'],
                                   res[pk]['team']['ba'])
        t_s, p_s = stats.ttest_ind(res[pk]['individual']['sa'],
                                   res[pk]['team']['sa'])
        res[pk]['ttest_bold']  = {'t': t_b, 'p': p_b}
        res[pk]['ttest_score'] = {'t': t_s, 'p': p_s}
        print(f'  [{pk}] bold t={t_b:.2f} p={p_b:.4f} {sig_str(p_b)}\n')

    _plot_B(res)
    return res

def _plot_B(res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('white')
    fig.suptitle('Experiment B: Individual vs Team Reward\n'
                 'Does the reward function determine what strategy is learned?',
                 fontsize=13, fontweight='bold', color='#222222')

    for ax, (pk, sm, label) in zip(axes, PROBLEMS):
        d     = res[pk]
        modes = ['individual', 'team']
        bp_data = [d[m]['ba'] for m in modes]

        vp = ax.violinplot(bp_data, positions=[0,1], widths=0.5,
                            showmeans=True, showmedians=False)
        for body, m in zip(vp['bodies'], modes):
            body.set_facecolor(MODE_COLOR[m]); body.set_alpha(0.5)
        vp['cmeans'].set_color('black'); vp['cmeans'].set_linewidth(2)
        for part in ['cbars','cmins','cmaxes']:
            vp[part].set_color('black'); vp[part].set_linewidth(1.2)

        rng = np.random.RandomState(7)
        for xi, m in enumerate(modes):
            jit = rng.uniform(-0.07, 0.07, len(SEEDS))
            ax.scatter(xi+jit, d[m]['ba'], color=MODE_COLOR[m],
                       edgecolors='black', s=60, zorder=4, lw=0.7)

        ax.axhline(0.78, color='#AAAAAA', ls=':', lw=1.5, alpha=0.7,
                   label='Initial prior 78%')

        tt = d['ttest_bold']; p = tt['p']
        y_top = max(max(d['individual']['ba']), max(d['team']['ba'])) + 0.04
        ax.plot([0,1],[y_top,y_top], color='#444444', lw=1.5)
        ax.plot([0,0],[y_top-0.01,y_top], color='#444444', lw=1.5)
        ax.plot([1,1],[y_top-0.01,y_top], color='#444444', lw=1.5)
        ax.text(0.5, y_top+0.01, f't={tt["t"]:.2f}, p={p:.4f} {sig_str(p)}',
                ha='center', fontsize=9.5, fontweight='bold',
                color='#27AE60' if p < 0.05 else '#AAAAAA')

        for xi, m in enumerate(modes):
            ax.text(xi, d[m]['bm']-0.06,
                    f'{d[m]["bm"]:.1%}+/-{d[m]["bs"]:.1%}',
                    ha='center', fontsize=9, fontweight='bold',
                    color=MODE_COLOR[m])

        ax.set_xticks([0,1])
        ax.set_xticklabels(['Individual\nReward', 'Team\nReward'], fontsize=11)
        ax.set_ylim(0.1, y_top+0.08)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
        ax.set_ylabel('Final Greedy Bold Proportion', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold', color=PK_COLOR[pk])
        ax.legend(fontsize=8.5, framealpha=0.9)
        _apply_ax_style(ax)

        ind  = d['individual']; team = d['team']
        diff = team['bm'] - ind['bm']
        tt2  = d['ttest_score']
        summary = (f'Individual: {ind["bm"]:.1%} +/- {ind["bs"]:.1%}\n'
                   f'Team:       {team["bm"]:.1%} +/- {team["bs"]:.1%}\n'
                   f'Delta bold: {diff:+.1%}\n'
                   f'Score diff: p={tt2["p"]:.4f} {sig_str(tt2["p"])}\n'
                   f'Paper opt:  78%')
        ax.text(0.02, 0.99, summary, transform=ax.transAxes,
                fontsize=9, va='top', family='monospace',
                bbox=SUMMARY_BOX)

    plt.tight_layout()
    savefig('expB_plot.png')


# -----------------------------------------------------------------------------
# 11. Experiment C: Initial bias sweep
# -----------------------------------------------------------------------------
def run_exp_C():
    print('\n' + '='*60)
    print('Experiment C: Initial Bias Sweep (0% to 100%)')
    print('='*60)

    BIAS_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                   0.6, 0.7, 0.78, 0.9, 1.0]
    res = {}
    for pk, sm, label in PROBLEMS:
        res[pk] = {'bias':[], 'bm':[], 'bs':[], 'sm':[], 'ss':[]}
        for bp in BIAS_LEVELS:
            bolds = []; scores = []
            for seed in SEEDS_C:
                rb, _, last_sc, _ = run_one(pk, sm, SIGMA, TRUST,
                                             N_STARTS, bp, DYNA_K, seed)
                bolds.append(rb['Bold']); scores.append(last_sc)
            res[pk]['bias'].append(bp)
            res[pk]['bm'].append(float(np.mean(bolds)))
            res[pk]['bs'].append(float(np.std(bolds)))
            res[pk]['sm'].append(float(np.mean(scores)))
            res[pk]['ss'].append(float(np.std(scores)))
            print(f'  [{pk}] bias={bp:.0%}  bold={np.mean(bolds):.1%}+/-'
                  f'{np.std(bolds):.1%}  score={np.mean(scores):.2f}')
        print()

    _plot_C(res)
    return res

def _plot_C(res):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle('Experiment C: Initial Bias Sweep\n'
                 'Does the starting prior affect the final learned strategy?',
                 fontsize=13, fontweight='bold', color='#222222')

    # Row 0: Final Bold vs Bias (per problem)
    for ax, (pk, sm, label) in zip(axes[0], PROBLEMS):
        ax2 = ax.twinx()
        d   = res[pk]
        xs  = np.array(d['bias'])
        ym  = np.array(d['bm']); ys = np.array(d['bs'])
        sm_ = np.array(d['sm']); ss = np.array(d['ss'])
        color = PK_COLOR[pk]

        ax.plot(xs, ym, color=color, lw=2.5, marker='o', ms=7,
                label='Final greedy bold', zorder=3)
        ax.fill_between(xs, ym-ys, ym+ys, color=color, alpha=0.15)
        ax.plot([0,1],[0,1], color='#AAAAAA', ls=':', lw=1.5, alpha=0.7,
                label='No learning')
        conv = np.mean(ym)
        ax.axhline(conv, color=color, ls='--', lw=1.5, alpha=0.45,
                   label=f'Convergence~{conv:.1%}')

        ax2.plot(xs, sm_, color='#9B59B6', lw=2, marker='s', ms=5,
                 ls='--', label='Score (last 1k)', alpha=0.75)
        ax2.fill_between(xs, sm_-ss, sm_+ss, color='#9B59B6', alpha=0.08)
        ax2.set_ylabel('Score', fontsize=10, color='#9B59B6')
        ax2.tick_params(axis='y', labelcolor='#9B59B6')

        ax.set_xlabel('Initial Bold Prior', fontsize=10)
        ax.set_ylabel('Final Greedy Bold Proportion', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold', color=color)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 0.8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
        l1,lb1 = ax.get_legend_handles_labels()
        l2,lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, lb1+lb2, fontsize=8, framealpha=0.9)
        _apply_ax_style(ax)

    # Row 1 left: Overlay comparison
    ax3 = axes[1][0]
    for pk, sm, label in PROBLEMS:
        d = res[pk]
        ax3.plot(d['bias'], d['bm'], color=PK_COLOR[pk], lw=2.5,
                 marker='o', ms=7, label=label)
        ax3.fill_between(d['bias'],
                          np.array(d['bm'])-np.array(d['bs']),
                          np.array(d['bm'])+np.array(d['bs']),
                          color=PK_COLOR[pk], alpha=0.12)
        ax3.axhline(np.mean(d['bm']), color=PK_COLOR[pk],
                    ls='--', lw=1.2, alpha=0.4)

    ax3.plot([0,1],[0,1], color='#AAAAAA', ls=':', lw=1.5, alpha=0.7,
             label='No learning')
    ax3.set_xlabel('Initial Bold Prior', fontsize=10)
    ax3.set_ylabel('Final Greedy Bold Proportion', fontsize=10)
    ax3.set_title('Complex vs Simple: Convergence Comparison',
                  fontsize=11, fontweight='bold')
    ax3.set_xlim(-0.02,1.02); ax3.set_ylim(0, 0.8)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
    ax3.legend(fontsize=9, framealpha=0.9); _apply_ax_style(ax3)

    # Row 1 right: Score vs Bias
    ax4 = axes[1][1]
    for pk, sm, label in PROBLEMS:
        d = res[pk]
        ax4.plot(d['bias'], d['sm'], color=PK_COLOR[pk], lw=2.5,
                 marker='o', ms=7, label=label)
        ax4.fill_between(d['bias'],
                          np.array(d['sm'])-np.array(d['ss']),
                          np.array(d['sm'])+np.array(d['ss']),
                          color=PK_COLOR[pk], alpha=0.12)
        best_idx = int(np.argmax(d['sm']))
        ax4.axvline(d['bias'][best_idx], color=PK_COLOR[pk],
                    ls='--', lw=1.5, alpha=0.6,
                    label=f'Best bias {pk}={d["bias"][best_idx]:.0%}')

    ax4.set_xlabel('Initial Bold Prior', fontsize=10)
    ax4.set_ylabel('Score (last 1k starts)', fontsize=10)
    ax4.set_title('Score vs Initial Bias', fontsize=11, fontweight='bold')
    ax4.set_xlim(-0.02,1.02)
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
    ax4.legend(fontsize=8.5, framealpha=0.9); _apply_ax_style(ax4)

    plt.tight_layout()
    savefig('expC_plot.png')


# -----------------------------------------------------------------------------
# 12. Experiment D: Dyna-Q vs fixed strategies
# -----------------------------------------------------------------------------
def run_exp_D():
    print('\n' + '='*60)
    print('Experiment D: Dyna-Q vs Fixed Strategies')
    print('='*60)

    order = ['AllConservative', 'AllBold', 'Mix78Bold',
             'DynaQ-Individual', 'DynaQ-Team']
    res = {pk: {k: [] for k in order} for pk, *_ in PROBLEMS}

    for pk, sm, label in PROBLEMS:
        print(f'\n  [{pk}]')
        # fixed strategies
        for name, heuristics in FIXED_HEURISTICS.items():
            for seed in SEEDS:
                sc = run_fixed(heuristics, pk, sm, SIGMA, N_STARTS, seed)
                res[pk][name].append(sc)
            print(f'    {name:20s}  mean={np.mean(res[pk][name]):.2f}')

        # Dyna-Q both reward modes
        for mode, key in [('individual', 'DynaQ-Individual'),
                           ('team',       'DynaQ-Team')]:
            for seed in SEEDS:
                _, _, sc, _ = run_one(pk, sm, SIGMA, TRUST,
                                      N_STARTS, 0.78, DYNA_K, seed,
                                      reward_mode=mode)
                res[pk][key].append(sc)
            print(f'    {key:20s}  mean={np.mean(res[pk][key]):.2f}')

        # t-tests vs Mix78Bold
        for key in ['DynaQ-Individual', 'DynaQ-Team']:
            t, p = stats.ttest_ind(res[pk][key], res[pk]['Mix78Bold'])
            print(f'    {key} vs Mix78Bold: t={t:.2f} p={p:.4f} {sig_str(p)}')

    _plot_D(res, order)
    return res

def _plot_D(res, order):
    COLORS_D = ['#1ABC9C', '#E67E22', '#9B59B6', '#E84393', '#3498DB']
    LABELS_D  = {
        'AllConservative':  'All\nConservative',
        'AllBold':          'All\nBold',
        'Mix78Bold':        'Mix 78% Bold\n(Paper optimal)',
        'DynaQ-Individual': 'Dyna-Q\nIndividual',
        'DynaQ-Team':       'Dyna-Q\nTeam',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')
    fig.suptitle('Experiment D: Dyna-Q vs Fixed Strategies\n'
                 f'8 seeds x n_starts={N_STARTS}  sigma={SIGMA}  trust={TRUST}',
                 fontsize=13, fontweight='bold', color='#222222')

    for ax, (pk, sm, label) in zip(axes, PROBLEMS):
        d     = res[pk]
        means = [np.mean(d[k]) for k in order]
        stds  = [np.std(d[k])  for k in order]
        x     = np.arange(len(order))

        bars = ax.bar(x, means, 0.55, color=COLORS_D, alpha=0.85,
                      edgecolor='white', lw=0.5,
                      yerr=stds, capsize=7,
                      error_kw=dict(lw=2, capthick=2), zorder=2)

        rng = np.random.RandomState(0)
        for xi, key in enumerate(order):
            jitter = rng.uniform(-0.1, 0.1, len(d[key]))
            ax.scatter(xi+jitter, d[key], color='black',
                       alpha=0.45, s=30, zorder=4)

        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, m+s+0.4,
                    f'{m:.2f}', ha='center', fontsize=10, fontweight='bold')

        mix_mean = np.mean(d['Mix78Bold'])
        ax.axhline(mix_mean, color='#9B59B6', ls='--', lw=1.8, alpha=0.6,
                   label=f'Mix78Bold mean={mix_mean:.2f}')

        for qkey in ['DynaQ-Individual', 'DynaQ-Team']:
            xi = order.index(qkey)
            t, p = stats.ttest_ind(d[qkey], d['Mix78Bold'])
            y_top = max(np.mean(d[qkey])+np.std(d[qkey]),
                        mix_mean+np.std(d['Mix78Bold'])) + 1.5
            ax.plot([order.index('Mix78Bold'), xi],
                    [y_top-0.3, y_top-0.3], color='black', lw=1.2, alpha=0.6)
            ax.text((order.index('Mix78Bold')+xi)/2, y_top,
                    f'p={p:.4f} {sig_str(p)}',
                    ha='center', fontsize=9, fontweight='bold',
                    color='#27AE60' if p < 0.05 else '#AAAAAA')

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS_D[k] for k in order], fontsize=9.5)
        ax.set_ylabel('Score (mean of last 1000 starts)', fontsize=11)
        ax.set_title(label, fontsize=11, fontweight='bold', color=PK_COLOR[pk])
        ymin = min(means)-max(stds)-3; ymax = max(means)+max(stds)+5
        ax.set_ylim(ymin, ymax)
        patches = [mpatches.Patch(color=c, alpha=0.85, label=LABELS_D[k])
                   for c, k in zip(COLORS_D, order)]
        ax.legend(handles=patches, fontsize=8.5, loc='lower right', framealpha=0.9)
        _apply_ax_style(ax)

    plt.tight_layout()
    savefig('expD_plot.png')


# -----------------------------------------------------------------------------
# 13. Experiment E: State coverage growth
# -----------------------------------------------------------------------------
def run_exp_E():
    print('\n' + '='*60)
    print('Experiment E: State Coverage Growth Analysis')
    print('='*60)

    N_MAX         = 10000
    CHECKPOINTS   = [500, 1000, 2000, 4000, 6000, 8000, 10000]
    SEED          = SEEDS[0]

    res = {}
    for pk, sm, label in PROBLEMS:
        print(f'\n  [{pk}]')
        np.random.seed(SEED); random.seed(SEED)
        L      = make_landscape(pk, sm, SEED)
        prior  = make_prior(0.78)
        agents = [DynaQAgent(i, L, SIGMA, dyna_k=DYNA_K,
                              action_prior=prior) for i in range(9)]
        team   = RLTeam(agents, L, trust_level=TRUST)
        rng    = np.random.default_rng(SEED)
        starts = rng.integers(0, L.length, size=N_MAX).tolist()

        log = []
        for idx, s in enumerate(starts):
            team.reset_rl_memory()
            team.tournament(s)
            n = idx + 1
            if n in CHECKPOINTS:
                # FIX 2b: use unique-state helpers - same fix as run_one().
                visited    = count_unique_visited(agents)
                bold_ratio = greedy_ratio(agents).get('Bold', 0.)
                log.append({'n': n, 'visited': visited,
                             'pct': visited/N_STATES, 'bold': bold_ratio})
                print(f'    n={n:6d}  visited={visited:4d}/{N_STATES} '
                      f'({visited/N_STATES:.1%})  bold={bold_ratio:.1%}')
        res[pk] = log

    _plot_E(res)
    return res



# -----------------------------------------------------------------------------
# 14. Entry point
# -----------------------------------------------------------------------------
def main():
    global N_STARTS, OUTPUT_DIR
    # -- Jupyter compatibility ------------------------------------------------
    # When exec()'d inside Jupyter, __name__ is '__main__' so main() is called.
    # Jupyter injects -f kernel-xxx.json into sys.argv, crashing parse_args().
    # Fix: use parse_known_args() to silently ignore unrecognised arguments.
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Dyna-Q Second-Order Agent Experiment Suite')
    parser.add_argument('--exp',      default='ALL',
                        help='which experiment to run (default: ALL)')
    parser.add_argument('--n_starts', type=int, default=N_STARTS,
                        help=f'search starts per seed (default {N_STARTS})')
    parser.add_argument('--outdir',   default=OUTPUT_DIR,
                        help='output directory for figures')
    args, _unknown = parser.parse_known_args()  # _unknown absorbs Jupyter's -f flag

    N_STARTS   = args.n_starts
    OUTPUT_DIR = args.outdir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'\nConfiguration:')
    print(f'  SIGMA={SIGMA}  TRUST={TRUST}  N_STARTS={N_STARTS}  DYNA_K={DYNA_K}')
    print(f'  Output directory: {OUTPUT_DIR}')

    exp = args.exp.upper()
    if exp in ('ALL', 'A'): run_exp_A()
    if exp in ('ALL', 'B'): run_exp_B()
    if exp in ('ALL', 'C'): run_exp_C()
    if exp in ('ALL', 'D'): run_exp_D()
    if exp in ('ALL', 'E'): run_exp_E()

    print('\nAll experiments complete.')

def _plot_E(res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle('Experiment E: State Coverage Growth Analysis\n'
                 f'Single seed ({SEEDS[0]}), {N_STATES} total states',
                 fontsize=13, fontweight='bold', color='#222222')

    # Left: coverage growth
    ax1 = axes[0]
    all_pct = []
    for pk, sm, label in PROBLEMS:
        ns  = [e['n']       for e in res[pk]]
        pct = [e['pct']*100 for e in res[pk]]
        vis = [e['visited'] for e in res[pk]]
        all_pct.extend(pct)
        ax1.plot(ns, pct, color=PK_COLOR[pk], lw=2.5, marker='o',
                 ms=8, label=label)
        for n, p, v in zip(ns, pct, vis):
            ax1.text(n, p + max(all_pct)*0.015, str(v),
                     ha='center', fontsize=8,
                     color=PK_COLOR[pk], fontweight='bold')

    # Fix 1: Use a data-driven y-axis range instead of a fixed upper limit.
    y_max = max(all_pct) * 1.25
    ax1.axhline(y_max * 0.85, color='#AAAAAA', ls='--', lw=1.5, alpha=0.7,
                label=f'80% coverage target\n(actual max={max(all_pct):.1f}%)')
    ax1.set_xlabel('n_starts (cumulative)', fontsize=11)
    ax1.set_ylabel('State Coverage (%)', fontsize=11)
    ax1.set_title('Coverage Growth vs n_starts\n'
                  '(Numbers = absolute visited states)',
                  fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 11000)
    ax1.set_ylim(0, y_max)
    ax1.legend(fontsize=10, framealpha=0.9)
    _apply_ax_style(ax1)

    # Right: Bold proportion stability vs coverage
    ax2 = axes[1]
    all_pct2 = []
    for pk, sm, label in PROBLEMS:
        pct  = [e['pct']*100  for e in res[pk]]
        bold = [e['bold']*100 for e in res[pk]]
        all_pct2.extend(pct)
        ax2.plot(pct, bold, color=PK_COLOR[pk], lw=2.5, marker='o',
                 ms=8, label=label)

    # Fix 2: Keep annotations inside the plot area by choosing the offset direction adaptively.
    x_mid = (min(all_pct2) + max(all_pct2)) / 2
    for pk, sm, label in PROBLEMS:
        pct  = [e['pct']*100  for e in res[pk]]
        bold = [e['bold']*100 for e in res[pk]]
        for i in [-2, -1]:
            offset_x = -1.5 if pct[i] > x_mid else 1.5
            ax2.annotate(f'{bold[i]:.0f}%',
                         xy=(pct[i], bold[i]),
                         xytext=(pct[i] + offset_x, bold[i] + 1.5),
                         fontsize=8.5, color=PK_COLOR[pk], fontweight='bold')

    ax2.set_xlabel('State Coverage (%)', fontsize=11)
    ax2.set_ylabel('Greedy Bold Proportion (%)', fontsize=11)
    ax2.set_title('Bold Proportion Stability\nas Coverage Increases',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9.5, framealpha=0.9)
    _apply_ax_style(ax2)

    plt.tight_layout()
    savefig('expE_plot.png')


if __name__ == '__main__':
    main()
