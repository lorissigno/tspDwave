################################################################################
# Esempio TSP in QUBO con 3 nodi
################################################################################
from pyqubo import Binary, Constraint, Placeholder
from dimod import ExactSolver
from neal import SimulatedAnnealingSampler
import pandas as pd


# ===================== Parametri problema (N=3) =====================
""" N = 3
W = {
    (0,1): 10, (1,0): 10,
    (0,2): 20, (2,0): 20,
    (1,2): 15, (2,1): 15
} """

N = 10
W = {
    (0,1): 34,
    (0,3): 41,
    (0,7): 45,
    (0,8): 49,
    (0,9): 28,
    (1,0): 34,
    (1,3): 22,
    (1,6): 32,
    (1,9): 17,
    (2,4): 57,
    (2,5): 35,
    (2,6): 61,
    (3,0): 41,
    (3,1): 22,
    (3,6): 21,
    (3,9): 25,
    (4,2): 57,
    (4,5): 18,
    (4,7): 45,
    (4,8): 62,
    (5,2): 35,
    (5,4): 18,
    (5,6): 36,
    (6,1): 32,
    (6,2): 61,
    (6,3): 21,
    (6,5): 36,
    (6,9): 34,
    (7,0): 45,
    (7,4): 45,
    (7,8): 28,
    (8,0): 49,
    (8,4): 62,
    (8,7): 28,
    (9,0): 28,
    (9,1): 17,
    (9,3): 25,
    (9,6): 34,
}


max_w = max(W.values())
A_val = 10.0 * N * max_w + 1.0   # ≈ 6201
B_val = 0.5


# ===================== Costruzione Hamiltoniano =====================
V = range(N)
x = {(v,i): Binary(f"x_{v}_{i}") for v in V for i in range(N)}

nodes_once = sum((1 - sum(x[(v,i)] for i in range(N)))**2 for v in V)
pos_once   = sum((1 - sum(x[(v,i)] for v in V))**2 for i in range(N))

invalid_edge = 0
# Penalità archi mancanti (grafi sparsi): costruisci solo se serve
invalid_terms = []
for u in V:
    for v in V:
        if u != v and (u, v) not in W:
            for i in range(N):
                j = (i + 1) % N
                invalid_terms.append(x[(u, i)] * x[(v, j)])


cost = 0
for (u,v), w in W.items():
    for i in range(N):
        j = (i+1) % N
        cost += w * x[(u,i)] * x[(v,j)]

A = Placeholder("A")
B = Placeholder("B")

ham  = A * Constraint(nodes_once, "nodes_once")
ham += A * Constraint(pos_once,   "pos_once")
if invalid_terms:
    invalid_edge = sum(invalid_terms)
    ham += A * Constraint(invalid_edge, "invalid_edge")
ham += B * cost

ham_internal = ham.compile()
bqm = ham_internal.to_bqm(feed_dict={"A": A_val, "B": B_val})

###############################################################################
# Funzioni utili
###############################################################################
def constraints_ok(s, labels):
    cons = s.constraints()
    def ok(lbl): return (lbl not in cons) or (cons[lbl][0] is True)
    return all(ok(lbl) for lbl in labels)

labels = ["nodes_once","pos_once","invalid_edge"]

def stampa_tabella_dup(decoded_sampleset, tag="ES", N=3):
    data = []
    for s in decoded_sampleset:
        row = {
            "Campionatore": tag,
            "Energia": s.energy,
            "Feasible": constraints_ok(s, labels),
        }
        # ordino le variabili per posizione (colonna j)
        for j in range(N):        # posizione nel tour
            for v in range(N):    # nodo
                key = f"x_{v}_{j}"
                row[key] = s.sample.get(key, 0)
        data.append(row)

    df = pd.DataFrame(data)
    print("\n===============================")
    print(f"   Risultati {tag}")
    print("===============================")
    print(df.head(20).to_string(index=False)) 
    
def stampa_tabella(decoded_sampleset, tag="ES", N=3, unique=True):
    data = []
    seen = set()
    for s in decoded_sampleset:
        # costruiamo la chiave univoca dal sample
        key = tuple(s.sample[k] for j in range(N) for v in range(N) for k in [f"x_{v}_{j}"])
        if unique and key in seen:
            continue
        seen.add(key)

        row = {
            "Campionatore": tag,
            "Energia": s.energy,
            "Feasible": constraints_ok(s, labels),
        }
        for j in range(N):
            for v in range(N):
                key2 = f"x_{v}_{j}"
                row[key2] = s.sample.get(key2, 0)
        data.append(row)

    df = pd.DataFrame(data)
    print("\n===============================")
    print(f"   Risultati {tag}")
    print("===============================")
    print(df.head(20).to_string(index=False)) 
    
def stampa_tabella_compact(decoded_sampleset, tag="ES", N=3, unique=True, max_rows=20):
    from math import inf

    def decode_tour(sample):
        """Ritorna la lista nodi per posizione [v0,v1,...] oppure con '-' dove manca."""
        tour = []
        for j in range(N):
            vj = None
            for v in range(N):
                if int(round(sample.get(f"x_{v}_{j}", 0))) == 1:
                    vj = v
                    break
            tour.append(vj if vj is not None else "-")
        return tour

    def tour_cost(tour, W):
        # valido se nessun '-' e tutti gli archi esistono
        if any(t == "-" for t in tour): 
            return inf
        c = 0
        for i in range(N):
            u = tour[i]
            v = tour[(i+1) % N]
            if (u, v) not in W:
                return inf
            c += W[(u, v)]
        return c

    rows, seen = [], set()
    for s in decoded_sampleset:
        tour = decode_tour(s.sample)
        key = tuple(tour)
        if unique and key in seen:
            continue
        seen.add(key)

        rows.append({
            "Campionatore": tag,
            "Energia": s.energy,
            "Feasible": constraints_ok(s, labels),
            "Tour": tour,
            "CostoW": tour_cost(tour, W)
        })

        if len(rows) >= max_rows:
            break

    df = pd.DataFrame(rows)
    # Ordina prima per Feasible (True sopra), poi per Energia, poi per CostoW
    df = df.sort_values(by=["Feasible", "Energia", "CostoW"], ascending=[False, True, True])

    # stampa
    print("\n===============================")
    print(f"   Risultati {tag}")
    print("===============================")
    # formattazione costo: '∞' per inf
    def fmt(x): return "∞" if x == float("inf") else x
    if not df.empty:
        df["CostoW"] = df["CostoW"].map(fmt)
    print(df.to_string(index=False))




###############################################################################
# Campionamento con ExactSolver
###############################################################################
""" ES = ExactSolver()
sampleset = ES.sample(bqm)
decoded_sampleset = ham_internal.decode_sampleset(sampleset, feed_dict={"A": A_val, "B": B_val})
 """
# print("Sampleset:\n", sampleset)
# print("Lunghezza Sampleset: ", len(sampleset))
# print("\n -- decoded_sampleset[0]:\n", decoded_sampleset[0])

#stampa_tabella(decoded_sampleset, tag="ExactSolver")

###############################################################################
# Campionamento con Simulated Annealing
###############################################################################
SA = SimulatedAnnealingSampler()
#sa_sampleset = SA.sample(bqm, num_reads=50, num_sweeps=500)
sa_sampleset = SA.sample(bqm, num_reads=500, num_sweeps=5000, seed=12345)

decoded_sa = ham_internal.decode_sampleset(sa_sampleset, feed_dict={"A": A_val, "B": B_val})

# print("Sampleset (SA):\n", sa_sampleset)
# print("Lunghezza Sampleset (SA): ", len(sa_sampleset))
# print("\n -- decoded_sa[0]:\n", decoded_sa[0])

stampa_tabella_compact(decoded_sa, tag="SimulatedAnnealing", N=N)
