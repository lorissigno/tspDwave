################################################################################
# Esempio TSP in QUBO con 3 nodi
################################################################################
from pyqubo import Binary, Constraint, Placeholder
from dimod import ExactSolver
from neal import SimulatedAnnealingSampler
import pandas as pd


# ===================== Parametri problema (N=3) =====================
N = 3
W = {
    (0,1): 10, (1,0): 10,
    (0,2): 20, (2,0): 20,
    (1,2): 15, (2,1): 15
}

A_val = 50.0
B_val = 1.0

# ===================== Costruzione Hamiltoniano =====================
V = range(N)
x = {(v,i): Binary(f"x_{v}_{i}") for v in V for i in range(N)}

nodes_once = sum((1 - sum(x[(v,i)] for i in range(N)))**2 for v in V)
pos_once   = sum((1 - sum(x[(v,i)] for v in V))**2 for i in range(N))

invalid_edge = 0
for u in V:
    for v in V:
        if u != v and (u,v) not in W:
            for i in range(N):
                j = (i+1) % N
                invalid_edge += x[(u,i)] * x[(v,j)]

cost = 0
for (u,v), w in W.items():
    for i in range(N):
        j = (i+1) % N
        cost += w * x[(u,i)] * x[(v,j)]

A = Placeholder("A")
B = Placeholder("B")

ham  = A * Constraint(nodes_once, "nodes_once")
ham += A * Constraint(pos_once,   "pos_once")
if invalid_edge != 0:
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
    print(df.to_string(index=False))



###############################################################################
# Campionamento con ExactSolver
###############################################################################
ES = ExactSolver()
sampleset = ES.sample(bqm)
decoded_sampleset = ham_internal.decode_sampleset(sampleset, feed_dict={"A": A_val, "B": B_val})

# print("Sampleset:\n", sampleset)
# print("Lunghezza Sampleset: ", len(sampleset))
# print("\n -- decoded_sampleset[0]:\n", decoded_sampleset[0])

stampa_tabella(decoded_sampleset, tag="ExactSolver")

###############################################################################
# Campionamento con Simulated Annealing
###############################################################################
SA = SimulatedAnnealingSampler()
sa_sampleset = SA.sample(bqm, num_reads=50, num_sweeps=500)
decoded_sa = ham_internal.decode_sampleset(sa_sampleset, feed_dict={"A": A_val, "B": B_val})

# print("Sampleset (SA):\n", sa_sampleset)
# print("Lunghezza Sampleset (SA): ", len(sa_sampleset))
# print("\n -- decoded_sa[0]:\n", decoded_sa[0])

stampa_tabella(decoded_sa, tag="SimulatedAnnealing")


# === D-Wave LeapHybridSampler (ibrido) ===
#   print("\n=== D-Wave LeapHybridSampler ===")
#   try:
#       hybrid = LeapHybridSampler()
#       # time_limit in secondi (rispettare i min/max del solver)
#       hyb_samples = hybrid.sample(bqm, time_limit=5)
#       best_hyb, tour_hyb, energy_hyb = decode_best_feasible(model, hyb_samples, feed, N)
#       if tour_hyb:
#           print("Tour:", tour_hyb, "Energy:", energy_hyb)
#       else:
#           print("Nessuna soluzione valida")
#   except Exception as e:
#       print("LeapHybridSampler non disponibile:", e)
#
#   # === D-WaveSampler (QPU reale) ===
#   print("\n=== D-WaveSampler (QPU) ===")
#   try:
#       qpu = EmbeddingComposite(DWaveSampler())
#       qpu_samples = qpu.sample(bqm, num_reads=1000)
#       best_qpu, tour_qpu, energy_qpu = decode_best_feasible(model, qpu_samples, feed, N)
#       if tour_qpu:
#           print("Tour:", tour_qpu, "Energy:", energy_qpu)
#       else:
#           print("Nessuna soluzione valida")
#   except Exception as e:
#       print("DWaveSampler non disponibile o embedding fallito:", e)