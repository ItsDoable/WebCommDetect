from graph_vis_plotly_cgpt import *
import math
import random
import networkx as nx
import time

def detect_lpa_communities_neu(graph):
    G = nx.Graph()
    G.add_edges_from(graph["kanten"])

    communities = nx.algorithms.community.label_propagation_communities(G)

    # Ergebnis ist ein Generator → in Liste umwandeln
    communities = list(communities)

    # Optional: Als Listen ausgeben
    communities = [list(c) for c in communities]

    return communities

start_time = time.time()

graph_fn = "graph_1000"

with open(graph_fn + ".json") as f:
    graph = json.load(f)
print("Graph geladen")

# Speichern tatsächliche ccTLD-Communities der Knoten, um sie mit den errechneten zu vergleichen.
# Communities_zu_lpa ist Entsprechungsliste

Communities = {}
Communities_zu_gefunden = {}

for knoten in graph['knoten']:
    ccTLD = knoten.split(".")[-1]
    if ccTLD in Communities:
        Communities[ccTLD].append(knoten)
    else:
        Communities[ccTLD] = [knoten]
        Communities_zu_gefunden[ccTLD] = None

#poss = fruchterman_reingold_graph(graph)
with open(graph_fn + "_poss.json", "rb") as f:
    poss = json.load(f)

LPA_Communities = {}
lpa_comm = detect_lpa_communities_neu(graph)

count = 0
for comm in lpa_comm:
    print(comm)
    LPA_Communities[count] = comm
    count += 1


# Finde zu jeder tatsächlichen Community die beste Entsprechung

Communities_zu_gefunden = finden(Communities, LPA_Communities, graph)

"""schon_vergebene = []
for Community in Communities:
    print("Community", Community)
    beste_lpa = None
    beste_anzahl = 0
    for LPA_Community_index in LPA_Communities:
        LPA_Community = LPA_Communities[LPA_Community_index]
        if LPA_Community in schon_vergebene:
            continue
        gedeckte_anzahl = 0
        for domain in Communities[Community]:
            if domain in LPA_Community:
                print(f"Domain {domain} aus Community {Community} in Louvain-Community {LPA_Community} gefunden.")
                graph["gefunden_lpa"].append([domain, LPA_Community_index])
                gedeckte_anzahl += 1
        if gedeckte_anzahl > beste_anzahl:
            beste_lpa = LPA_Community
            beste_anzahl = gedeckte_anzahl
    schon_vergebene.append(beste_lpa)
    Communities_zu_lpa[Community] = beste_lpa
    print(f"Setzen Commuinities_zu_l[{Community}] auf {beste_lpa}")
    print("Hat Entsprechung", beste_lpa)
print(graph["gefunden_lpa"])"""

# Nutzen hier Knotenpositionen für Graphik

# IDs und Koordinaten extrahieren

coords = np.array(list(poss.values()))

def kanten_zu_größe(n): # sigmoid-ähnliche Fkt als Vierecksgröße abh. von Kantenmenge
    return (0.9 / (1 + math.exp(-n))) - 0.4

# Schritt 1: Abstandsmatrix und minimale Nachbardistanz berechnen
from scipy.spatial import distance_matrix
dists = distance_matrix(coords, coords)
np.fill_diagonal(dists, np.inf)
min_dist = np.min(dists, axis=1) # Array, wo zu jedem Knoten die kleinste Distanz zu einem anderen steht
koeffs = np.array([kanten_zu_größe(graph["kantenmengen"][i]) for i in range(len(graph["knoten"]))])

# Schritt 2: Vierecksgröße basierend auf Kantenmenge
box_sizes = min_dist * koeffs

ids = list(poss.keys())

# Die folgende Farbgebung passiert dreimal: Einmal für die Louvain-Communities, dann für die tatsächlichen Communities, und schließlich für die Differenz

# Schritt 3: Rechtecke vorbereiten
squares_louvain = []
louvain_colors = {"nichts": [255, 255, 255]}
for Louvain_Community in LPA_Communities:
    louvain_colors[Louvain_Community] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
for (x, y), size, id_ in zip(coords, box_sizes, ids):
    half = size / 2
    square_x = [x - half, x + half, x + half, x - half, x - half]
    square_y = [y - half, y - half, y + half, y + half, y - half]
    louvain_comm = "nichts"
    for item in graph["gefunden"]:
        if item[0] == id_:
            louvain_comm = item[1]
    square_color = "rgb(" + ",".join([str(c) for c in louvain_colors[louvain_comm]]) + ")"
    squares_louvain.append(go.Scatter(
        x=square_x,
        y=square_y,
        mode="lines",
        line=dict(color=square_color),
        fillcolor=square_color,
        fill="toself",
        text=f"ID: {id_}",
        hoverinfo="text",
        showlegend=False
    ))

# Schritt 4: Layout anpassen
x_margin = max(box_sizes)
y_margin = max(box_sizes)

x_min = np.min(coords[:, 0] - box_sizes/2) - x_margin
x_max = np.max(coords[:, 0] + box_sizes/2) + x_margin
y_min = np.min(coords[:, 1] - box_sizes/2) - y_margin
y_max = np.max(coords[:, 1] + box_sizes/2) + y_margin

fig = go.Figure(data=squares_louvain)
fig.update_layout(
    plot_bgcolor="black",
    paper_bgcolor="black",
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    xaxis_range=[x_min, x_max],
    yaxis_range=[y_min, y_max],
    margin=dict(l=0, r=0, t=0, b=0)
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.show()

squares_diff = []
colors_diff = ["rgb(0, 0, 255)", "rgb(255, 0, 0)"]
menge_gefunden = 0
for (x, y), size, id_ in zip(coords, box_sizes, ids):
    half = size / 2
    square_x = [x - half, x + half, x + half, x - half, x - half]
    square_y = [y - half, y - half, y + half, y + half, y - half]
    group = 1 # Nicht gefunden
    ccTLD = id_.split(".")[-1]
    if Communities_zu_gefunden[ccTLD] is not None:
        if id_ in Communities_zu_gefunden[ccTLD]:
            print(id_)
            group = 0 # Gefunden
            menge_gefunden += 1
    square_color = colors_diff[group]
    squares_diff.append(go.Scatter(
        x=square_x,
        y=square_y,
        mode="lines",
        line=dict(color=square_color),
        fillcolor=square_color,
        fill="toself",
        text=f"ID: {id_}",
        hoverinfo="text",
        showlegend=False
    ))
print("Mit Girvan-Newman gefunden: ", menge_gefunden / len(coords))

# Schritt 4: Layout anpassen
x_margin = max(box_sizes)
y_margin = max(box_sizes)

x_min = np.min(coords[:, 0] - box_sizes/2) - x_margin
x_max = np.max(coords[:, 0] + box_sizes/2) + x_margin
y_min = np.min(coords[:, 1] - box_sizes/2) - y_margin
y_max = np.max(coords[:, 1] + box_sizes/2) + y_margin

fig = go.Figure(data=squares_diff)
fig.update_layout(
    plot_bgcolor="black",
    paper_bgcolor="black",
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    xaxis_range=[x_min, x_max],
    yaxis_range=[y_min, y_max],
    margin=dict(l=0, r=0, t=0, b=0)
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.show()
# NMI berechnen und ausgeben

ccTLDs = []
for node in graph["knoten"]:
    ccTLD = node.split(".")[-1]
    if ccTLD not in ccTLDs:
        ccTLDs.append(ccTLD)
comm_for_nmi_orig = [ccTLDs.index(node.split(".")[-1]) for node in graph["knoten"]]

comm_for_nmi_found = []
for node in graph["knoten"]:
    for i in range(len(lpa_comm)):
        if node in lpa_comm[i]:
            comm_for_nmi_found.append(i)

len_a = len(comm_for_nmi_orig)
len_b = len(comm_for_nmi_found)
if len_a > len_b:
    comm_for_nmi_orig = comm_for_nmi_orig[0: len_b]
elif len_b > len_a:
    comm_for_nmi_found = comm_for_nmi_found[0: len_a]

from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

nmi = normalized_mutual_info_score(comm_for_nmi_orig, comm_for_nmi_found)
print("NMI: ", nmi)
ami = adjusted_mutual_info_score(comm_for_nmi_orig, comm_for_nmi_found)
print("AMI: ", ami)

print("--- %s seconds ---" % (time.time() - start_time))