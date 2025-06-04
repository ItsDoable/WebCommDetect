from graph_vis_plotly_cgpt import *
import math
from sklearn.cluster import HDBSCAN
import random
import time

def hdbscan(data): # data 2D-Array
    hdb = HDBSCAN(min_cluster_size=2)
    hdb.fit(data)
    return hdb.labels_

start_time = time.time()

graph_fn = "graph_1000"

with open(graph_fn + ".json") as f:
    graph = json.load(f)
print("Graph geladen")

# Speichern tatsächliche ccTLD-Communities der Knoten, um sie mit den errechneten zu vergleichen.
# Communities_zu_hdbscan ist Entsprechungsliste

Communities = {}
Communities_zu_hdbscan = {}

for knoten in graph["knoten"]:
    ccTLD = knoten.split(".")[-1]
    if ccTLD in Communities:
        Communities[ccTLD].append(knoten)
    else:
        Communities[ccTLD] = [knoten]
        Communities_zu_hdbscan[ccTLD] = None

#poss = fruchterman_reingold_graph(graph)
with open(graph_fn + "_poss.json", "rb") as f:
    poss = json.load(f)

# Übersetzen poss von dict zu list
poss_arr = []
for node in graph["knoten"]:
    node_pos = poss[node]
    poss_arr.append(node_pos)

HDBSCAN_Communities = {}
hdbscan_labels = hdbscan(poss_arr)
# Müssen Labelliste in Comm-Liste überführen
anzahl_klassen = len(set(hdbscan_labels)) + 1
hdbscan_comm = [[] for i in range(anzahl_klassen)]
for i in range(len(hdbscan_labels)):
    label = hdbscan_labels[i] + 1 # Versetzen alles um 1, sodaß die Klasse -1 zu 0 wird und ein gültiger Index ist
    hdbscan_comm[label].append(graph["knoten"][i])

count = 0
for comm in hdbscan_comm:
    print(comm)
    HDBSCAN_Communities[count] = comm
    count += 1

# Finde zu jeder tatsächlichen Community die beste Entsprechung

Communities_zu_hdbscan = finden(Communities, HDBSCAN_Communities, graph)

"""schon_vergebene = []
for Community in Communities:
    print("Community", Community)
    beste_hdbscan = None
    beste_anzahl = 0
    for HDBSCAN_Community_index in HDBSCAN_Communities:
        HDBSCAN_Community = HDBSCAN_Communities[HDBSCAN_Community_index]
        if HDBSCAN_Community in schon_vergebene:
            continue
        gedeckte_anzahl = 0
        for domain in Communities[Community]:
            if domain in HDBSCAN_Community:
                print(f"Domain {domain} aus Community {Community} in Louvain-Community {HDBSCAN_Community} gefunden.")
                graph["gefunden_hdbscan"].append([domain, HDBSCAN_Community_index])
                gedeckte_anzahl += 1
        if gedeckte_anzahl > beste_anzahl:
            beste_hdbscan = HDBSCAN_Community
            beste_anzahl = gedeckte_anzahl
    schon_vergebene.append(beste_hdbscan)
    Communities_zu_hdbscan[Community] = beste_hdbscan
    print(f"Setzen Commuinities_zu_l[{Community}] auf {beste_hdbscan}")
    print("Hat Entsprechung", beste_hdbscan)
print(graph["gefunden_hdbscan"])"""

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
for Louvain_Community in HDBSCAN_Communities:
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
    if Communities_zu_hdbscan[ccTLD] is not None:
        if id_ in Communities_zu_hdbscan[ccTLD]:
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
print("Mit Übereinstimmung gefunden: ", menge_gefunden / len(coords))

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
    for i in range(len(hdbscan_comm)):
        if node in hdbscan_comm[i]:
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