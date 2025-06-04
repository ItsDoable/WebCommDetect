import community as community_louvain
from graph_vis_plotly_cgpt import *
import math
import random
import networkx as nx
import time

def detect_louvain_communities(graph):
    """
    Führt Louvain-Community-Detection auf einem gerichteten Graphen durch.

    :param graph: dict mit Schlüsseln:
                  - 'nodes': List[str]
                  - 'edges': List[Tuple[str, str]]
    :return: dict {node_id: community_id}
    """
    G = nx.DiGraph()  # gerichteter Graph

    # Knoten und Kanten hinzufügen
    G.add_edges_from(graph['kanten'])

    # Für Louvain brauchen wir einen ungerichteten Graphen (empfohlen)
    G_undirected = G.to_undirected()

    # Community-Erkennung
    partition = community_louvain.best_partition(G_undirected)

    return partition

start_time = time.time()

graph_fn = "graph_1000"

with open(graph_fn + ".json") as f:
    graph = json.load(f)
print("Graph geladen")

# Speichern tatsächliche ccTLD-Communities der Knoten, um sie mit den errechneten zu vergleichen

Communities = {}
Communities_zu_l = {}

for knoten in graph['knoten']:
    ccTLD = knoten.split(".")[-1]
    if ccTLD in Communities:
        Communities[ccTLD].append(knoten)
    else:
        Communities[ccTLD] = [knoten]
        Communities_zu_l[ccTLD] = None

# Finden Louvain-Comminities. Funktion gibt Dict mit {domain: comm_id, ...}, wir machen daraus dict {comm_id: [domains], ...}

Gefunden_Communities = {}
louvain_comm = detect_louvain_communities(graph)
for knoten in louvain_comm:
    print(knoten, louvain_comm[knoten])
    if louvain_comm[knoten] in Gefunden_Communities:
        Gefunden_Communities[louvain_comm[knoten]].append(knoten)
    else:
        Gefunden_Communities[louvain_comm[knoten]] = [knoten]
print(Gefunden_Communities)

# Finde zu jeder tatsächlichen Community die beste Entsprechung

Communities_zu_l = finden(Communities, Gefunden_Communities, graph)

#poss = fruchterman_reingold_graph(graph)
with open(graph_fn + "_poss.json", "rb") as f:
    poss = json.load(f)

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
for Louvain_Community in Gefunden_Communities:
    louvain_colors[Louvain_Community] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
for (x, y), size, id_ in zip(coords, box_sizes, ids):
    half = size / 2
    square_x = [x - half, x + half, x + half, x - half, x - half]
    square_y = [y - half, y - half, y + half, y + half, y - half]
    louvain_comm_title = "nichts"
    for item in graph["gefunden"]:
        if item[0] == id_:
            louvain_comm_title = item[1]
    square_color = "rgb(" + ",".join([str(c) for c in louvain_colors[louvain_comm_title]]) + ")"
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
    title="Louvain Communities",
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

squares = []
colors = {}
ids = list(poss.keys())
for id in ids:
    ccTLD = id.split(".")[-1]
    if ccTLD not in colors:
        colors[ccTLD] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
for (x, y), size, id_ in zip(coords, box_sizes, ids):
    half = size / 2
    square_x = [x - half, x + half, x + half, x - half, x - half]
    square_y = [y - half, y - half, y + half, y + half, y - half]
    ccTLD = id_.split(".")[-1]
    square_color = "rgb(" + ",".join([str(c) for c in colors[ccTLD]]) + ")"
    squares.append(go.Scatter(
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

fig = go.Figure(data=squares)
fig.update_layout(
    title="ccTLD-Communities",
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
    if Communities_zu_l[ccTLD] is not None:
        if id_ in Communities_zu_l[ccTLD]:
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
print("Mit Louvain gefunden: ", menge_gefunden / len(coords))

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
comm_for_nmi_found = list(louvain_comm.values())
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