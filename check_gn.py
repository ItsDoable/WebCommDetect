from networkx.algorithms.community import girvan_newman, modularity
from graph_vis_plotly_cgpt import *
import math
import random
import networkx as nx
import time

def detect_GN_Communities(graph, k = 2):
    G = nx.Graph()
    G.add_edges_from(graph["kanten"])

    print("Davor")
    comp_gen = girvan_newman(G)
    print("Danach")
    best_mod = -1.0
    best_partition = None

    try:
        while True:
            communities = next(comp_gen)
            communities = [list(c) for c in communities]
            mod = modularity(G, communities)
            if mod > best_mod:
                print("Aktuelle mod: ", mod)
                best_mod = mod
                best_partition = communities
            # Uncomment these lines and comment the next break to repeat and increase modularity (takes much time)
            #else:
            #    break  # Modularität sinkt → vorheriges war das Beste"""
            break
    except StopIteration:
        pass

    return best_partition

start_time = time.time()

graph_fn = "graph_zuletzt"

with open(graph_fn + ".json") as f:
    graph = json.load(f)
print("Graph geladen")

# Speichern tatsächliche ccTLD-Communities der Knoten, um sie mit den errechneten zu vergleichen

Communities = {}
Communities_zu_gefunden = {}
for knoten in graph['knoten']:
    ccTLD = knoten.split(".")[-1]
    if ccTLD in Communities:
        Communities[ccTLD].append(knoten)
    else:
        Communities[ccTLD] = [knoten]
        Communities_zu_gefunden[ccTLD] = None

Gefunden_Communities = {}
gn_comm = detect_GN_Communities(graph, 2)
count = 0
for knoten in gn_comm:
    print(knoten)
    Gefunden_Communities[count] = knoten
    count += 1

# Finde zu jeder tatsächlichen Community die beste Entsprechung

Communities_zu_gefunden = finden(Communities, Gefunden_Communities, graph)

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

# Die folgende Farbgebung passiert dreimal: Einmal für die gefundene Communities, dann für die tatsächlichen Communities, und schließlich für die Differenz

# Schritt 3: Rechtecke vorbereiten
squares_gn = []
gn_colors = {"nichts": [255, 255, 255]}
for gn_community in Gefunden_Communities:
    gn_colors[gn_community] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
for (x, y), size, id_ in zip(coords, box_sizes, ids):
    half = size / 2
    square_x = [x - half, x + half, x + half, x - half, x - half]
    square_y = [y - half, y - half, y + half, y + half, y - half]
    gn_comm_title = "nichts"
    for item in graph["gefunden"]:
        if item[0] == id_:
            gn_comm_title = item[1]
    square_color = "rgb(" + ",".join([str(c) for c in gn_colors[gn_comm_title]]) + ")"
    squares_gn.append(go.Scatter(
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

fig = go.Figure(data=squares_gn)
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
    for i in range(len(gn_comm)):
        if node in gn_comm[i]:
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