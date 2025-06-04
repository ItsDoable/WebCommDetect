import json

import plotly.graph_objects as go
import numpy as np

def fruchterman_reingold_graph(graph, dim=2, iterations=50, area=10000.0, gravity=1.0, cooling=0.95):
    nodes = list(graph["knoten"])
    node_index = {node: idx for idx, node in enumerate(nodes)}
    n_nodes = len(nodes)
    pos = np.random.rand(n_nodes, dim) * np.sqrt(area)
    disp = np.zeros((n_nodes, dim))
    k = np.sqrt(area / n_nodes)
    t = np.sqrt(area) / 10

    # Adjazenzliste
    neighbors = [[] for _ in range(n_nodes)]
    for u, v in graph["kanten"]:
        ui, vi = node_index[u], node_index[v]
        neighbors[ui].append(vi)
        neighbors[vi].append(ui)

    for _ in range(iterations):
        old_pos = pos.copy()

        print(_ / iterations)

        disp[:] = 0

        # Repulsive forces
        for i in range(n_nodes):
            delta = pos[i] - pos
            distance = np.linalg.norm(delta, axis=1) + 1e-9
            repulsive_force = (k * k / distance**2)[:, np.newaxis]
            disp[i] += np.sum(delta * repulsive_force, axis=0)

        # Attractive forces
        for i in range(n_nodes):
            for j in neighbors[i]:
                delta = pos[i] - pos[j]
                dist = np.linalg.norm(delta) + 1e-9
                attractive_force = (dist * dist / k)
                disp[i] -= delta * (attractive_force / dist)

        # Gravity
        pos -= pos.mean(axis=0)
        disp -= pos * gravity

        # Update positions
        length = np.linalg.norm(disp, axis=1) + 1e-9
        pos += (disp.T * np.minimum(length, t) / length).T

        # Cooling
        t *= cooling

        print("Änderungen:")
        durchschnittsänderung = 0
        for i in range(len(pos)):
            durchschnittsänderung += np.linalg.norm(old_pos[i] - pos[i])
        durchschnittsänderung /= len(pos)
        print(durchschnittsänderung)

    # Rückgabe: dict {node_id: [x, y]}
    return {node: pos[idx].tolist() for node, idx in node_index.items()}

def finden(Communities, Gefunden_Communities, graph):
    graph["gefunden"] = []
    Communities_zu_gefunden = {}
    schon_vergebene = []
    Communities_sorted = sorted([Communities[comm] for comm in Communities], key=len, reverse=True)
    for Community in Communities_sorted:
        ccTLD = Community[0].split(".")[-1]  # Finden nach Sortierung ccTLD dieser Community
        print("Community", Community)
        beste_gefunden = None
        beste_anzahl = 0
        for gefunden_community_index in Gefunden_Communities:
            gefunden_community = Gefunden_Communities[gefunden_community_index]
            if gefunden_community in schon_vergebene:
                continue
            gedeckte_anzahl = 0
            for domain in Community:
                if domain in gefunden_community:
                    print(
                        f"Domain {domain} aus Community {Community} in gefundener Community {gefunden_community} gefunden.")
                    graph["gefunden"].append([domain, gefunden_community_index])
                    gedeckte_anzahl += 1
            if gedeckte_anzahl > beste_anzahl:
                beste_gefunden = gefunden_community
                beste_anzahl = gedeckte_anzahl
        schon_vergebene.append(beste_gefunden)
        Communities_zu_gefunden[ccTLD] = beste_gefunden
        print(f"Setzen Commuinities_zu_l[{ccTLD}] auf {beste_gefunden}")
        print("Hat Entsprechung", beste_gefunden)
    return Communities_zu_gefunden

if __name__ == '__main__':
    import plotly.graph_objects as go
    import json
    import random
    import math

    # Beispiel-Datenstruktur
    with open("zustand.json", "r", encoding="utf-8") as f:
        objekt = json.load(f)
        print(objekt)
    Arr = objekt[2]
    print(Arr)
    print(len(Arr))
    #Arr = dict(list(Arr.items())[0:1000]) # Um einen kleineren Graphen zu bekommen
    #print(len(Arr))

    graph = {"knoten": [], "kanten": [], "kantenmengen": [], "ccTLDs": {}} # Kanten ist als Array von Tupeln aussagelos über die Kantenmengen einzelner Domains. Diese werden separat gespeichert

    def add_node(node):
        if not node in graph["knoten"]:
            graph["knoten"].append(node)

    def add_edge(edge):
        if not edge in graph["kanten"]:
            graph["kanten"].append(edge)

    def count_edges(node):
        count = 0
        for edge in graph["kanten"]:
            if node == edge[0] or node == edge[1]:
                count += 1
                if node == edge[0] and node == edge[1]: # Schleife, wird doppelt gezählt
                    count += 1
        return count

    def save(objekt, filename):
        with open(filename, "w") as f:
            json.dump(objekt, f)

    count = 0
    for source, targets in Arr.items():
        if count % 100 == 0:
            print(count / len(Arr))
        add_node(source)
        for target in targets:
            add_node(target)
            add_edge((source, target))
        count += 1

    colors = {}
    for domain in graph["knoten"]:
        graph["kantenmengen"].append(count_edges(domain))
        ccTLD = domain.split(".")[-1]
        if ccTLD not in colors:
            graph["ccTLDs"][ccTLD] = [domain]
            colors[ccTLD] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        else:
            graph["ccTLDs"][ccTLD].append(domain)

    #save(graph, "graph_1000.json")
    #poss = fruchterman_reingold_graph(graph)
    #save(poss, "graph_1000_poss.json")

    def kanten_zu_größe(n): # sigmoid / 2 als Vierecksgröße abh. von Kantenmenge
        return (0.9 / (1 + math.exp(-n))) - 0.4

    def display():
        coords = np.array(list(poss.values()))
        # Schritt 1: Abstandsmatrix und minimale Nachbardistanz berechnen
        from scipy.spatial import distance_matrix
        dists = distance_matrix(coords, coords)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists, axis=1) # Array, wo zu jedem Knoten die kleinste Distanz zu einem anderen steht
        koeffs = np.array([kanten_zu_größe(graph["kantenmengen"][i]) for i in range(len(graph["knoten"]))])

        # Schritt 2: Vierecksgröße basierend auf Kantenmenge
        box_sizes = min_dist * koeffs

        # Schritt 3: Rechtecke vorbereiten
        squares = []
        for (x, y), size, id_ in zip(coords, box_sizes, graph["knoten"]):
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

    for i in range(50):
        poss = fruchterman_reingold_graph(graph, iterations=1)
        display()