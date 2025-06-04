# WebCommDetect
Community detection on web domains

DISCLAIMER: Code very messy.

USAGE: Download the f"check_{algorithm}.py" files, and the files "graph_1000.json", "graph_1000_poss.json". Then, run the python files to obtain their partitioning in communities of the graph domain dataset. The check_louvain.py also displays the graph's natural communities, that is, those given by the domain's TLD.
If you want, you can crawl your own domain list and process it to a graph using the "länder.py" and "graph_vis_plotly_cgpt.py" files: run the first one for a while, it will gather a domain list dataset and save it. The second will process it to a graph, and save it to make it usable for the f"check_{algorithm}.py" files.
