import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import requests
import matplotlib.colors as mcolors
from mplsoccer import Pitch

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events"

def remote_events(match_id: int) -> pd.DataFrame:
    url = f"{BASE}/{match_id}.json"
    return pd.DataFrame(requests.get(url).json())

def extract_name(obj):
    if isinstance(obj, dict):
        return obj.get("name", "Unknown")
    return str(obj) if obj is not None else "Unknown"

def build_pass_network_with_positions(events_df, team_name, min_touches=10):
    events_df = events_df.copy()
    events_df["team_name"]   = events_df["team"].apply(extract_name)
    events_df["player_name"] = events_df["player"].apply(lambda x: extract_name(x) if x is not None else None)
    events_df["event_type"]  = events_df["type"].apply(extract_name)

    team_events = events_df[events_df["team_name"] == team_name]
    passes_df   = team_events[team_events["event_type"] == "Pass"]

    # average positions
    player_locs = {}
    for player, group in team_events.groupby("player_name"):
        locs = group["location"].dropna().tolist()
        locs = [l for l in locs if isinstance(l, list) and len(l) == 2]
        if len(locs) >= min_touches:
            arr = np.array(locs)
            player_locs[player] = (arr[:, 0].mean(), arr[:, 1].mean())

    # build graph
    G = nx.DiGraph()
    for _, row in passes_df.iterrows():
        passer = row["player_name"]
        pass_info = row.get("pass", {})
        if not isinstance(pass_info, dict):
            continue

        rec = pass_info.get("recipient")
        receiver = extract_name(rec) if rec else None
        outcome = pass_info.get("outcome")
        is_successful = not (isinstance(outcome, dict) and outcome.get("name") == "Incomplete")

        if (
            passer and receiver and is_successful
            and passer in player_locs and receiver in player_locs
            and passer != receiver
        ):
            if G.has_edge(passer, receiver):
                G[passer][receiver]["weight"] += 1
            else:
                G.add_edge(passer, receiver, weight=1)

    # ensure isolated nodes appear
    for p in player_locs.keys():
        if p not in G:
            G.add_node(p)

    return G, player_locs

def plot_team_networks_pitch(G_left, pos_left, team_left,
                             G_right, pos_right, team_right):

    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    def draw_one(G, pos, team, ax, node_color):
        density = nx.density(G)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=node_color,
                               node_size=500,
                               edgecolors="black",
                               linewidths=1.0)

        weights = [G[u][v]["weight"] for u, v in G.edges()]
        if weights:
            max_w = max(weights)
            widths = [1 + 4 * (w / max_w) for w in weights]
        else:
            widths = 1

        nx.draw_networkx_edges(G, pos, ax=ax,
                               edge_color="gray",
                               alpha=0.6,
                               width=widths,
                               arrows=False)

        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

        ax.set_title(
            f"{team}\nDensity:{density:.3f} | Nodes:{n_nodes} | Edges:{n_edges}",
            fontsize=10, fontweight="bold"
        )
        ax.set_xlim(0, 120)
        ax.set_ylim(80, 0)
        ax.axis("off")

    draw_one(G_left,  pos_left,  team_left,  axes[0], "#a6cee3")
    draw_one(G_right, pos_right, team_right, axes[1], "#b2df8a")

    fig.tight_layout()
    
    # --- CHANGED: Saved as PNG instead of plt.show() ---
    filename = "passing_networks.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Saved passing networks visualization to {filename}")

def calculate_centralities(pass_network):
    centralities = {}

    centralities['degree'] = nx.degree_centrality(pass_network)
    centralities['pagerank'] = nx.pagerank(pass_network, weight='weight')
    centralities['betweenness'] = nx.betweenness_centrality(pass_network, weight='weight')

    closeness = {}
    for comp in nx.weakly_connected_components(pass_network):
        sub = pass_network.subgraph(comp)
        closeness.update(nx.closeness_centrality(sub))

    centralities['closeness'] = closeness
    return centralities

def extract_shot_characteristics(events_df):
    if "event_type" not in events_df.columns:
        events_df["event_type"] = events_df["type"].apply(lambda x: x.get("name") if isinstance(x, dict) else str(x))

    shots = events_df[events_df["event_type"] == "Shot"].copy()
    shot_analysis = []

    for idx, shot in shots.iterrows():
        shot_info = shot.get("shot", {})
        location = shot.get("location")

        if isinstance(shot_info, dict) and isinstance(location, list) and len(location) == 2:
            xg_value = shot_info.get("statsbomb_xg", 0.0)
            outcome = shot_info.get("outcome", {}).get("name", "Unknown")

            goal_x, goal_y = 120, 40
            distance = np.sqrt((location[0] - goal_x)**2 + (location[1] - goal_y)**2)

            goal_width = 8
            y_diff = abs(location[1] - goal_y)
            if location[0] < goal_x:
                angle_rad = np.arctan(goal_width / (2 * np.sqrt((goal_x - location[0])**2 + y_diff**2)))
                angle_deg = np.degrees(angle_rad)
            else:
                angle_deg = 0.0

            shot_analysis.append({
                "shot_id": idx,
                "x": location[0],
                "y": location[1],
                "xg": xg_value,
                "outcome": outcome,
                "distance_to_goal": distance,
                "shot_angle": angle_deg,
                "player": extract_name(shot.get("player")),
                "team": extract_name(shot.get("team")),
                "minute": shot.get("minute", 0)
            })

    return pd.DataFrame(shot_analysis)

# --- Execution Flow ---
events_df = remote_events(15946)

G_barca, pos_barca = build_pass_network_with_positions(events_df, "Barcelona")
G_alaves, pos_alaves = build_pass_network_with_positions(events_df, "Deportivo Alavés")

# Generates and saves passing_networks.png
plot_team_networks_pitch(G_barca, pos_barca, "Barcelona",
                         G_alaves, pos_alaves, "Deportivo Alavés")

# --- Extract shot data ---
shots_df = extract_shot_characteristics(events_df)

# --- Create StatsBomb pitch ---
pitch = Pitch(pitch_type='statsbomb', pitch_color='white',
              line_color='black', linewidth=2)
fig, ax = pitch.draw(figsize=(10, 7))

# --- Normalized colormap (blue→red) ---
norm = mcolors.Normalize(vmin=0, vmax=shots_df['xg'].max())
cmap = plt.get_cmap('coolwarm')

# --- Plot non‑goals ---
non_goals = shots_df[shots_df['outcome'] != 'Goal']
sc = ax.scatter(
    non_goals['x'], non_goals['y'],
    c=non_goals['xg'], cmap=cmap, norm=norm,
    s=non_goals['xg'] * 800,
    edgecolors='black', linewidths=0.5, alpha=0.9
)

# --- Plot goals as stars ---
goals = shots_df[shots_df['outcome'] == 'Goal']
ax.scatter(
    goals['x'], goals['y'],
    c=goals['xg'], cmap=cmap, norm=norm,
    s=goals['xg'] * 1000,
    marker='*', edgecolors='black', linewidths=0.8, alpha=1.0
)

# --- Add colorbar with consistent scale ---
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Expected Goals (xG)', fontsize=11)
cbar.ax.tick_params(labelsize=9)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1.2)

# --- Title ---
total_xg = shots_df['xg'].sum()
ax.set_title(f"Shot Map with xG Values\n{len(shots_df)} shots, {total_xg:.2f} total xG",
             fontsize=13, fontweight='bold')

# --- CHANGED: Saved as PNG instead of plt.show() ---
shotmap_filename = "shot_map.png"
plt.savefig(shotmap_filename, bbox_inches='tight', dpi=300)
print(f"Saved shot map visualization to {shotmap_filename}")
