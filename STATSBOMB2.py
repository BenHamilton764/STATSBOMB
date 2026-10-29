import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import requests
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
events_df = remote_events(15946)

G_barca, pos_barca = build_pass_network_with_positions(events_df, "Barcelona")
G_alaves, pos_alaves = build_pass_network_with_positions(events_df, "Deportivo Alavés")

plot_team_networks_pitch(G_barca, pos_barca, "Barcelona",
                         G_alaves, pos_alaves, "Deportivo Alavés")


from mplsoccer import Pitch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

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


def analyze_possession_sequences(events_df):
    """
    Analyze possession chain characteristics and build-up patterns.
    Compatible with StatsBomb open-data JSON structure.
    """

    df = events_df.copy()

    # Standardise fields (same as rest of your code)
    df["team_name"]   = df["team"].apply(extract_name)
    df["event_type"]  = df["type"].apply(extract_name)

    possession_chains = []

    # Group by possession ID
    for poss_id, group in df.groupby("possession"):
        group = group.sort_values(["minute", "second"])

        if len(group) < 2:
            continue

        team = group["team_name"].iloc[0]
        chain_length = len(group)

        # --- Duration ---
        start_minute = group["minute"].iloc[0]
        end_minute   = group["minute"].iloc[-1]
        start_second = group["second"].iloc[0]
        end_second   = group["second"].iloc[-1]

        duration = (end_minute - start_minute) * 60 + (end_second - start_second)
        duration = max(duration, 0)

        # --- Event type counts ---
        pass_count  = (group["event_type"] == "Pass").sum()
        carry_count = (group["event_type"] == "Carry").sum()
        shot_count  = (group["event_type"] == "Shot").sum()

        # --- Spatial progression ---
        locs = group["location"].dropna().tolist()
        locs = [l for l in locs if isinstance(l, list) and len(l) == 2]

        if len(locs) >= 2:
            field_progression = locs[-1][0] - locs[0][0]
        else:
            field_progression = 0

        possession_chains.append({
            "possession_id": poss_id,
            "team": team,
            "length": chain_length,
            "duration_seconds": duration,
            "passes": pass_count,
            "carries": carry_count,
            "shots": shot_count,
            "field_progression": field_progression,
            "end_event": group["event_type"].iloc[-1],
            "start_minute": start_minute
        })

    return pd.DataFrame(possession_chains)
# Load StatsBomb event data
events_df = remote_events(15946)

def extract_name(obj):
    """Extract names from StatsBomb nested dictionary structure."""
    if isinstance(obj, dict):
        name = obj.get("name", "Unknown")
        try:
            return name.encode("ascii", "ignore").decode("ascii")
        except:
            return "Unknown"
    return str(obj) if obj is not None else "Unknown"

# Standardise fields
events_df["team_name"]   = events_df["team"].apply(extract_name)
events_df["player_name"] = events_df["player"].apply(lambda x: extract_name(x) if x else None)
events_df["event_type"]  = events_df["type"].apply(extract_name)

print("Creating possession chain visualization...")

# ---------------------------------------------------------
# FIXED POSSESSION FINDER
# ---------------------------------------------------------
def find_interesting_possession(events_df):
    """Find a long, interesting possession chain with multiple events."""

    possession_analysis = []

    for poss_id, poss_events in events_df.groupby("possession"):
        poss_events = poss_events.sort_values(["minute", "second"])

        if len(poss_events) < 8:
            continue

        passes  = (poss_events["event_type"] == "Pass").sum()
        carries = (poss_events["event_type"] == "Carry").sum()
        shots   = (poss_events["event_type"] == "Shot").sum()

        # Extract valid locations
        locs = [
            loc for loc in poss_events["location"].tolist()
            if isinstance(loc, list) and len(loc) == 2
        ]

        if len(locs) < 5:
            continue

        progression = locs[-1][0] - locs[0][0]

        possession_analysis.append({
            "possession_id": poss_id,
            "length": len(poss_events),
            "passes": passes,
            "carries": carries,
            "shots": shots,
            "progression": progression,
            "team": poss_events["team_name"].iloc[0],
            "events": poss_events
        })

    if possession_analysis:
        possession_analysis.sort(
            key=lambda x: x["length"] + x["progression"]/10 + x["shots"]*5,
            reverse=True
        )
        return possession_analysis[0]

    return None

# ---------------------------------------------------------
# SELECT POSSESSION
# ---------------------------------------------------------
interesting_possession = find_interesting_possession(events_df)

if interesting_possession is None:
    print("No suitable possession chains found. Using fallback.")
    for poss_id, poss_events in events_df.groupby("possession"):
        if len(poss_events) >= 5:
            interesting_possession = {
                "possession_id": poss_id,
                "events": poss_events.sort_values(["minute", "second"]),
                "team": poss_events["team_name"].iloc[0],
                "length": len(poss_events)
            }
            break

if not interesting_possession:
    print("No possession chains found.")
    raise SystemExit()

# ---------------------------------------------------------
# BUILD SEQUENCE
# ---------------------------------------------------------
possession_events = interesting_possession["events"]
team_name = interesting_possession["team"]

print(f"Visualizing possession chain: {interesting_possession['possession_id']}")
print(f"Team: {team_name}")
print(f"Length: {interesting_possession['length']} events")

fig, ax = plt.subplots(figsize=(18, 12))
pitch = Pitch(pitch_type="statsbomb", pitch_color="#2d5a2d",
              line_color="white", linewidth=3)
pitch.draw(ax=ax)

sequence_events = []

for _, event in possession_events.iterrows():
    loc = event.get("location")
    if not (isinstance(loc, list) and len(loc) == 2):
        continue

    ev = {
        "x": loc[0],
        "y": loc[1],
        "event_type": event["event_type"],
        "player": event["player_name"],
        "minute": event["minute"],
        "second": event["second"],
    }

    # Pass end location
    if event["event_type"] == "Pass":
        p = event.get("pass", {})
        end = p.get("end_location")
        if isinstance(end, list) and len(end) == 2:
            ev["end_x"] = end[0]
            ev["end_y"] = end[1]

    # Carry end location
    if event["event_type"] == "Carry":
        c = event.get("carry", {})
        end = c.get("end_location")
        if isinstance(end, list) and len(end) == 2:
            ev["end_x"] = end[0]
            ev["end_y"] = end[1]

    sequence_events.append(ev)

print(f"Found {len(sequence_events)} events with locations")

# ---------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------
event_colors = {
    "Pass": "#4CAF50",
    "Carry": "#FF9800",
    "Ball Receipt*": "#2196F3",
    "Shot": "#F44336",
    "Dribble": "#9C27B0",
    "Pressure": "#795548",
    "Ball Recovery": "#607D8B"
}

for i, ev in enumerate(sequence_events):
    color = event_colors.get(ev["event_type"], "#FFC107")

    ax.add_patch(plt.Circle((ev["x"], ev["y"]), radius=2,
                            color=color, alpha=0.8, zorder=5))

    ax.text(ev["x"], ev["y"], str(i+1),
            color="white", fontweight="bold", fontsize=10,
            ha="center", va="center", zorder=6)

    # Connect to previous
    if i > 0:
        prev = sequence_events[i-1]
        ax.plot([prev["x"], ev["x"]], [prev["y"], ev["y"]],
                color="white", linewidth=3, alpha=0.7, zorder=2)

        ax.annotate("", xy=(ev["x"], ev["y"]), xytext=(prev["x"], prev["y"]),
                    arrowprops=dict(arrowstyle="->", color="white",
                                    alpha=0.8, lw=2), zorder=3)

    # Pass / carry arrows
    if "end_x" in ev:
        ax.plot([ev["x"], ev["end_x"]], [ev["y"], ev["end_y"]],
                color="yellow" if ev["event_type"] == "Pass" else "orange",
                linewidth=2.5, linestyle="--", alpha=0.9, zorder=4)

# ---------------------------------------------------------
# TITLE + STATS
# ---------------------------------------------------------
start = sequence_events[0]
end   = sequence_events[-1]

start_time = f"{start['minute']}:{start['second']:02d}"
end_time   = f"{end['minute']}:{end['second']:02d}"

total_distance = sum(
    np.sqrt((sequence_events[i]["x"] - sequence_events[i-1]["x"])**2 +
            (sequence_events[i]["y"] - sequence_events[i-1]["y"])**2)
    for i in range(1, len(sequence_events))
)

field_progression = end["x"] - start["x"]

title = (
    f"Possession Chain Analysis: {team_name}\n"
    f"Sequence: {start_time}–{end_time} | "
    f"Events: {len(sequence_events)} | "
    f"Distance: {total_distance:.1f} yards | "
    f"Progression: {field_progression:+.1f} yards"
)

ax.text(60, 85, title, ha="center", va="center", fontsize=14,
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=1", facecolor="black", alpha=0.8))
# --- Legend and breakdown box (like your example) ---
event_colors = {
    "Pressure": "#795548",
    "Dispossessed": "#9E9E9E",
    "Carry": "#FF9800",
    "Ball Receipt*": "#2196F3",
    "Interception": "#8BC34A",
    "Pass": "#4CAF50",
    "Ball Recovery": "#607D8B",
    "Duel": "#9C27B0"
}

# Legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color, markersize=10,
               label=etype, markeredgecolor='black')
    for etype, color in event_colors.items()
]

ax.legend(handles=legend_elements, loc='upper left',
          bbox_to_anchor=(0.02, 0.98), fontsize=10,
          facecolor='black', edgecolor='white', labelcolor='white',
          title='Event Types', title_fontsize=11)

# Breakdown box
source_counts = possession_events['event_type'].value_counts()
details_lines = [f"{etype}: {count} events" for etype, count in source_counts.items()]
details_text = "POSSESSION SOURCE BREAKDOWN:\n" + "\n".join(details_lines)

ax.text(5, 10, details_text, ha='left', va='top', fontsize=9,
        color='white', family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig("Figure26_Possession_Chain.png", dpi=300, bbox_inches="tight",
            facecolor="#1a3d1a")

def analyze_defensive_patterns(events_df):
    """
    Analyze defensive action patterns and spatial clustering.
    Compatible with StatsBomb open-data structure.
    """

    defensive_events = [
        "Tackle", "Interception", "Block", "Clearance",
        "Ball Recovery", "Pressure", "Duel"
    ]

    df = events_df.copy()
    df["event_type"] = df["type"].apply(extract_name)
    df["team_name"]  = df["team"].apply(extract_name)
    df["player_name"] = df["player"].apply(lambda x: extract_name(x) if x else None)

    defensive_actions = df[df["event_type"].isin(defensive_events)].copy()
    defensive_analysis = []

    for idx, action in defensive_actions.iterrows():
        loc = action.get("location")
        if not (isinstance(loc, list) and len(loc) == 2):
            continue

        x, y = loc

        # Field thirds
        if x < 40:
            field_third = "Defensive"
        elif x < 80:
            field_third = "Middle"
        else:
            field_third = "Attacking"

        # Lateral zones
        if y < 26.67:
            lateral_zone = "Left"
        elif y < 53.33:
            lateral_zone = "Center"
        else:
            lateral_zone = "Right"

        defensive_analysis.append({
            "action_id": idx,
            "action_type": action["event_type"],
            "player": action["player_name"],
            "team": action["team_name"],
            "minute": action.get("minute", 0),
            "x": x,
            "y": y,
            "field_third": field_third,
            "lateral_zone": lateral_zone,
            "distance_from_own_goal": x,  # assuming goal at x=0
            "under_pressure": bool(action.get("under_pressure", False)),
            "possession": action.get("possession", None),
        })

    return pd.DataFrame(defensive_analysis)
import seaborn as sns
from sklearn.cluster import KMeans

fig = plt.figure(figsize=(20, 16))
defensive_actions_df = analyze_defensive_patterns(events_df)
teams = defensive_actions_df['team'].unique().tolist()

# -------------------------------------------------------------------
# PANEL 1 — SPATIAL CLUSTERING
# -------------------------------------------------------------------
ax1 = plt.subplot(3, 3, 1)
pitch1 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black', linewidth=2)
pitch1.draw(ax=ax1)

locations = defensive_actions_df[['x', 'y']].values

if len(locations) >= 5:
    kmeans = KMeans(n_clusters=min(5, len(locations)), random_state=42, n_init=10)
    clusters = kmeans.fit_predict(locations)

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(max(clusters) + 1):
        pts = locations[clusters == i]
        ax1.scatter(pts[:, 0], pts[:, 1],
                    c=colors[i % len(colors)], alpha=0.7, s=60,
                    label=f'Cluster {i+1}')

        cx, cy = kmeans.cluster_centers_[i]
        ax1.scatter(cx, cy, c='black', marker='x', s=200, linewidth=3)

ax1.set_title('Spatial Clustering of Defensive Actions\n(X marks cluster centers)', fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# -------------------------------------------------------------------
# PANEL 2 — TEAM 1 HEATMAP
# -------------------------------------------------------------------
ax2 = plt.subplot(3, 3, 2)
pitch2 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black', linewidth=2)
pitch2.draw(ax=ax2)

team1 = teams[0]
team1_df = defensive_actions_df[defensive_actions_df['team'] == team1]

if len(team1_df) > 0:
    heat = ax2.hexbin(team1_df['x'], team1_df['y'],
                      gridsize=20, cmap='Reds', alpha=0.8, mincnt=1)
    plt.colorbar(heat, ax=ax2, shrink=0.8, label='Action Density')

ax2.set_title(f'{team1}: Defensive Action Heatmap', fontweight='bold')

# -------------------------------------------------------------------
# PANEL 3 — ACTION TYPE DISTRIBUTION
# -------------------------------------------------------------------
ax3 = plt.subplot(3, 3, 3)
counts = defensive_actions_df['action_type'].value_counts()
ax3.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
ax3.set_title('Defensive Action Types Distribution', fontweight='bold')

# -------------------------------------------------------------------
# PANEL 4 — TEMPORAL DISTRIBUTION
# -------------------------------------------------------------------
ax4 = plt.subplot(3, 3, 4)

df_temp = defensive_actions_df.copy()
df_temp['time_period'] = pd.cut(
    df_temp['minute'],
    bins=[0, 15, 30, 45, 60, 75, 90, 120],
    labels=['0-15', '15-30', '30-45', '45-60', '60-75', '75-90', '90+']
)

temporal_counts = df_temp['time_period'].value_counts().sort_index()
bars = ax4.bar(range(len(temporal_counts)), temporal_counts.values,
               color='lightcoral', alpha=0.8, edgecolor='black')

ax4.set_xticks(range(len(temporal_counts)))
ax4.set_xticklabels(temporal_counts.index, rotation=45)
ax4.set_ylabel('Number of Actions')
ax4.set_title('Temporal Distribution of Defensive Actions', fontweight='bold')

for bar in bars:
    h = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., h + 1,
             f'{int(h)}', ha='center', va='bottom', fontsize=9)

# -------------------------------------------------------------------
# PANEL 5 — FIELD ZONE HEATMAP
# -------------------------------------------------------------------
ax5 = plt.subplot(3, 3, 5)
zone_matrix = defensive_actions_df.groupby(['field_third', 'lateral_zone']).size().unstack(fill_value=0)
sns.heatmap(zone_matrix, annot=True, cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Action Count'})
ax5.set_title('Defensive Actions by Field Zones', fontweight='bold')
ax5.set_ylabel('Field Third')
ax5.set_xlabel('Lateral Zone')

# -------------------------------------------------------------------
# PANEL 6 — PLAYER ANALYSIS
# -------------------------------------------------------------------
ax6 = plt.subplot(3, 3, 6)

player_stats = defensive_actions_df.groupby('player').agg(
    total_actions=('action_type', 'count'),
    aggressive_actions=('field_third', lambda x: (x == 'Attacking').sum()),
    under_pressure=('under_pressure', 'sum')
)

top_players = player_stats.sort_values('total_actions', ascending=False).head(8)

xpos = range(len(top_players))
ax6.bar([x - 0.2 for x in xpos], top_players['total_actions'],
        width=0.4, label='Total Actions', color='lightblue')
ax6.bar([x + 0.2 for x in xpos], top_players['aggressive_actions'],
        width=0.4, label='Attacking Third', color='darkred')

ax6.set_xticks(xpos)
ax6.set_xticklabels([p.split()[-1] for p in top_players.index], rotation=45, ha='right')
ax6.set_ylabel('Number of Actions')
ax6.set_title('Top Defensive Players', fontweight='bold')
ax6.legend()

# -------------------------------------------------------------------
# PANEL 7 — PRESSURE EVENTS
# -------------------------------------------------------------------
ax7 = plt.subplot(3, 3, 7)
pitch7 = Pitch(pitch_type='statsbomb', pitch_color='#f0f8ff', line_color='black', linewidth=2)
pitch7.draw(ax=ax7)

pressure_df = defensive_actions_df[defensive_actions_df['action_type'] == 'Pressure']

for i, team in enumerate(teams):
    pts = pressure_df[pressure_df['team'] == team]
    if len(pts) > 0:
        ax7.scatter(pts['x'], pts['y'], c=['red', 'blue'][i % 2],
                    alpha=0.7, s=40, label=f'{team} Pressure')

ax7.set_title('Pressure Events Spatial Distribution', fontweight='bold')
ax7.legend()

# -------------------------------------------------------------------
# PANEL 8 — FIELD THIRD EFFICIENCY
# -------------------------------------------------------------------
ax8 = plt.subplot(3, 3, 8)

eff = defensive_actions_df.groupby('field_third').agg(
    total_actions=('action_type', 'count'),
    possessions_affected=('possession', 'nunique')
)
eff['efficiency'] = eff['possessions_affected'] / eff['total_actions'] * 100

bars = ax8.bar(eff.index, eff['efficiency'],
               color=['green', 'orange', 'red'], alpha=0.8)

ax8.set_ylabel('Efficiency (%)')
ax8.set_title('Defensive Efficiency by Field Third', fontweight='bold')
ax8.set_ylim(0, 100)

for bar in bars:
    h = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., h + 2,
             f'{h:.1f}%', ha='center', va='bottom', fontweight='bold')

# -------------------------------------------------------------------
# PANEL 9 — TEAM COMPARISON
# -------------------------------------------------------------------
ax9 = plt.subplot(3, 3, 9)

if len(teams) >= 2:
    comp = defensive_actions_df.groupby(['team', 'action_type']).size().unstack(fill_value=0)
    comp.plot(kind='bar', ax=ax9, alpha=0.8)
    ax9.set_title('Team Defensive Profile Comparison', fontweight='bold')
    ax9.set_xlabel('Team')
    ax9.set_ylabel('Number of Actions')
    ax9.legend(title='Action Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax9.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('Figure27_Defensive_Analysis.png', dpi=300, bbox_inches='tight')



def bayesian_player_rating(player_passes):
    """
    Bayesian player passing ability estimation using a Beta-Binomial model.
    
    Parameters:
    - player_passes: DataFrame indexed by player name, with columns:
        'successful_passes', 'total_attempts'
    
    Returns:
    - DataFrame with Bayesian estimates and uncertainty intervals
    """

    # Prior: Beta(2,2) = weakly informative, centred at 0.5
    prior_alpha = 2.0
    prior_beta = 2.0

    # Import here to avoid errors if scipy isn't installed globally
    from scipy.stats import beta

    results = []

    for player, row in player_passes.iterrows():
        successes = row['successful_passes']
        attempts = row['total_attempts']

        # Guard against division by zero
        if attempts == 0:
            empirical_rate = np.nan
            failures = 0
        else:
            empirical_rate = successes / attempts
            failures = attempts - successes

        # Posterior update
        post_alpha = prior_alpha + successes
        post_beta = prior_beta + failures

        # Posterior mean
        bayes_mean = post_alpha / (post_alpha + post_beta)

        # Posterior variance
        post_var = (post_alpha * post_beta) / (
            (post_alpha + post_beta)**2 * (post_alpha + post_beta + 1)
        )

        # Credible intervals
        ci95 = beta.interval(0.95, post_alpha, post_beta)
        ci80 = beta.interval(0.80, post_alpha, post_beta)

        results.append({
            "player": player,
            "empirical_rate": empirical_rate,
            "bayesian_estimate": bayes_mean,
            "uncertainty": np.sqrt(post_var),
            "credible_95_lower": ci95[0],
            "credible_95_upper": ci95[1],
            "credible_80_lower": ci80[0],
            "credible_80_upper": ci80[1],
            "posterior_alpha": post_alpha,
            "posterior_beta": post_beta
        })

    return pd.DataFrame(results)
def bayesian_team_strength(team_goal_data):
    """
    Bayesian estimation of team scoring strength using a Poisson–Gamma model.
    
    Parameters:
    - team_goal_data: DataFrame indexed by team name, with columns:
        'total_goals', 'matches'
    
    Returns:
    - DataFrame with Bayesian strength estimates and credible intervals
    """

    # Prior: Gamma(shape, rate)
    prior_shape = 1.5   # α
    prior_rate  = 1.0   # β

    # Import locally to avoid global dependency issues
    from scipy.stats import gamma

    results = []

    for team, row in team_goal_data.iterrows():
        goals = row['total_goals']
        matches = row['matches']

        # Guard against division by zero
        if matches == 0:
            empirical_rate = np.nan
        else:
            empirical_rate = goals / matches

        # Posterior update (Gamma is conjugate to Poisson)
        post_shape = prior_shape + goals
        post_rate  = prior_rate + matches

        # Posterior mean and variance
        post_mean = post_shape / post_rate
        post_var  = post_shape / (post_rate ** 2)

        # 95% credible interval
        ci_lower, ci_upper = gamma.interval(
            0.95,
            post_shape,
            scale=1 / post_rate
        )

        results.append({
            "team": team,
            "empirical_rate": empirical_rate,
            "bayesian_strength": post_mean,
            "strength_uncertainty": np.sqrt(post_var),
            "credible_lower": ci_lower,
            "credible_upper": ci_upper,
            "posterior_shape": post_shape,
            "posterior_rate": post_rate,
            "matches_observed": matches
        })

    return pd.DataFrame(results)


import pandas as pd
import matplotlib.pyplot as plt

def analyze_temporal_patterns(events_df):
    """
    Analyze temporal patterns in football events throughout match duration.
    Compatible with StatsBomb event data.
    """

    df = events_df.copy()
    

    # Ensure proper time handling
    df["minute_bin"] = df["minute"].fillna(0).astype(int)

    # Ensure event_type exists
    if "event_type" not in df.columns:
        df["event_type"] = df["type"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else str(x)
        )

    # Event frequency per minute
    minute_counts = df.groupby(["minute_bin", "event_type"]).size().unstack(fill_value=0)

    # Overall event density by minute
    total_events_per_minute = df["minute_bin"].value_counts().sort_index()

    # Match phase intensities
    first_third = total_events_per_minute.loc[total_events_per_minute.index < 30].mean() \
        if len(total_events_per_minute.loc[total_events_per_minute.index < 30]) > 0 else 0
    middle_third = total_events_per_minute.loc[
        (total_events_per_minute.index >= 30) & (total_events_per_minute.index < 60)
    ].mean() if len(total_events_per_minute.loc[
        (total_events_per_minute.index >= 30) & (total_events_per_minute.index < 60)
    ]) > 0 else 0
    final_third = total_events_per_minute.loc[total_events_per_minute.index >= 60].mean() \
        if len(total_events_per_minute.loc[total_events_per_minute.index >= 60]) > 0 else 0

    # Rolling averages for trend detection
    rolling_window = 5
    if len(total_events_per_minute) >= rolling_window:
        rolling_mean = total_events_per_minute.rolling(
            window=rolling_window, center=True
        ).mean()
        rolling_std = total_events_per_minute.rolling(
            window=rolling_window, center=True
        ).std()

        volatility_threshold = rolling_mean + 2 * rolling_std
        high_intensity_periods = total_events_per_minute[
            total_events_per_minute > volatility_threshold
        ]
    else:
        rolling_mean = total_events_per_minute
        high_intensity_periods = pd.Series(dtype=float)

    # Event type evolution (kept for completeness, not used in rhythm plot)
    event_evolution = {}
    key_events = ["Pass", "Pressure", "Ball Recovery", "Duel"]

    for event_type in key_events:
        if event_type in minute_counts.columns:
            series = minute_counts[event_type]
            first_half = series.loc[series.index < 45].sum() / 45 \
                if len(series.loc[series.index < 45]) > 0 else 0
            second_half = series.loc[series.index >= 45].sum() / max(
                1, len(series.loc[series.index >= 45])
            ) if len(series.loc[series.index >= 45]) > 0 else 0

            event_evolution[event_type] = {
                "first_half_rate": first_half,
                "second_half_rate": second_half,
                "half_ratio": second_half / max(0.001, first_half),
                "trend": "increasing" if second_half > first_half else "decreasing",
            }

    return {
        "phase_intensities": {
            "first_third": first_third,
            "middle_third": middle_third,
            "final_third": final_third,
        },
        "rolling_analysis": {
            "mean_series": rolling_mean,
            "high_intensity_minutes": high_intensity_periods.index.tolist(),
        },
        "event_evolution": event_evolution,
        "total_timeline": total_events_per_minute,
    }


def plot_match_rhythm(events_df, title="Match Rhythm", filename="match_rhythm.png"):
    """
    Plot match rhythm (events per minute + rolling average)
    and save the figure instead of showing it.
    """

    df = events_df.copy()

    # Ensure minute column exists
    if "minute" not in df.columns:
        print("ERROR: events_df has no 'minute' column.")
        return

    # Clean minute values
    df["minute_bin"] = (
        df["minute"]
        .fillna(0)
        .astype(float)
        .astype(int)
        .clip(lower=0, upper=120)
    )

    # Count events per minute
    total_events = df["minute_bin"].value_counts().sort_index()

    if total_events.empty:
        print("No events found — cannot plot match rhythm.")
        return

    # Rolling average
    rolling_mean = total_events.rolling(window=5, center=True).mean()

    # Create figure
    plt.figure(figsize=(14, 6))

    # Raw event frequency
    plt.plot(
        total_events.index,
        total_events.values,
        color="tab:blue",
        linewidth=1.5,
        label="Events per Minute",
        alpha=0.7,
    )

    # Rolling average
    plt.plot(
        rolling_mean.index,
        rolling_mean.values,
        color="tab:red",
        linewidth=2.5,
        label="5-min Rolling Average",
    )

    plt.title(f"{title}: Event Frequency Timeline", fontsize=16, fontweight="bold")
    plt.xlabel("Match Minute")
    plt.ylabel("Events per Minute")
    plt.grid(alpha=0.2, linestyle="--")
    plt.legend()
    plt.tight_layout()

    # SAVE instead of show
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved match rhythm plot → {filename}")

plot_match_rhythm(events_df, title="Barcelona vs Alavés")

def analyze_network_evolution(events_df, time_window=15):
    """
    Analyze how team passing networks evolve throughout the match.
    Compatible with StatsBomb event data.
    """

    import networkx as nx
    import numpy as np
    import pandas as pd

    df = events_df.copy()

    # ---------------------------------------------------------
    # 1. Ensure event_type exists
    # ---------------------------------------------------------
    if "event_type" not in df.columns:
        df["event_type"] = df["type"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else str(x)
        )

    # ---------------------------------------------------------
    # 2. Filter to passes only
    # ---------------------------------------------------------
    passes = df[df["event_type"] == "Pass"].copy()
    if passes.empty:
        return []

    # ---------------------------------------------------------
    # 3. Ensure player_name and team_name exist
    # ---------------------------------------------------------
    if "player_name" not in passes.columns:
        passes["player_name"] = passes["player"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else "Unknown"
        )

    if "team_name" not in passes.columns:
        passes["team_name"] = passes["team"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else "Unknown"
        )

    passes["passer"] = passes["player_name"].fillna("Unknown")

    # ---------------------------------------------------------
    # 4. Extract pass recipient safely
    # ---------------------------------------------------------
    def get_pass_recipient(row):
        p = row.get("pass", {})
        if isinstance(p, dict):
            r = p.get("recipient")
            if isinstance(r, dict):
                return r.get("name", "Unknown")
        return "Unknown"

    passes["recipient"] = passes.apply(get_pass_recipient, axis=1)
    passes = passes[passes["recipient"] != "Unknown"]

    if passes.empty:
        return []

    # ---------------------------------------------------------
    # 5. Clean minute values
    # ---------------------------------------------------------
    passes["minute"] = (
        passes["minute"]
        .fillna(0)
        .astype(float)
        .astype(int)
        .clip(lower=0, upper=120)
    )

    max_minute = passes["minute"].max()
    time_bins = list(range(0, max_minute + time_window, time_window))

    network_evolution = []

    # ---------------------------------------------------------
    # 6. Loop through time windows
    # ---------------------------------------------------------
    for i in range(len(time_bins) - 1):
        start_minute = time_bins[i]
        end_minute = time_bins[i + 1]

        window_passes = passes[
            (passes["minute"] >= start_minute)
            & (passes["minute"] < end_minute)
        ]

        if len(window_passes) < 5:
            continue

        team_networks = {}

        # ---------------------------------------------------------
        # 7. Build networks per team
        # ---------------------------------------------------------
        for team in window_passes["team_name"].unique():
            team_passes = window_passes[window_passes["team_name"] == team]

            if len(team_passes) < 3:
                continue

            G = nx.DiGraph()

            # Add edges
            for _, row in team_passes.iterrows():
                passer = row["passer"]
                recipient = row["recipient"]

                if G.has_edge(passer, recipient):
                    G[passer][recipient]["weight"] += 1
                else:
                    G.add_edge(passer, recipient, weight=1)

            if len(G.nodes()) > 2:
                # Density
                density = nx.density(G)

                # Clustering
                try:
                    clustering = nx.average_clustering(G.to_undirected())
                except:
                    clustering = 0

                # Centrality
                degree_centrality = nx.degree_centrality(G)
                if degree_centrality:
                    most_central = max(degree_centrality, key=degree_centrality.get)
                    max_centrality = degree_centrality[most_central]
                else:
                    most_central = "Unknown"
                    max_centrality = 0

                # Efficiency
                try:
                    if nx.is_weakly_connected(G):
                        efficiency = 1 / nx.average_shortest_path_length(G)
                    else:
                        largest_cc = max(nx.weakly_connected_components(G), key=len)
                        sub = G.subgraph(largest_cc)
                        efficiency = (
                            1 / nx.average_shortest_path_length(sub)
                            if len(sub) > 1
                            else 0
                        )
                except:
                    efficiency = 0

                team_networks[team] = {
                    "density": density,
                    "clustering": clustering,
                    "max_centrality": max_centrality,
                    "most_central_player": most_central,
                    "efficiency": efficiency,
                    "nodes": len(G.nodes()),
                    "edges": len(G.edges()),
                    "total_passes": len(team_passes),
                }

        network_evolution.append(
            {
                "time_period": f"{start_minute}-{end_minute}",
                "start_minute": start_minute,
                "end_minute": end_minute,
                "team_networks": team_networks,
            }
        )

    return network_evolution
import matplotlib.pyplot as plt

def plot_network_evolution(network_evolution, filename="network_evolution.png"):
    """
    Plot network density evolution for each team across match time windows.
    
    Parameters:
    - network_evolution: output from analyze_network_evolution()
    - filename: file to save the figure
    """

    if not network_evolution or len(network_evolution) == 0:
        print("No network evolution data available.")
        return

    # Collect teams
    all_teams = set()
    for window in network_evolution:
        all_teams.update(window["team_networks"].keys())

    # Prepare time axis
    time_labels = [w["time_period"] for w in network_evolution]
    x = range(len(time_labels))

    plt.figure(figsize=(14, 6))

    # Plot each team
    for team in sorted(all_teams):
        densities = []
        for window in network_evolution:
            team_data = window["team_networks"].get(team, None)
            if team_data:
                densities.append(team_data["density"])
            else:
                densities.append(None)

        # Convert None → NaN for plotting
        densities = [d if d is not None else float("nan") for d in densities]

        plt.plot(
            x,
            densities,
            marker="o",
            linewidth=2,
            label=team
        )

    plt.xticks(x, time_labels, rotation=45, ha="right")
    plt.xlabel("Match Time Window (minutes)")
    plt.ylabel("Network Density")
    plt.title("Network Structure Evolution Across Match", fontsize=16, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved network evolution plot → {filename}")
network_evo = analyze_network_evolution(events_df, time_window=15)
plot_network_evolution(network_evo, filename="Figure_NetworkEvolution.png")


def analyze_performance_decline(events_df, player_id=None):
    """
    Analyze performance decline indicators throughout match duration.
    Focus on pass accuracy, event frequency, and spatial coverage.
    """

    from scipy import stats
    import numpy as np
    import pandas as pd

    df = events_df.copy()

    # ---------------------------------------------------------
    # Ensure event_type exists
    # ---------------------------------------------------------
    if "event_type" not in df.columns:
        df["event_type"] = df["type"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else str(x)
        )

    # ---------------------------------------------------------
    # Filter passes
    # ---------------------------------------------------------
    passes = df[df["event_type"] == "Pass"].copy()
    if passes.empty:
        return None

    # Ensure team_name and player_name exist
    if "team_name" not in passes.columns:
        passes["team_name"] = passes["team"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else "Unknown"
        )

    if "player_name" not in passes.columns:
        passes["player_name"] = passes["player"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else "Unknown"
        )

    # ---------------------------------------------------------
    # Pass success flag
    # ---------------------------------------------------------
    def is_pass_successful(row):
        p = row.get("pass", {})
        if isinstance(p, dict):
            outcome = p.get("outcome")
            if isinstance(outcome, dict):
                return outcome.get("name") != "Incomplete"
        return True

    passes["successful"] = passes.apply(is_pass_successful, axis=1)

    # Clean minute values
    passes["minute_bin"] = (
        passes["minute"].fillna(0).astype(float).astype(int).clip(0, 120) // 15 * 15
    )

    decline_analysis = {}

    # ---------------------------------------------------------
    # TEAM‑LEVEL PERFORMANCE DECLINE
    # ---------------------------------------------------------
    for team in passes["team_name"].unique():
        team_passes = passes[passes["team_name"] == team]

        grouped = team_passes.groupby("minute_bin").agg(
            successes=("successful", "sum"),
            attempts=("successful", "count"),
            accuracy=("successful", "mean"),
        )

        grouped["period"] = grouped.index

        valid = grouped["attempts"] >= 5
        if valid.sum() < 3:
            continue

        data = grouped[valid]

        # Accuracy trend
        slope, intercept, r, p, _ = stats.linregress(
            data["period"], data["accuracy"]
        )

        # Frequency trend
        freq_slope, _, _, _, _ = stats.linregress(
            data["period"], data["attempts"]
        )

        decline_analysis[team] = {
            "accuracy_trend": {
                "slope": slope,
                "r_squared": r**2,
                "p_value": p,
                "interpretation": (
                    "declining" if slope < -0.01 and p < 0.05
                    else "improving" if slope > 0.01 and p < 0.05
                    else "stable"
                ),
            },
            "frequency_trend": {
                "slope": freq_slope,
                "interpretation": (
                    "decreasing" if freq_slope < -0.5
                    else "increasing" if freq_slope > 0.5
                    else "stable"
                ),
            },
            "time_periods": data.to_dict("records"),
        }

    # ---------------------------------------------------------
    # FATIGUE INDICATORS
    # ---------------------------------------------------------
    intensity_events = df[df["event_type"].isin(["Pressure", "Duel", "Tackle"])]

    fatigue_indicators = {}

    if not intensity_events.empty:
        intensity_events["minute_bin"] = (
            intensity_events["minute"].fillna(0).astype(int).clip(0, 120) // 15 * 15
        )

        intensity_by_period = intensity_events.groupby(
            ["minute_bin", "team_name"]
        ).size().unstack(fill_value=0)

        for team in intensity_by_period.columns:
            series = intensity_by_period[team]
            if len(series) < 3:
                continue

            slope, _, r, p, _ = stats.linregress(series.index, series.values)

            fatigue_indicators[team] = {
                "intensity_decline": slope,
                "p_value": p,
                "r_squared": r**2,
                "interpretation": (
                    "significant_decline" if slope < -0.5 and p < 0.05
                    else "moderate_decline" if slope < -0.2
                    else "stable"
                ),
            }

    return {
        "team_performance_trends": decline_analysis,
        "fatigue_indicators": fatigue_indicators,
        "analysis_summary": {
            "total_teams_analyzed": len(decline_analysis),
            "significant_decline_teams": sum(
                1
                for t in decline_analysis.values()
                if t["accuracy_trend"]["interpretation"] == "declining"
            ),
        },
    }
def plot_performance_decline(performance_data, filename="performance_decline.png"):
    """
    Plot pass accuracy decline across match periods for each team.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    trends = performance_data["team_performance_trends"]

    if not trends:
        print("No performance decline data available.")
        return

    plt.figure(figsize=(12, 6))

    for team, data in trends.items():
        periods = [p["period"] for p in data["time_periods"]]
        accuracy = [p["accuracy"] for p in data["time_periods"]]

        plt.plot(
            periods,
            accuracy,
            marker="o",
            linewidth=2,
            label=f"{team} (trend: {data['accuracy_trend']['interpretation']})",
        )

    plt.xlabel("Match Minute (binned in 15-min windows)")
    plt.ylabel("Pass Accuracy")
    plt.title("Performance Decline Analysis", fontsize=16, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved performance decline plot → {filename}")
perf = analyze_performance_decline(events_df)
plot_performance_decline(perf, filename="Figure_PerformanceDecline.png")


def analyze_substitution_impact(events_df, lineups_data=None):
    """
    Analyze the impact of substitutions on team performance metrics.
    """

    import pandas as pd
    import numpy as np
    from scipy import stats

    df = events_df.copy()

    # ---------------------------------------------------------
    # Ensure event_type exists
    # ---------------------------------------------------------
    if "event_type" not in df.columns:
        df["event_type"] = df["type"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else str(x)
        )

    # Ensure team_name exists
    if "team_name" not in df.columns:
        df["team_name"] = df["team"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else "Unknown"
        )

    # ---------------------------------------------------------
    # Identify substitutions
    # ---------------------------------------------------------
    subs = df[df["event_type"] == "Substitution"].copy()
    if subs.empty:
        return {"substitution_impacts": [], "summary_statistics": None}

    substitution_analysis = []

    # ---------------------------------------------------------
    # Loop through substitutions
    # ---------------------------------------------------------
    for _, sub in subs.iterrows():
        sub_minute = sub["minute"]
        team = sub["team_name"]

        if pd.isna(sub_minute) or sub_minute < 15 or sub_minute > 75:
            continue

        # Define windows
        pre_start = max(0, sub_minute - 15)
        pre_end = sub_minute
        post_start = sub_minute
        post_end = min(90, sub_minute + 15)

        team_events = df[df["team_name"] == team]

        pre_events = team_events[
            (team_events["minute"] >= pre_start)
            & (team_events["minute"] < pre_end)
        ]

        post_events = team_events[
            (team_events["minute"] >= post_start)
            & (team_events["minute"] < post_end)
        ]

        metrics = {}

        # ---------------------------------------------------------
        # Event frequency
        # ---------------------------------------------------------
        pre_rate = len(pre_events) / 15
        post_rate = len(post_events) / 15

        metrics["event_frequency"] = {
            "pre_substitution": pre_rate,
            "post_substitution": post_rate,
            "change": post_rate - pre_rate,
            "percent_change": ((post_rate - pre_rate) / max(0.1, pre_rate)) * 100,
        }

        # ---------------------------------------------------------
        # Pass accuracy
        # ---------------------------------------------------------
        def is_pass_successful(row):
            p = row.get("pass", {})
            if isinstance(p, dict):
                outcome = p.get("outcome")
                if isinstance(outcome, dict):
                    return outcome.get("name") != "Incomplete"
            return True

        metrics["pass_accuracy"] = {}

        for label, ev in [("pre", pre_events), ("post", post_events)]:
            passes = ev[ev["event_type"] == "Pass"]
            if not passes.empty:
                acc = passes.apply(is_pass_successful, axis=1).mean()
            else:
                acc = np.nan
            metrics["pass_accuracy"][f"{label}_substitution"] = acc

        # Compute accuracy change
        pre_acc = metrics["pass_accuracy"].get("pre_substitution", np.nan)
        post_acc = metrics["pass_accuracy"].get("post_substitution", np.nan)

        if not np.isnan(pre_acc) and not np.isnan(post_acc):
            metrics["pass_accuracy"]["change"] = post_acc - pre_acc
            metrics["pass_accuracy"]["percent_change"] = (
                (post_acc - pre_acc) / max(0.1, pre_acc)
            ) * 100

        # ---------------------------------------------------------
        # Defensive intensity
        # ---------------------------------------------------------
        defensive_events = ["Pressure", "Tackle", "Interception", "Block"]

        pre_def = pre_events[pre_events["event_type"].isin(defensive_events)]
        post_def = post_events[post_events["event_type"].isin(defensive_events)]

        pre_def_rate = len(pre_def) / 15
        post_def_rate = len(post_def) / 15

        metrics["defensive_intensity"] = {
            "pre_substitution": pre_def_rate,
            "post_substitution": post_def_rate,
            "change": post_def_rate - pre_def_rate,
            "percent_change": ((post_def_rate - pre_def_rate) / max(0.1, pre_def_rate))
            * 100,
        }

        # ---------------------------------------------------------
        # Statistical significance (event frequency)
        # ---------------------------------------------------------
        if len(pre_events) >= 5 and len(post_events) >= 5:
            contingency = [
                [len(pre_events), 15],
                [len(post_events), 15],
            ]
            try:
                chi2, p, _, _ = stats.chi2_contingency(contingency)
                sig = p < 0.05
            except:
                sig = False
        else:
            sig = False

        substitution_analysis.append(
            {
                "substitution_minute": sub_minute,
                "team": team,
                "metrics_comparison": metrics,
                "statistical_significance": sig,
                "sample_sizes": {
                    "pre_events": len(pre_events),
                    "post_events": len(post_events),
                },
            }
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    if substitution_analysis:
        summary = {
            "total_substitutions_analyzed": len(substitution_analysis),
            "statistically_significant_impacts": sum(
                1 for s in substitution_analysis if s["statistical_significance"]
            ),
            "average_event_frequency_change": np.mean(
                [s["metrics_comparison"]["event_frequency"]["change"] for s in substitution_analysis]
            ),
            "positive_impact_substitutions": sum(
                1
                for s in substitution_analysis
                if s["metrics_comparison"]["event_frequency"]["change"] > 0
            ),
        }
    else:
        summary = None

    return {
        "substitution_impacts": substitution_analysis,
        "summary_statistics": summary,
    }
def plot_substitution_impact(sub_data, filename="substitution_impact.png"):
    """
    Plot substitution impact:
    - X: substitution minute
    - Y: event frequency change
    - Color: pass accuracy change
    - Size: defensive intensity change
    """

    import matplotlib.pyplot as plt
    import numpy as np

    impacts = sub_data["substitution_impacts"]
    if not impacts:
        print("No substitution impact data available.")
        return

    minutes = []
    event_changes = []
    acc_changes = []
    def_changes = []

    for s in impacts:
        minutes.append(s["substitution_minute"])
        event_changes.append(s["metrics_comparison"]["event_frequency"]["change"])
        acc_changes.append(s["metrics_comparison"]["pass_accuracy"].get("change", 0))
        def_changes.append(s["metrics_comparison"]["defensive_intensity"]["change"])

    sizes = [abs(x) * 300 for x in def_changes]  # bubble size

    plt.figure(figsize=(12, 6))

    scatter = plt.scatter(
        minutes,
        event_changes,
        c=acc_changes,
        s=sizes,
        cmap="coolwarm",
        edgecolor="black",
        alpha=0.85,
    )

    plt.colorbar(scatter, label="Pass Accuracy Change")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)

    plt.xlabel("Substitution Minute")
    plt.ylabel("Event Frequency Change (events/min)")
    plt.title("Substitution Impact Analysis", fontsize=16, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved substitution impact plot → {filename}")
subs = analyze_substitution_impact(events_df)
plot_substitution_impact(subs, filename="Figure_SubstitutionImpact.png")
