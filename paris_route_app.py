import streamlit as st
import networkx as nx
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
import pickle
import ast
from geopy.distance import geodesic

# --- Load preprocessed graph with beauty scores ---
st.markdown(
    """
    <style>
        body {
            background-color: #f8f9fa;
        }
        .stApp {
            background-color: #f8f9fa;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.image("WanderPath logo.png", width=250)
st.title("Paris Scenic Walking Route Recommender 🇫🇷")
st.markdown("Plan a beautiful walking route through Paris! 🚶🏽‍♀️🥐")

@st.cache_resource
def load_graph():
    with open("graph_with_beauty.pkl", "rb") as f:
        G_proj = pickle.load(f)
    return G_proj

@st.cache_data
def load_pois():
    df = pd.read_csv("POIs_Paris.csv")
    df = df.drop_duplicates()
    if df['Types'].apply(type).eq(str).any():
        df['Types'] = df['Types'].apply(ast.literal_eval)
    return df

G_proj = load_graph()
df_pois = load_pois()

# --- Sidebar for user input ---
poi_names = df_pois['DisplayName'].tolist()

start_poi = st.selectbox("Select your starting point:", poi_names)
end_poi = st.selectbox("Select your destination:", poi_names)

# Flatten all unique types for the multiselect menu
all_types = sorted(set(t for sublist in df_pois['Types'] for t in sublist))
categories = st.multiselect("Select types of places to include:", all_types)

# Let user select number of places per type
type_limits = {}
for cat in categories:
    count = st.number_input(f"How many places of type '{cat}'?", min_value=1, max_value=5, value=1, step=1)
    type_limits[cat] = count

# --- Helper functions ---
def find_nearest_node(lat, lon, G):
    return ox.distance.nearest_nodes(G, lon, lat)

def compute_beautiful_route(G, start_lat, start_lon, end_lat, end_lon, via_pois):
    start_node = find_nearest_node(start_lat, start_lon, G)
    end_node = find_nearest_node(end_lat, end_lon, G)

    # Intermediate POIs
    poi_nodes = []
    for _, row in via_pois.iterrows():
        node = find_nearest_node(row['Latitude'], row['Longitude'], G)
        poi_nodes.append((node, row['DisplayName']))

    full_route = [start_node]
    current_node = start_node
    for node, _ in poi_nodes:
        path = nx.shortest_path(G, current_node, node, weight='cost')
        full_route.extend(path[1:])
        current_node = node

    # Final segment
    path = nx.shortest_path(G, current_node, end_node, weight='cost')
    full_route.extend(path[1:])
    return full_route

def create_map(G, route_nodes, df_all_pois, start_coords, end_coords):
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=14)

    for _, row in df_all_pois.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=row['DisplayName'],
            icon=folium.Icon(color='gray', icon='info-sign')
        ).add_to(m)

    route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route_nodes]
    folium.PolyLine(route_coords, color="red", weight=4.5, opacity=0.9).add_to(m)

    folium.Marker(location=start_coords, popup="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(location=end_coords, popup="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    return m

def select_nearest_pois(start_coords, end_coords, df, type_limits):
    selected_pois = pd.DataFrame(columns=df.columns)
    for cat, limit in type_limits.items():
        pois_of_type = df[df['Types'].apply(lambda types: cat in types)].copy()
        pois_of_type['distance_to_start'] = pois_of_type.apply(lambda row: geodesic((row['Latitude'], row['Longitude']), start_coords).meters, axis=1)
        pois_of_type['distance_to_end'] = pois_of_type.apply(lambda row: geodesic((row['Latitude'], row['Longitude']), end_coords).meters, axis=1)
        pois_of_type['total_dist'] = pois_of_type['distance_to_start'] + pois_of_type['distance_to_end']
        selected = pois_of_type.nsmallest(limit, 'total_dist')
        selected_pois = pd.concat([selected_pois, selected])
    return selected_pois

# --- Generate route ---
if 'route' not in st.session_state:
    st.session_state.route = None
    st.session_state.poi_list = []

if st.button("Generate Scenic Route"):
    start_row = df_pois[df_pois['DisplayName'] == start_poi].iloc[0]
    end_row = df_pois[df_pois['DisplayName'] == end_poi].iloc[0]

    filtered_pois = select_nearest_pois(
        (start_row['Latitude'], start_row['Longitude']),
        (end_row['Latitude'], end_row['Longitude']),
        df_pois,
        type_limits
    )

    route = compute_beautiful_route(
        G_proj,
        start_row['Latitude'], start_row['Longitude'],
        end_row['Latitude'], end_row['Longitude'],
        filtered_pois
    )

    st.session_state.route = route
    st.session_state.poi_list = pd.concat([start_row.to_frame().T, filtered_pois, end_row.to_frame().T])
    st.success(f"Route with {len(route)} nodes generated.")

if st.session_state.route:
    route_map = create_map(
        G_proj,
        st.session_state.route,
        st.session_state.poi_list,
        (st.session_state.poi_list.iloc[0]['Latitude'], st.session_state.poi_list.iloc[0]['Longitude']),
        (st.session_state.poi_list.iloc[-1]['Latitude'], st.session_state.poi_list.iloc[-1]['Longitude'])
    )
    _ = st_folium(route_map, width=800, height=600)

    st.markdown("### Points of Interest on the Route")
    st.write(st.session_state.poi_list['DisplayName'].tolist())
