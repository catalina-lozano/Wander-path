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

# Chatbot specific imports
from dotenv import load_dotenv
import os
from llama_index.core.chat_engine import SimpleChatEngine, ContextChatEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json 

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="WanderPal Paris")

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

# --- Load preprocessed graph with beauty scores ---
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

# --- Helper functions for Route Recommender ---
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

# --- Route Recommender UI ---
st.header("🚶‍♀️ Plan Your Scenic Route")
col1, col2 = st.columns([1, 1])

with col1:
    poi_names = df_pois['DisplayName'].tolist()
    start_poi = st.selectbox("Select your starting point:", poi_names, key="start_poi_route")
    end_poi = st.selectbox("Select your destination:", poi_names, key="end_poi_route")

with col2:
    all_types = sorted(set(t for sublist in df_pois['Types'] for t in sublist))
    categories = st.multiselect("Select types of places to include:", all_types, key="categories_route")
    
    type_limits = {}
    for cat in categories:
        count = st.number_input(f"How many places of type '{cat}'?", min_value=1, max_value=5, value=1, step=1, key=f"count_{cat}_route")
        type_limits[cat] = count

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
    st_folium(route_map, width=1000, height=600)

    st.markdown("### Points of Interest on the Route")
    st.write(st.session_state.poi_list['DisplayName'].tolist())


# --- Chatbot Integration ---
st.header("💬 Ask WanderPal about Parisian Activities!")

# Load API key
load_dotenv()
api_key = st.secrets["API_groq_cloud"]
HF_Token = st.secrets["HF_TOKEN"]

# Load Paris Activities Data
@st.cache_data
def load_paris_activities():
    df = pd.read_csv('paris_activities.csv')
    # Ensure 'lat_lon' column is processed if it's stored as string representation of dict
    if 'lat_lon' in df.columns and df['lat_lon'].apply(type).eq(str).any():
        df['lat_lon'] = df['lat_lon'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else None)
    return df

df_paris_activities = load_paris_activities()

# Data Preprocessing for RAG (Chatbot)
@st.cache_resource
def prepare_rag_documents(df):
    documents = []
    for index, row in df.iterrows():
        title_text = row.get('title', 'No Title Available')
        lead_text = row.get('lead_text', '')
        description_text = row.get('description', '')
        date_description_text = row.get('date_description', '')
        
        # --- Start of FIX for qfap_tags and universe_tags ---
        # Handle qfap_tags
        qfap_tags_raw = row.get('qfap_tags')
        qfap_tags_list = []
        if isinstance(qfap_tags_raw, str):
            try:
                # Attempt to parse as a list
                parsed_tags = ast.literal_eval(qfap_tags_raw)
                if isinstance(parsed_tags, list):
                    qfap_tags_list = parsed_tags
            except (ValueError, SyntaxError):
                # If parsing fails, treat the string as a single tag or ignore if empty
                if qfap_tags_raw.strip():
                    qfap_tags_list = [qfap_tags_raw.strip()]
        elif isinstance(qfap_tags_raw, list):
            qfap_tags_list = qfap_tags_raw
        # Convert list of tags to a comma-separated string
        qfap_tags_text = ', '.join(qfap_tags_list)

        # Handle universe_tags - apply similar logic
        universe_tags_raw = row.get('universe_tags')
        universe_tags_list = []
        if isinstance(universe_tags_raw, str):
            try:
                parsed_tags = ast.literal_eval(universe_tags_raw)
                if isinstance(parsed_tags, list):
                    universe_tags_list = parsed_tags
            except (ValueError, SyntaxError):
                if universe_tags_raw.strip():
                    universe_tags_list = [universe_tags_raw.strip()]
        elif isinstance(universe_tags_raw, list):
            universe_tags_list = universe_tags_raw
        # Convert list of tags to a comma-separated string
        universe_tags_text = ', '.join(universe_tags_list)
        # --- End of FIX ---

        category_text = row.get('category', '')
        
        address_name = row.get('address_name', '')
        address_street = row.get('address_street', '')
        address_zipcode = row.get('address_zipcode', '')
        address_city = row.get('address_city', '')
        full_address = f"{address_name}, {address_street}, {address_zipcode} {address_city}".strip(', ').replace(', ,', ',')

        price_type = row.get('price_type', '')
        price_detail = row.get('price_detail', '')
        access_type = row.get('access_type', '')
        
        audience = row.get('audience', '')
        childrens = row.get('childrens', False)
        group = row.get('group', False)
        
        full_text_content = f"Title: {title_text}\n"
        if lead_text: full_text_content += f"Summary: {lead_text}\n"
        if description_text: full_text_content += f"Details: {description_text}\n"
        if category_text: full_text_content += f"Category: {category_text}\n"
        if qfap_tags_text: full_text_content += f"Tags: {qfap_tags_text}\n"
        if universe_tags_text: full_text_content += f"Universe Tags: {universe_tags_text}\n"
        if full_address: full_text_content += f"Location: {full_address}\n"
        if date_description_text: full_text_content += f"When: {date_description_text}\n"
        if price_type: full_text_content += f"Price: {price_type}"
        if price_detail: full_text_content += f" ({price_detail})\n"
        else: full_text_content += "\n"
        if access_type: full_text_content += f"Access Type: {access_type}\n"
        if audience: full_text_content += f"Audience: {audience}\n"
        if childrens: full_text_content += f"Good for children: Yes\n"
        if group: full_text_content += f"Good for groups: Yes\n"
        if row.get('event_indoor'): full_text_content += f"Indoor event: Yes\n"
        if row.get('event_pets_allowed'): full_text_content += f"Pets allowed: Yes\n"
        if row.get('url'): full_text_content += f"More Info: {row['url']}\n"
        
        lat = None
        lon = None
        lat_lon_data = row.get('lat_lon')
        if isinstance(lat_lon_data, dict):
            lat = lat_lon_data.get('lat')
            lon = lat_lon_data.get('lon')

        metadata = {
            "id": row.get('id', None),
            "event_id": row.get('event_id', None),
            "title": title_text,
            "category": category_text,
            "main_tags": qfap_tags_text,
            "universe_tags": universe_tags_text,
            "address_name": address_name,
            "address_street": address_street,
            "address_zipcode": address_zipcode,
            "address_city": address_city,
            "full_address": full_address,
            "url": row.get('url', None),
            "lat": lat,
            "lon": lon,
            "date_start": row.get('date_start', None),
            "date_end": row.get('date_end', None),
            "date_description": date_description_text,
            "price_type": price_type,
            "access_type": access_type,
            "audience": audience,
            "childrens": childrens,
            "group": group,
            "event_indoor": row.get('event_indoor', False),
            "event_pets_allowed": row.get('event_pets_allowed', False),
            "source": "Open Data Paris - Que Faire",
            "updated_at": row.get('updated_at', None),
            "rank": row.get('rank', None),
            "weight": row.get('weight', None),
            "pmr": row.get('pmr', False),
            "blind": row.get('blind', False),
            "deaf": row.get('deaf', False),
            "sign_language": row.get('sign_language', False),
            "mental": row.get('mental', False),
        }
        metadata = {k: v for k, v in metadata.items() if v is not None and v != ''}
        documents.append(Document(text=full_text_content.strip(), metadata=metadata))
    return documents

@st.cache_resource
def setup_chatbot(api_key, _documents): # Added underscore here
    # Initialize LLM and Embedding Model
    llm = Groq(model="llama3-70b-8192", api_key=api_key)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2",
                                      token=HF_Token)

    # Set LlamaIndex Global Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 111000
    Settings.chunk_overlap = 20

    persist_dir = "./paris_activities_index"
    if not os.path.exists(persist_dir):
        # Create and persist index if it doesn't exist
        index = VectorStoreIndex.from_documents(_documents) # Use _documents here
        index.storage_context.persist(persist_dir=persist_dir)
        st.success("Chatbot index created and persisted.")
    else:
        # Load index from storage
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        st.info("Chatbot index loaded from storage.")
    
    retriever = index.as_retriever()
    
    project_system_prompts = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are WanderPal, a friendly and knowledgeable AI assistant designed to help users discover "
                "interesting events, activities, and places in Paris, France. "
                "Your goal is to provide accurate and helpful information based ONLY on the context provided "
                "from the Paris activities and places database. "
                "Do NOT invent information or hallucinate details beyond the provided context. "
                "If the information requested is not available in the provided context, clearly state that you "
                "do not have that information."
                "When answering, be concise yet informative. Prioritize key details like event titles, "
                "descriptions, dates, locations, categories, and price information."
                "If a user asks about something outside the scope of Parisian events/places (e.g., current news, "
                "personal opinions, general knowledge), politely redirect them to your purpose."
            ),
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Hello! I'm WanderPal, your guide to Parisian adventures. What are you looking for today?",
        ),
    ]

    context_chat_engine = ContextChatEngine.from_defaults(
        retriever=retriever,
        prefix_messages=project_system_prompts,
        chat_mode="condense_question"
    )
    return context_chat_engine, project_system_prompts[1].content # Return engine and initial greeting

if api_key:
    documents_for_rag = prepare_rag_documents(df_paris_activities)
    context_chat_engine, initial_chatbot_greeting = setup_chatbot(api_key, documents_for_rag)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": initial_chatbot_greeting}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask WanderPal..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        response = context_chat_engine.chat(prompt)
        msg = {"role": "assistant", "content": response.response}
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg["content"])

else:
    st.error("API Key for Groq not found. Please ensure it's set in your .env file.")
