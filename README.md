# WanderPath: Scenic Walking Route Recommender for Urban Exploration

## 🚶🏽‍♀️🥐 Try the Web App: [WanderPath Live Demo](https://wander-path-cl01.streamlit.app/)


## 📚 Table of Contents
- [📌 Project Overview](#-project-overview)  
- [🚀 Features](#-features)
- [🧠 Skills Demonstrated](#-skills-demonstrated)
- [🛠 Tools & Libraries Used](#-tools--libraries-used)
- [🌱 Future Enhancements](#-future-enhancements)

# 📌 Project Overview
WanderPath is a Python-based application designed to generate intelligent, "beautiful" walking routes through urban environments. Unlike standard shortest-path navigators, WanderPath prioritizes scenic and pleasant paths, taking into account proximity to green spaces, water bodies, and notable landmarks. This project demonstrates advanced geospatial data processing, graph theory applications, and algorithmic route planning.

The current implementation focuses on the charming arrondissements of Paris, allowing users to define a start and end point, with options to include intermediate points of interest (POIs) like museums and cafes.

<img width="740" height="483" alt="Captura de pantalla 2025-07-16 a la(s) 5 18 02 p  m" src="https://github.com/user-attachments/assets/ba32d457-90cf-4e48-8fd0-8a267b7686ae" />

# 🚀 Features
* **Custom "Beauty" Scoring:** Edges of the city's street network are assigned a dynamic "beauty" score. This score is influenced by:
  * Proximity to parks and green areas.
  * Proximity to rivers, streams, and other water features.
  * Proximity to monuments and relevant sights.
  * Street type (e.g., pedestrian streets receive a higher beauty bonus).

<img width="516" height="895" alt="paris_network_beauty_parks_water_monuments_and_route" src="https://github.com/user-attachments/assets/eeaea08a-bbbb-44b0-8662-b934e3d0b8a5" />

* **Scenic Route Planning:** Utilizes the calculated beauty scores to find paths that minimize a "cost" (length divided by beauty), effectively recommending routes that are not just short, but also visually appealing and enjoyable.
* **Point of Interest (POI) Integration:** Reads POI data from a CSV file, allowing for the selection of specific start, end, and intermediate waypoints for the route.
* **Dynamic Waypoint Ordering:** Implements a simplified "closest next" heuristic to order intermediate POIs, preventing inefficient zig-zagging and ensuring a sensible flow to the journey.
* **Geospatial Data Handling:** Leverages OpenStreetMap data to construct detailed street networks and extract natural and man-made features.
* **Route Details Output:** Provides a tabular summary of the suggested route, detailing each edge (street segment) with its start/end nodes, length, beauty score, and cost.
* **Visualization:** Generates a clear map visualizing the street network, identified green spaces, water bodies, monuments, selected POIs, and the final recommended walking route.
* **Persistent Graph Storage:** Saves the processed graph (with added beauty and cost attributes) to a pickle file for faster loading and re-use.

# 🧠 Skills Demonstrated

* **Geospatial Data Processing:** Expertise in acquiring, processing, and analyzing geographical data using libraries like OSMnx and GeoPandas.
* **Graph Theory & Network Analysis:** Application of NetworkX for building, manipulating, and querying complex graph structures (street networks).
* **Data Manipulation & Analysis:** Proficient use of Pandas for handling tabular data, filtering, and preparing data for geospatial operations.
* **Algorithmic Design:** Development of custom algorithms for beauty score calculation, robust geometry validation, and a simplified Traveling Salesperson Problem (TSP) heuristic for waypoint ordering.
* **Data Visualization:** Creation of informative and visually appealing maps using Matplotlib to represent complex geospatial data and routes.
* **Python Programming:** Strong foundational skills in Python, including error handling, modular code design, and working with external libraries.
* **Computational Geometry:** Use of Shapely for precise geometric operations like buffering and intersection tests, with robust error handling for real-world geospatial data imperfections.

# 🛠 Tools & Libraries Used

* **OSMnx:** For downloading, parsing, and visualizing street networks and other OpenStreetMap data.
* **NetworkX:** For graph creation, manipulation, and pathfinding algorithms.
* **GeoPandas:** For working with geospatial dataframes and performing spatial operations.
* **Pandas:** For general data manipulation and analysis.
* **Matplotlib:** For creating static visualizations of the maps and routes.
* **Shapely:** For computational geometry operations and validating geometries.
* **ast module:** For safely evaluating string representations of Python literals (e.g., lists of POI types).
* **pickle module:** For serializing and deserializing Python objects, used here to save the processed graph.

# 🌱 Future Enhancements

* **Interactive Web Interface:** Develop a web-based UI (e.g., with Flask/Django + Leaflet/Folium) to allow users to select start/end points and POI types interactively.
* **Expand to Other Locations:** Adapt the application to generate scenic routes in various other cities and urban areas globally, allowing users to input a location of their choice.
* **Advanced TSP Solvers:** Integrate more sophisticated Traveling Salesperson Problem algorithms for optimal ordering of a larger number of intermediate waypoints.
* **Dynamic Feature Selection:** Allow users to choose which "beauty" factors (parks, water, monuments, pedestrian streets) they want to prioritize.
* **Time-Varying Data:** Incorporate real-time data (e.g., weather, temporary street closures) to influence route recommendations.
* **User Preferences:** Implement user profiles to personalize beauty preference
* **Accessibility Features:** Add options to consider accessibility, such as avoiding steep inclines or stairs.
