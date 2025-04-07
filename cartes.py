import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import osmnx as ox


df = pd.DataFrame(
    {
        "City": ["Yonne"],
        "Country": ["France"],
        "Latitude": [ 47.806 ],
        "Longitude": [3.639],
    }
)

# Départements français (version simplifiée)
url_departements = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
departements = gpd.read_file(url_departements)


# Régions françaises
url_regions = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-version-simplifiee.geojson"
regions = gpd.read_file(url_regions)
print("ile de france", regions["code"][0])
# print(regions)

# Créer la figure et les axes
fig, ax = plt.subplots(figsize=(12, 12))

# Afficher toutes les régions, ou pas? commentez svp
# regions.plot(ax=ax, edgecolor="black", facecolor="lightgrey", linewidth=0.5)

# Mettre en évidence l'ile de france
# ile_de_france = regions[regions["code"] == "11"]
# ile_de_france.plot(ax=ax, edgecolor="black", facecolor="red", linewidth=1)

# Mettre en évidence la bourgogne franche comté
bourgogne= regions[regions["code"] == "27"]
bourgogne.plot(ax=ax, edgecolor="black", facecolor="gray", linewidth=1)


# Ajouter un titre et supprimer les axes
plt.title("Régions françaises avec l'Île-de-France en surbrillance", fontsize=16)
ax.axis('off')

# Ajouter le point pour Yonne
ax.plot(3.639, 47.806, 'bo', markersize=8, label="Venoy")
ax.annotate('Venoy', xy=(3.639, 47.806), xytext=(5, 5), textcoords="offset points")

plt.tight_layout()
plt.show()
