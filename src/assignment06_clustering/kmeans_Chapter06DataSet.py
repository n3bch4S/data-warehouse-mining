import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.axes import Axes

file_path: str = "../../datasets/assignment06_clustering/Chapter06DataSet.csv"
data: DataFrame = pd.read_csv(file_path)
print(data.head())

scaler: StandardScaler = StandardScaler()
scaled_data: NDArray = scaler.fit_transform(data)

optimal_k: int = 4
kmeans: KMeans = KMeans(n_clusters=optimal_k, random_state=0)
clusters: NDArray = kmeans.fit_predict(scaled_data)
data["Cluster"] = clusters

fig: Figure = plt.figure(figsize=(10, 8))
ax: Axes = fig.add_subplot(111, projection="3d")
ax.scatter(
    scaled_data[:, 0],
    scaled_data[:, 1],
    scaled_data[:, 2],
    c=clusters,
    cmap="viridis",
)
ax.set_title("Chapter06DataSet")
ax.set_xlabel("Weight")
ax.set_ylabel("Cholesterol")
ax.set_zlabel("Gender")
plt.show()
