import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.axes import Axes

file_path: str = "../../datasets/assignment06_clustering/Mall_Customers.csv"
data: DataFrame = pd.read_csv(file_path).drop(["CustomerID"], axis=1)
data["Gender"] = data["Gender"].replace({"Male": 1, "Female": 0})
print(data.head())

scaler: StandardScaler = StandardScaler()
scaled_data: NDArray = scaler.fit_transform(data)

optimal_k: int = 5
kmeans: KMeans = KMeans(n_clusters=optimal_k, random_state=0)
clusters: NDArray = kmeans.fit_predict(scaled_data)

data["Cluster"] = clusters

fig: Figure = plt.figure(figsize=(10, 8))
ax: Axes = fig.add_subplot(111, projection="3d")

# project Age to 3d space
ax.scatter(
    scaled_data[:, 0],
    scaled_data[:, 2],
    scaled_data[:, 3],
    c=clusters,
    cmap="viridis",
)
ax.set_title("Mall_Customers")
ax.set_xlabel("Gender")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
plt.show()
