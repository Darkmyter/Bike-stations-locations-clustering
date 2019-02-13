from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import webbrowser
import argparse
import folium


def cluster(df, n_clusters, model):
	""" Cluster the bike stations using K-means based on the location.
	Two models are used: K-means and Agglomerative Hierarchical Clustering

	Parameters
	----------
	df: dataframe
		Bike stations data.
	n_clusters: int
		Number of clusters.
	model: str
		The model to train.
	Returns
	-------
	list
		centroids of the the clusters
	list
		label of each bike station
	"""
	if model == "kmeans":
		kmeans = KMeans(n_clusters=n_clusters)

		# fit kmeans to the location of bike stations
		kmeans.fit(df.loc[:,["longitude","latitude"]])

		centroids = kmeans.cluster_centers_
		labels = kmeans.labels_

	else:
		hc = AgglomerativeClustering(n_clusters=n_clusters, affinity = 'euclidean', linkage = 'ward')
		labels = hc.fit_predict(df.loc[:,["longitude","latitude"]])

	return labels

def map(df, labels, hex_colors):
	""" Creates an interactive map of the city of Brisbane with the labeled bike stations.

	Parameters
	----------
	df: dataframe
		Bike stations data.
	labels: list
		labels of the bike stations
	hex_colors: list
		The color representing each cluster
	
	Returns
	-------
	map
		folium map of city of Brisbane.
	"""
	
	folium_map = folium.Map(location=[-27.470125, 153.021072], zoom_start=13,tiles='Stamen Toner')
	for station in range(df.shape[0]):
		folium.Circle(location=df.loc[station,["latitude","longitude"]], fill=True, radius=50, color=hex_colors[labels[station]]).add_child(folium.Popup(df.loc[station,"name"])).add_to(folium_map)
	return folium_map

def plot_clusters(df, labels, colors, output):
	""" Plots the culsters of bike stations.

	Parameters
	----------
	df: dataframe
		Bike stations data.
	labels: list
		labels of the bike stations
	centroids: list
		centroids of the clusters
	hex_colors: list
		The color representing each cluster
	output: str
		path where figure is saved.
	"""

	plt.scatter(df.loc[:,"longitude"],df.loc[:,"latitude"], color=[colors[l_] for l_ in labels], label=labels)
	#plt.scatter(centroids[:, 0],centroids[:, 1], color=[c for c in colors[:len(centroids)]], marker = "x", s=150, linewidths = 5, zorder = 10)
	plt.axis('off')
	if output != 'skip':
		plt.savefig(output + 'bike_stations')
	plt.show()


def main(n_clusters, model, plot, output, input_file):
	""" Main function that trains and plots the figuers.

	Parameters
	----------
	n_clusters: int
		Number of clusters.
	model: str
		The model to train.
	plot: str
		Method of displaying the results.
	output: str
		path where figure is saved.	
	"""
	df = pd.read_json(input_file)

	colors = sns.color_palette('Set1', n_clusters)
	hex_colors = colors.as_hex()

	labels = cluster(df, n_clusters, model)

	if plot == "map":
		if output == 'skip':
			file = 'map.html'
		else:
			file = output + 'map.html'
		map(df, labels, hex_colors).save(file)
		webbrowser.open(file)
	else:
		plot_clusters(df, labels, colors, output)




if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='A program for clustering the bike stations in Brisbane using their location. \
	 First, the model is trained using either K-means or Agglomerative Hierarchical Clustering \
	 for a user choosen number of cluster. And second, the clsuters are displayed in using an intertactive html map or matplotlib standard figure. \
	 The intertactive map need to be saved on the computer in order to be displayed and requires the package folium. The standard figure can also be \
	 saved by pasing a path using the command lien options.')
	parser.add_argument("-n", "--n_clusters", type=int, default=4, help='Number of clusters.')
	parser.add_argument("-l", "--plot", type=str, default="figure", choices=['map', 'figure'], help="Method of desplaying cluters: \
		- map: interractive html map. - figrue: standard figrue.")
	parser.add_argument("-m", "--model", type=str, default="knn", choices=['kmeans', 'ahc'], help="Model used for clustering the bike stations.")
	parser.add_argument('-p', '--path', type=str, default='skip', help="Path where the figure need to be saved.")
	parser.add_argument('-i', '--file_path', type=str, help='the path of data file in json.', required=True)

	args = vars(parser.parse_args())

	n_clusters = args["n_clusters"]
	plot = args["plot"]
	model = args["model"]
	output = args['path']
	input_file = args['file_path']

	main(n_clusters, model, plot, output, input_file)