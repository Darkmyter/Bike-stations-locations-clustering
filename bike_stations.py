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
	Two models are used: K-means and Agglomerative Hierarchical Clustering.

	Parameters
	----------
	df: dataframe
		Bike stations data.
	n_clusters: int
		Number of clusters.
	model: str
		The model used for training.
	Returns
	-------
	list
		centroids of the the clusters
	list
		label of each bike station
	"""
	if model == "kmeans":
		kmeans = KMeans(n_clusters=n_clusters)

		# fit kmeans based on the location of bike stations
		kmeans.fit(df.loc[:, ["longitude","latitude"]])
		#stations labels
		labels = kmeans.labels_
	else:
		# fit AC based on the location of bike stations
		hc = AgglomerativeClustering(n_clusters=n_clusters, affinity = 'euclidean', linkage = 'ward')
		labels = hc.fit_predict(df.loc[:, ["longitude", "latitude"]])

	return labels

def map(df, labels, hex_colors, city):
	""" Creates an interactive map of the city with the labeled bike stations.

	Parameters
	----------
	df: dataframe
		Bike stations data.
	labels: list
		labels of the bike stations
	hex_colors: list
		The color representing each cluster
	city: list of flaot
		lattitude and longitude of the city.
	
	Returns
	-------
	map
		folium map of city of Brisbane.
	"""
	#create the object folium map
	folium_map = folium.Map(location=city, zoom_start=13, tiles='Stamen Toner')
	#add bike station markers
	for station in range(df.shape[0]):
		folium.Circle(location=df.loc[station, ["latitude", "longitude"]], fill=True, radius=50, color=hex_colors[labels[station]]).add_child(folium.Popup(df.loc[station, "name"])).add_to(folium_map)
	return folium_map

def plot_clusters(df, labels, colors, output):
	""" Plots the culsters of bike stations.

	Parameters
	----------
	df: dataframe
		Bike stations data.
	labels: list of int
		labels of the bike stations
	hex_colors: list of colors
		The color representing each cluster
	output: str
		location to save figure to.
	"""
	#plot the bike station location with corresponding labels
	plt.scatter(df.loc[:, "longitude"],df.loc[:, "latitude"], color=[colors[l_] for l_ in labels], label=labels)
	plt.axis('off')

	#save figure
	if output != 'skip':
		plt.savefig(output + 'bike_stations')
	
	plt.show()


def main(n_clusters, model, plot, output, input_file, city):
	""" Main function that trains, plots the figuers and saves new file.

	Parameters
	----------
	n_clusters: int
		Number of clusters.
	model: str
		The model to train.
	plot: str
		Method of displaying the results.
	output: str
		location to figure figure to.	
	input_file: str
		location of the bike stations json file.
	city: list of float
		lattitude and longitude of the city.
	"""
	#read data from jsob file
	df = pd.read_json(input_file)

	colors = sns.color_palette('Set1', n_clusters)
	hex_colors = colors.as_hex()

	#perform clustering 
	labels = cluster(df, n_clusters, model)

	#plot the clusters
	if plot == "map":
		if output == 'skip':
			file = 'map.html'
		else:
			file = output + 'map.html'
		map(df, labels, hex_colors, city).save(file)
		webbrowser.open(file)
	else:
		plot_clusters(df, labels, colors, output)

	if output != "skip":
		df['label'] = labels
		df.to_json(output + "labeld_bike_stations.json")


if __name__ == "__main__":
	#description
	parser = argparse.ArgumentParser(description='A script for clustering bike stations in a using their location. \
	 First, the model is trained using either K-means or Agglomerative Hierarchical Clustering \
	 for a user choosen number of cluster. And second, the clusters are displayed using an intertactive html map or matplotlib standard figure. \
	 The intertactive map need to be saved on the computer in order to be displayed and requires the package folium. The standard figure can also be \
	 saved by pasing a path using the command lien options. Finaly, new json file is saved with labeled stations. Data can be found in JCdecaux open data.')
	#arguments
	parser.add_argument("-n", "--n_clusters", type=int, default=4, help='number of clusters.')
	parser.add_argument("-l", "--plot", type=str, default="figure", choices=['map', 'figure'], help="method of desplaying cluters: \
		- map: interractive html map. - figrue: standard figrue.")
	parser.add_argument("-m", "--model", type=str, default="knn", choices=['kmeans', 'ahc'], help="model used for clustering the bike stations.")
	parser.add_argument('-p', '--path', type=str, default='skip', help="path where the figure need to be saved.")
	parser.add_argument('-i', '--file_path', type=str, help='path of the json data file.', required=True)
	parser.add_argument('-c', '--city', type=float, nargs=2, default=[-27.470125, 153.021072], help='lattitude and longitude corrdinates of the city')

	#retreiving arguments
	args = vars(parser.parse_args())
	n_clusters = args["n_clusters"]
	plot = args["plot"]
	model = args["model"]
	output = args['path']
	input_file = args['file_path']
	city = args['city']

	main(n_clusters, model, plot, output, input_file,city)