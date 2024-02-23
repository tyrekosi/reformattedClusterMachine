from utils.dataUtils import dbscan_clustering, integrate_small_clusters, output_data

EPS_dbscan = 0.04
CLUSTER_MIN = 0 # Broken...
COORDINATE_FILE = 'sources/everycoord.txt'

def read_coords(path):
    dat = {}
    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            iden = parts[0]
            crds = list(map(float, parts[1:]))
            dat[iden] = crds
    return dat

def main():
    coordinates = read_coords(COORDINATE_FILE)

    clusters, cluster_metrics = dbscan_clustering(coordinates, eps=EPS_dbscan)
    clusters = integrate_small_clusters(coordinates, clusters, eps=EPS_dbscan, min_cluster_size=CLUSTER_MIN)

    output_data(coordinates, clusters, cluster_metrics)

if __name__ == "__main__":
    main()