import re
import random
import requests

def preprocess_tweet(tweet):
    """Preprocesses a single tweet by removing unnecessary elements and converting to lowercase."""
    tweet = re.sub(r'^\d+\t\d+-\d+-\d+\s\d+:\d+:\d+\t', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    return tweet.lower().strip()

def jaccard_distance(set1, set2):
    """Calculate the Jaccard distance between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - intersection / union

def compute_centroid(cluster):
    """Compute the centroid of a cluster based on minimizing the sum of Jaccard distances."""
    min_dist_sum = float('inf')
    centroid = cluster[0]
    for tweet in cluster:
        dist_sum = sum(jaccard_distance(set(tweet.split()), set(other.split())) for other in cluster)
        if dist_sum < min_dist_sum:
            min_dist_sum = dist_sum
            centroid = tweet
    return centroid

def k_means(tweets, k):
    """Perform K-means clustering with Jaccard distance."""
    centroids = random.sample(tweets, k)
    iterations = 10  # Set fixed number of iterations for simplicity
    for _ in range(iterations):
        clusters = {i: [] for i in range(k)}
        for tweet in tweets:
            distances = [jaccard_distance(set(tweet.split()), set(centroids[i].split())) for i in range(k)]
            closest_centroid_index = distances.index(min(distances))
            clusters[closest_centroid_index].append(tweet)
        
        new_centroids = [compute_centroid(clusters[i]) for i in range(k)]
        if new_centroids == centroids:
            break
        centroids = new_centroids

    sse = 0
    for i in range(k):
        centroid_set = set(centroids[i].split())
        sse += sum(jaccard_distance(set(tweet.split()), centroid_set) ** 2 for tweet in clusters[i])
    
    return clusters, sse

# Main program to load data, preprocess, and cluster

def main():
    file_url = "https://raw.githubusercontent.com/rahulvinay029/CS6375_ML_HW2/main/usnewshealth.txt"
    #file_path = r"https://raw.githubusercontent.com/rahulvinay029/CS6375_ML_HW2/main/usnewshealth.txt"  # Update this to the path of your tweet file
    response = requests.get(file_url)
    tweets = []
    #with open(file_path, 'r', encoding='utf-8') as file:
        #for line in file:
            #cleaned_tweet = preprocess_tweet(line)
            #if cleaned_tweet:
                #tweets.append(cleaned_tweet)
    if response.status_code == 200:
        lines = response.text.split('\n')
        for line in lines:
            cleaned_tweet = preprocess_tweet(line)
            if cleaned_tweet:
                tweets.append(cleaned_tweet)

    # Values of K to test
    ks = [5, 10, 15, 20, 25]
    for k in ks:
        clusters, sse = k_means(tweets, k)
        print(f'For K={k}, SSE={sse}, Cluster Sizes={[len(cluster) for cluster in clusters.values()]}')

if __name__ == '__main__':
    main()
