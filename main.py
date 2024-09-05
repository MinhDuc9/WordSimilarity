import os
import openai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_openai_api_key():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


def set_openai_key(api_key):
    openai.api_key = api_key


def get_openai_embedding(phrase):
    response = openai.Embedding.create(input=phrase, model="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    return np.array(embedding)


# Calculate cosine similarity, euclidean distance, and dot product
def calculate_similarity_metrics(central_vector, other_vectors):
    cos_sim = cosine_similarity([central_vector], other_vectors)[0]
    euclidean_dist = euclidean_distances([central_vector], other_vectors)[0]
    dot_products = np.dot(central_vector, np.array(other_vectors).T)

    return cos_sim, euclidean_dist, dot_products


# Create a table of comparison
def create_comparison_table(phrases, cos_sim, euclidean_dist, dot_products):
    return pd.DataFrame(
        {
            "Phrase": phrases,
            "Cosine Similarity": cos_sim,
            "Euclidean Distance": euclidean_dist,
            "Dot Product": dot_products,
        }
    )


# Perform t-SNE for dimensionality reduction
def perform_tsne(vectors, n_components=2, perplexity=2):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(vectors)


# Plot the t-SNE result in 2D
def plot_tsne_result_2d(tsne_result, phrases):
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])

    for i, phrase in enumerate(phrases):
        plt.annotate(phrase, (tsne_result[i, 0], tsne_result[i, 1]))

    plt.title("t-SNE Graph")
    plt.xlabel("X dimension")
    plt.ylabel("Y dimension")
    plt.show()


def main():
    api_key = load_openai_api_key()
    set_openai_key(api_key)

    central_phrase = "khao hoc mi tinh"
    phrases = ["computer science", "KHMT", "data science", "khoa học máy tính"]

    central_vector = get_openai_embedding(central_phrase)
    other_vectors = [get_openai_embedding(phrase) for phrase in phrases]

    cos_sim, euclidean_dist, dot_products = calculate_similarity_metrics(
        central_vector, other_vectors
    )

    comparison_table = create_comparison_table(
        phrases, cos_sim, euclidean_dist, dot_products
    )
    print(comparison_table)

    # Perform t-SNE for dimensionality reduction
    vectors = np.vstack([central_vector] + other_vectors)
    tsne_result = perform_tsne(vectors)

    # Plot t-SNE result in 2D
    plot_tsne_result_2d(tsne_result, [central_phrase] + phrases)


if __name__ == "__main__":
    main()
