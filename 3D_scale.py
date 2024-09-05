from main import load_openai_api_key, perform_tsne, set_openai_key, get_openai_embedding, calculate_similarity_metrics, create_comparison_table
import matplotlib.pyplot as plt
import numpy as np

# Perform t-SNE for dimensionality reduction
def plot_tsne_result_3d(tsne_result, phrases):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(tsne_result[0, 0], tsne_result[0, 1], tsne_result[0, 2], c='red', marker='o', s=100, label='Central Phrase')
    ax.scatter(tsne_result[1:, 0], tsne_result[1:, 1], tsne_result[1:, 2], c='blue', marker='o', s=50, label='Other Phrases')

    # Annotate each point with the corresponding phrase
    for i, phrase in enumerate(phrases):
        ax.text(tsne_result[i, 0], tsne_result[i, 1], tsne_result[i, 2], phrase)

    ax.set_title("3D Graph")
    ax.set_xlabel("X dimension")
    ax.set_ylabel("Y dimension")
    ax.set_zlabel("Z dimension")

    ax.legend()

    plt.show()

def main():
    api_key = load_openai_api_key()
    set_openai_key(api_key)

    central_phrase = "khoa học máy tính"
    phrases = ["computer science", "KHMT", "data science", "khoa học máy tính"]

    central_vector = get_openai_embedding(central_phrase)
    other_vectors = [get_openai_embedding(phrase) for phrase in phrases]

    # Calculate similarity metrics
    cos_sim, euclidean_dist, dot_products = calculate_similarity_metrics(central_vector, other_vectors)

    comparison_table = create_comparison_table(phrases, cos_sim, euclidean_dist, dot_products)
    print(comparison_table)

    # Perform t-SNE for dimensionality reduction in 3D
    vectors = np.vstack([central_vector] + other_vectors)
    tsne_result = perform_tsne(vectors, n_components=3)  # Perform t-SNE for 3D

    # Plot t-SNE result in 3D
    plot_tsne_result_3d(tsne_result, [central_phrase] + phrases)


if __name__ == "__main__":
    main()
    