import json
import matplotlib.pyplot as plt
import argparse


def load_json(file_path):
    """
    Load the JSON file containing probabilities and labels.
    
    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: List of entries from the JSON file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def analyze_spread(data):
    """
    Extract probabilities and true labels for visualization.
    
    Args:
        data (list): List of dictionaries with probabilities and true labels.

    Returns:
        tuple: Probabilities and corresponding true labels.
    """
    probabilities = [entry["probabilities"]["Human"] for entry in data]
    true_labels = [entry["true_label"] for entry in data]
    return probabilities, true_labels


def visualize_distribution_with_histogram(probabilities, true_labels):
    """
    Visualize the row distribution of probabilities and add a histogram below to show concentration.
    
    Args:
        probabilities (list): Probabilities for the "Human" class.
        true_labels (list): Corresponding true labels.
    """
    # Convert true_labels into colors: 0 -> red, 1 -> green
    colors = ['red' if label == 0 else 'green' for label in true_labels]
    
    # Create a figure with two subplots (row distribution + histogram)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 2]})
    
    # Row distribution
    axes[0].scatter(probabilities, [0] * len(probabilities), c=colors, alpha=0.6, s=50)
    axes[0].set_xlim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_title("Row Distribution of Human Probabilities Colored by True Label")
    axes[0].grid(alpha=0.3, axis='x')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='True Label: 0 (AI)')
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True Label: 1 (Human)')
    axes[0].legend(handles=[red_patch, green_patch], loc="upper left")
    
    # Histogram
    bins = 20  # Number of bins
    human_probs = [p for p, label in zip(probabilities, true_labels) if label == 1]
    ai_probs = [p for p, label in zip(probabilities, true_labels) if label == 0]
    
    axes[1].hist(human_probs, bins=bins, color='green', alpha=0.6, label='True Label: 1 (Human)')
    axes[1].hist(ai_probs, bins=bins, color='red', alpha=0.6, label='True Label: 0 (AI)')
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability of Human")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Histogram of Human Probabilities")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze and visualize probability results.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name, e.g., pythia-160m, pythia-1b, llama-3.2-1b-instruct.")
    args = parser.parse_args()
    
    # Path to the probabilities JSON file
    input_path = f"probabilities_results_{args.model_name}.json"
    
    # Load data and analyze spread
    data = load_json(input_path)
    probabilities, true_labels = analyze_spread(data)
    
    # Visualize the distribution and histogram
    visualize_distribution_with_histogram(probabilities, true_labels)


if __name__ == "__main__":
    main()
