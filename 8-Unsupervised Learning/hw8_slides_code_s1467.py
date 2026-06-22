import pandas as pd
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Configuration
student_id = "s1467"
corpus_file = f"hw8_corpus_{student_id}.csv"
results_file = f"hw8_results_{student_id}.json"
silhouette_img = f"hw8_slides_plot_{student_id}.png"
output_pdf = f"hw8_slides_{student_id}.pdf"

print(f"Generating slides for {student_id}...")

# Load Data
try:
    df = pd.read_csv(corpus_file)
    # Calculate lengths for EDA
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
except FileNotFoundError:
    print(f"Error: {corpus_file} not found. Ensure it is in the same directory.")
    sys.exit(1)

# Load JSON results (to get the Optimal K and labels)
try:
    with open(results_file, 'r') as f:
        results_data = json.load(f)
        best_k = results_data['K_hat']
        # Map cluster labels back to dataframe
        cluster_map = {item['document_id']: item['cluster'] for item in results_data['clusters']}
        df['cluster'] = df['document_id'].map(cluster_map)
except FileNotFoundError:
    print(f"Warning: {results_file} not found. Please run the main analysis code first.")
    sys.exit(1)

with PdfPages(output_pdf) as pdf:
    
    # ==========================================
    # SLIDE 1: Data & EDA
    # ==========================================
    fig1 = plt.figure(figsize=(11, 8.5))
    fig1.suptitle('Data', fontsize=18, fontweight='bold', y=0.95)
    
    # Text Block for Summary Statistics
    stats_text = (
        f"Corpus Summary Statistics:\n"
        f"• Total Documents: {len(df)}\n"
        f"• Average Word Count: {df['word_count'].mean():.1f} words\n"
        f"• Min Word Count: {df['word_count'].min()} words\n"
        f"• Max Word Count: {df['word_count'].max()} words\n"
        f"\nPreprocessing Steps:\n"
        f"• Lowercasing & punctuation removal\n"
        f"• English stop-word filtering\n"
        f"• Frequency bounds: min_df=2, max_df=0.95"
    )
    fig1.text(0.265, 0.87, stats_text, fontsize=14, va='top', family='monospace', 
              bbox=dict(facecolor='#f0f0f0', edgecolor='gray', pad=10.0))
    
    # Plot: Word Count Distribution
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.45]) # [left, bottom, width, height]
    ax1.hist(df['word_count'], bins=20, color='skyblue', edgecolor='black')
    ax1.set_title('Document Word Count Distribution')
    ax1.set_xlabel('Number of Words')
    ax1.set_ylabel('Frequency')
    ax1.grid(axis='y', alpha=0.7)
    
    pdf.savefig(fig1)
    plt.close(fig1)

    # ==========================================
    # SLIDE 2: Methodology & Results
    # ==========================================
    fig2 = plt.figure(figsize=(11, 8.5))
    fig2.suptitle('Methodology & Results', fontsize=18, fontweight='bold', y=0.95)
    
    methodology_text = (
        "METHODOLOGY:\n"
        "1. Representation: TF-IDF (Term Frequency-Inverse Document Frequency) was used to\n"
        "   vectorize the text, naturally down-weighting ubiquitous vocabulary.\n\n"
        "2. Dimension Reduction: Truncated SVD (Latent Semantic Analysis) reduced the highly\n"
        "   sparse text vectors down to 50 principal components to isolate latent topics and\n"
        "   discard noise.\n\n"
        "3. Clustering: K-Means clustering was applied with n_init=20 random restarts to\n"
        "   avoid local minima.\n\n"
        "4. Libraries Used: pandas, numpy, scikit-learn, matplotlib.\n\n"
        "ASSUMPTIONS:\n"
        "• Documents strictly belong to exactly one topic (Hard Clustering assumption).\n"
        "• TF-IDF assumes bag-of-words (word order does not dictate topic).\n\n"
        "RESULTS:\n"
        f"• Optimal Number of Topics (K-hat) = {best_k}\n"
        "• Chosen by maximizing the average Silhouette Score over a search grid of K=[2, 15]."
    )
    fig2.text(0.1, 0.85, methodology_text, fontsize=14, va='top', linespacing=1.5)
    
    pdf.savefig(fig2)
    plt.close(fig2)

    # ==========================================
    # SLIDE 3: Visualization & Interpretations
    # ==========================================
    fig3 = plt.figure(figsize=(11, 8.5))
    fig3.suptitle('Slide 3: Clustering Visualizations & Topics', fontsize=18, fontweight='bold', y=0.95)
    
    # Left Subplot (Top): Silhouette Score Image
    ax3_left = fig3.add_axes([0.05, 0.48, 0.4, 0.40])
    if os.path.exists(silhouette_img):
        img = plt.imread(silhouette_img)
        ax3_left.imshow(img)
        ax3_left.axis('off')
        ax3_left.set_title("Optimal K Selection (Silhouette)", pad=10)
    else:
        ax3_left.text(0.5, 0.5, f"[Missing {silhouette_img}]", ha='center', va='center')
        ax3_left.axis('off')

    # Right Subplot (Top): 2D Dot Plot
    ax3_right = fig3.add_axes([0.55, 0.48, 0.4, 0.40])
    ax3_right.set_title("2D LSA Projection", pad=10)
    
    # Fast TF-IDF / SVD for the visualization
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.95)
    X_tfidf = vectorizer.fit_transform(df['text'])
    svd_2d = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd_2d.fit_transform(X_tfidf)
    
    scatter = ax3_right.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='tab20', s=50, alpha=0.8)
    ax3_right.set_xlabel("LSA Component 1")
    ax3_right.set_ylabel("LSA Component 2")
    ax3_right.grid(True, linestyle='--', alpha=0.5)
    
    # Internal minimal legend to map colors to numbers
    legend1 = ax3_right.legend(*scatter.legend_elements(), title="Cluster ID", 
                               loc='best', fontsize='small')
    ax3_right.add_artist(legend1)

    # Extract dynamic topic interpretations
    terms = vectorizer.get_feature_names_out()
    cluster_labels_dict = {}
    unique_clusters = sorted(df['cluster'].unique())
    
    for c in unique_clusters:
        idx = df['cluster'] == c
        # Get mean vector for this specific cluster to find its most important words
        mean_vec = X_tfidf[idx].mean(axis=0).A1
        # Grab top 4 words
        top_idx = mean_vec.argsort()[-4:][::-1]
        top_words = [terms[i] for i in top_idx]
        cluster_labels_dict[c] = f"Cluster {c:2d}: {', '.join(top_words)}"

    # Format the extracted topics into two distinct columns for the bottom of the slide
    half_point = (len(unique_clusters) + 1) // 2
    col1_text = "\n".join([cluster_labels_dict[c] for c in unique_clusters[:half_point]])
    col2_text = "\n".join([cluster_labels_dict[c] for c in unique_clusters[half_point:]])

    # Text Block (Bottom Half)
    fig3.text(0.05, 0.38, "Cluster Interpretations (Top TF-IDF Terms):", fontsize=14, fontweight='bold', va='top')
    fig3.text(0.05, 0.32, col1_text, fontsize=12, va='top', family='monospace', linespacing=2.0)
    fig3.text(0.55, 0.32, col2_text, fontsize=12, va='top', family='monospace', linespacing=2.0)
    
    pdf.savefig(fig3)
    plt.close(fig3)

print(f"Successfully generated presentation: {output_pdf}")