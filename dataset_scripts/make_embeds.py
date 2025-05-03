import sys
sys.path.append('../code')

import argparse
from embeds import make_embeds

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="A script to embed MultiClaim data.")

    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset\'s root directory')
    parser.add_argument('--embeddings_dir', type=str, default=None, help='Directory where embeddings should be stored. By default, they go under dataset_path/embeddings')
    parser.add_argument('--image_dir', type=str, default='images', help='Directory containing images')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use for data loading')
    parser.add_argument('--device', type=str, default=None, help='Device to use for embeddings')

    parser.add_argument('--type', type=str, choices=['image', 'text', 'ocr', 'ocr_combined', 'ocr_all_combined', 'oldocr_all_combined', 'none'], required=True, help='Type of data to embed: image, text, ocr_combined, ocr_all_combined or oldocr_all_combined or none if you just want to compute the index')
    parser.add_argument('--data', type=str, choices=['posts', 'articles'], required=True, help='Which data to embed: posts, articles')
    parser.add_argument('--language', type=str, default='original', help='Language to use for text embeddings: original, translated')
    
    parser.add_argument('--model', type=str, default='CLIP', help='Model to use for embeddings')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embeddings')
    parser.add_argument('--no-compile', action='store_true', help='Flag to disable compilation of the model')

    parser.add_argument('--deduplication_hash', type=str, default='sha', help='Hash function to use for deduplication: sha, phash, None')
    parser.add_argument('--output', type=str, default=None, help='Overrides the path of the embeddings output file.')

    parser.add_argument('--index', type=str, default=None, help='Path to a precomputed index file for the data')

    args = parser.parse_args()
    return make_embeds(args)

if __name__ == "__main__":
    main()
