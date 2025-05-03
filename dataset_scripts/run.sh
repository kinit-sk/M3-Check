# Example usage

# Generate the data index to be used by further script runs:
python make_embeds.py --dataset_path=../../disai_dataset/ --type=none --data=posts --index=auto
python make_embeds.py --dataset_path=../../disai_dataset/ --type=none --data=articles --index=auto

# Build CLIP embeddings for images:
python make_embeds.py --dataset_path=../../disai_dataset/ --type=image --data=posts --index=auto --model=CLIP
python make_embeds.py --dataset_path=../../disai_dataset/ --type=image --data=articles --index=auto --model=CLIP

# Build CLIP embeddings for texts:
python make_embeds.py --dataset_path=../../disai_dataset/ --type=text --data=posts --index=auto --model=CLIP
python make_embeds.py --dataset_path=../../disai_dataset/ --type=text --data=articles --index=auto --model=CLIP

# Build CLIP embeddings for OCRs:
python make_embeds.py --dataset_path=../../disai_dataset/ --type=ocr --data=posts --index=auto --model=CLIP
python make_embeds.py --dataset_path=../../disai_dataset/ --type=ocr --data=articles --index=auto --model=CLIP
