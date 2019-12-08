rm -r ../storage
CUDA_VISIBLE_DEVICES=0 allennlp train \
	-s ../storage \
	-o '{"trainer": {"cuda_device": -1}}' \
	training_config/bidaf.jsonnet

