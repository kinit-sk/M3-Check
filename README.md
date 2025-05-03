# M3-Check

Official repository for paper "Multimodal and Multilingual Fact-Checked Article Retrieval".

Experiments can be run using the `run_experiment.py` script by specifying some command line arguments, mainly the experiment config file to use, for instance this code:
```bash
python run_experiment.py --config-name "opti_free_concat"
```
runs the experiment with the config file `configs/opti_free_concat.yaml`.

To make repeat runs with different random seeds, you can use the `--seed` argument. For example:
```bash
for seed in {0..4}; do
    python run_experiment.py --config-name "linear_projection" +args.seed=$seed
done
```

Results are logged using `MLFlow` by default, but can also be logged using `Weights & Biases` by modifying the `args.logger` argument. For more details, please, inspect `_base_config` in `code/routines.py`.
