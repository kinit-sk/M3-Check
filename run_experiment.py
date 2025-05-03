import sys
import hydra

sys.path.append('code')
from routines import Experiment
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    experiment = Experiment(config)
    experiment()

if __name__ == "__main__":
    main()
