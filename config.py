import pathlib
import dataclasses
import yaml
import inspect


@dataclasses.dataclass
class YamlConfig:
    def save(self, config_path: pathlib.Path):
        """ Export config as YAML file """
        assert config_path.parent.exists(), f'directory {config_path.parent} does not exist'

        def convert_dict(data):
            for key, val in data.items():
                if isinstance(val, pathlib.Path):
                    data[key] = str(val)
                if isinstance(val, dict):
                    data[key] = convert_dict(val)
            return data

        with open(config_path, 'w') as f:
            yaml.dump(convert_dict(dataclasses.asdict(self)), f)

    @classmethod
    def load(cls, config_path: pathlib.Path):
        """ Load config from YAML file """
        assert config_path.exists(), f'YAML config {config_path} does not exist'

        def convert_from_dict(parent_cls, data):
            for key, val in data.items():
                child_class = parent_cls.__dataclass_fields__[key].type
                if child_class == pathlib.Path:
                    data[key] = pathlib.Path(val)
                if inspect.isclass(child_class) and issubclass(child_class, YamlConfig):
                    data[key] = child_class(**convert_from_dict(child_class, val))
            return data

        with open(config_path) as f:
            config_data = yaml.full_load(f)
            # recursively convert config item to YamlConfig
            config_data = convert_from_dict(cls, config_data)
            return cls(**config_data)


@dataclasses.dataclass
class PathConfig:
    log_dir: pathlib.Path


@dataclasses.dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int


@dataclasses.dataclass
class TrainConfig:
    manual_seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    betas: list[float]
    weight_decay: float
    lr_decay: float
    save_epochs: int
    validation_epochs: int
    logging_steps: int


@dataclasses.dataclass
class SLMConfig(YamlConfig):
    dataset: str
    tokenizer: str
    path: PathConfig
    model: ModelConfig
    train: TrainConfig
