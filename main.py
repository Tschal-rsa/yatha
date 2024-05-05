import hydra

from config import Config, store_config
from interface import Interface


@hydra.main(version_base=None, config_name='config', config_path='config')
def my_app(cfg: Config) -> float | None:
    interface = Interface(cfg)
    interface.run()
    return interface.objective


if __name__ == '__main__':
    store_config()
    my_app()
