import os

from config import Config

from .proto import ProtoPreprocessor


class ProtoPUMC(ProtoPreprocessor):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    def get_source_output_pairs(self) -> list[tuple[str, str]]:
        source_output_pairs = []
        splits = ['train', 'val']
        for split in splits:
            source_root = os.path.join(self.cfg.dataset.root, split)
            output_root = os.path.join(self.cfg.dataset.preprocess_path, split)
            for category in os.listdir(source_root):
                source_base = os.path.join(source_root, category)
                output_base = os.path.join(output_root, category)
                source_output_pairs.append((source_base, output_base))
        return source_output_pairs
