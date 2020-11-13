import torch
import torch.utils.data

from detectron2.data import DatasetCatalog, build_batch_data_loader
from detectron2.data.build import worker_init_reset_seed, trivial_batch_collator
from detectron2.data.samplers import TrainingSampler, InferenceSampler
from detectron2.engine import DefaultTrainer
from torch.distributed import get_world_size

from easy_attributes.utils.trainer_utils import get_batch_size


class CustomTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        sampler = TrainingSampler(len(dataset))

        batch_size = get_batch_size(cfg.SOLVER.IMS_PER_BATCH)

        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=True
        )  # drop_last so the batch always have the same size
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            # collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset = DatasetCatalog.get(dataset_name)
        sampler = InferenceSampler(len(dataset))
        batch_size = get_batch_size(cfg.SOLVER.IMS_PER_BATCH)
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            # collate_fn=trivial_batch_collator,
        )
