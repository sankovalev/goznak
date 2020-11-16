import os
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
import numpy as np
import pandas as pd
from torch import optim, nn, Tensor, stack, load, no_grad
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


class AudioDataset(Dataset):
    """
    Датасет для подгрузки аудиофайлов, один фолд.
    Идея следующая: генерируем пары noisy-clear, ЛИБО
                    clear-clear, чтобы сетка делала тождественное преобразование,
                    а классификационная голова научилась предсказывать метку класса
    """

    def __init__(self, meta_file, data_path, transforms, valid=False, prob=0.3, train_ratio=0.8):
        self.df = self._get_df(meta_file, valid=valid, train_ratio=train_ratio)
        self.filenames = self.df.path.unique()
        self.data_path = data_path
        self.transforms = transforms
        self.valid = valid
        self.prob = prob

    def _get_df(self, file_path, valid, train_ratio):
        np.random.seed(97)
        df = pd.read_csv(file_path)
        unique_ids = list(df.id.unique())
        # делим на трейн и валидацию (путаница в терминах, val в задании - отложенная выборка по сути)
        # не делаем пересечений внутри одного id, все записи целиком попадают либо в трейн, либо в валидацию
        train_ids = np.random.choice(unique_ids, int(train_ratio * len(unique_ids)), False)
        if valid:
            return df.loc[~df.id.isin(train_ids)]
        else:
            return df.loc[df.id.isin(train_ids)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        rows_df = self.df.loc[self.df.path == filename]

        if np.random.rand() < self.prob:
            # случай, когда одинаковые изображения и удалять шум не надо
            label = 0.0
            noisy_row = rows_df.loc[rows_df.type == 'clean'].iloc[0]
            clean_row = rows_df.loc[rows_df.type == 'clean'].iloc[0]
        else:
            label = 1.0
            noisy_row = rows_df.loc[rows_df.type == 'noisy'].iloc[0]
            clean_row = rows_df.loc[rows_df.type == 'clean'].iloc[0]

        noisy_path = os.path.join(self.data_path,
                                  noisy_row.get('folder'),
                                  noisy_row.get('type'),
                                  str(noisy_row.get('id')),
                                  filename)
        clean_path = os.path.join(self.data_path,
                                  clean_row.get('folder'),
                                  clean_row.get('type'),
                                  str(clean_row.get('id')),
                                  filename)

        noisy = np.expand_dims(np.load(noisy_path).astype(float), 2)
        clean = np.expand_dims(np.load(clean_path).astype(float), 2)

        augmented = self.transforms(image=noisy,
                                    mask=clean)

        return {
            "noisy": augmented['image'][0].unsqueeze(0).float(), # 1xHxW
            "clean": augmented['mask'][:, :, 0].unsqueeze(0).float(), #1xHxW
            "label": Tensor([label]).float()
        }


class BaselineLearner(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        aux_params=dict(
            pooling='avg',
            dropout=0.2,
            activation='sigmoid',
            classes=1)
        self.net = smp.Unet(encoder_name=self.config.training.model_encoder,
                            in_channels=1,
                            aux_params=aux_params) # классификационная голова
        self.losses = {
            "regression": nn.MSELoss(),
            "classification": nn.BCEWithLogitsLoss()
        }
        augs_list = [
            albu.PadIfNeeded(480, 80),
            albu.RandomCrop(480, 80),
            albu.Resize(576, 96), # должно быть кратно 32
            ToTensorV2()
            # без нормализации
            ]
        self.transforms = albu.Compose(augs_list)
        self.trainset = AudioDataset(config.sources.train_meta,
                                          config.sources.data_path,
                                          self.transforms,
                                          False)
        self.validset = AudioDataset(config.sources.train_meta,
                                          config.sources.data_path,
                                          self.transforms,
                                          True)

    def forward(self, x):
        return self.net(x)

    def _calc_losses(self, batch):
        noisy, gt_clean, gt_label = batch["noisy"], batch["clean"], batch["label"]
        pr_clean, pr_label = self(noisy)
        loss_reg = self.losses['regression'](pr_clean, gt_clean)
        loss_cls = self.losses['classification'](pr_label, gt_label)
        return loss_reg, loss_cls, 0.3 * loss_reg + 0.7 * loss_cls

    def training_step(self, batch, batch_nb):
        loss_reg, loss_cls, loss_value = self._calc_losses(batch)
        self.log('train_loss_reg', loss_reg, prog_bar=False)
        self.log('train_loss_cls', loss_cls, prog_bar=False)
        self.log('train_loss', loss_value, prog_bar=True)
        return loss_value

    def validation_step(self, batch, batch_idx):
        loss_reg, loss_cls, loss_value = self._calc_losses(batch)
        self.log('valid_loss_reg', loss_reg, prog_bar=False)
        self.log('valid_loss_cls', loss_cls, prog_bar=False)
        self.log('valid_loss', loss_value, prog_bar=False)
        return loss_value

    def validation_epoch_end(self, outputs):
        loss_value = stack([x for x in outputs]).mean()
        self.log('valid_loss', loss_value, prog_bar=True)

    def configure_optimizers(self):
        opt = optim.Adam(self.net.parameters(), lr=self.config.training.learning_rate)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.config.training.batch_size,
                          num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.config.training.batch_size,
                          num_workers=8, shuffle=False)


class BaselinePredictor(nn.Module):
    """
    Обертка для предсказаний и подсчета метрик на отложенной выборке (val папка)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        aux_params=dict(
            pooling='avg',
            dropout=0.2,
            activation='sigmoid',
            classes=1)
        self.net = smp.Unet(encoder_name=self.config.training.model_encoder,
                            in_channels=1,
                            aux_params=aux_params) # классификационная голова

        self.metrics = {
            "regression": pl.metrics.MeanSquaredError(),
            "classification": pl.metrics.Accuracy(self.config.prediction.threshold)
        }

        # грузим лучшие веса
        state_dict = self._rename_layers(load(os.path.join(config.sources.ckpt_path,
                                                           f"{config.name}.ckpt"))['state_dict'])
        self.net.load_state_dict(state_dict)
        self.net.eval()

        # TODO: вообще тут надо делать нарезку изображений на тайлы, а потом ресайз, но нет
        augs_list = [
            albu.PadIfNeeded(480, 80),
            albu.RandomCrop(480, 80),
            albu.Resize(576, 96), # должно быть кратно 32
            ToTensorV2()
            # без нормализации
            ]
        self.transforms = albu.Compose(augs_list)
        self.deferset = AudioDataset(config.sources.val_meta,
                                     config.sources.data_path,
                                     self.transforms,
                                     False,
                                     prob=0.5,
                                     train_ratio=1.0)
        self.deferloader = DataLoader(self.deferset, batch_size=1, shuffle=False)

    @staticmethod
    def _rename_layers(state_dict):
        new_state_dict = dict()
        for key, value in state_dict.items():
            new_state_dict[key.replace('net.', '')] = value
        return new_state_dict

    def calculate_metrics(self):
        classification_metric = list()
        regression_metric = list()

        with no_grad():
            for batch in tqdm(self.deferloader):
                noisy, gt_clean, gt_label = batch["noisy"], batch["clean"], batch["label"]
                pr_clean, pr_label = self.net(noisy)
                is_input_clean = bool(pr_label < self.config.prediction.threshold)

                if not self.config.prediction.independent_heads and is_input_clean:
                    # при зависимых выходах вообще не делаем регрессию, если классификатор говорит, что вход чистый
                    pr_clean = gt_clean

                regression_metric.append(self.metrics['regression'](pr_clean, gt_clean).detach().numpy())
                classification_metric.append(self.metrics['classification'](pr_label, gt_label).detach().numpy())

        return {
            "classification": classification_metric,
            "regression": regression_metric
        }


