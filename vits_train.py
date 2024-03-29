# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
import torch

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

from model.vits import VitsModel


@hydra_runner(config_path="conf", config_name="ref_vits")
def main(cfg):
    torch.multiprocessing.set_sharing_strategy('file_system')
    trainer = pl.Trainer(replace_sampler_ddp=False, **cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = VitsModel(cfg=cfg.model, trainer=trainer)

    trainer.callbacks.extend([pl.callbacks.LearningRateMonitor(), LogEpochTimeCallback()])
    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter