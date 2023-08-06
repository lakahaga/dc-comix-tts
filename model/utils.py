from nemo.collections.tts.torch.tts_data_types import MAIN_DATA_TYPES, WithLens
from torchdata.data_type import DATA_STR2DATA_CLASS
def average_features(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = torch.nn.functional.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = torch.nn.functional.pad(torch.cumsum(pitch, dim=2), (1, 0))
    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce) - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems)
    return pitch_avg

def extensive_process_batch(batch_data, sup_data_types_set):
    batch_dict = {}
    batch_index = 0

    for name, datatype in DATA_STR2DATA_CLASS.items():
        if datatype in MAIN_DATA_TYPES or datatype in sup_data_types_set:
            batch_dict[name] = batch_data[batch_index]
            batch_index = batch_index + 1
            if issubclass(datatype, WithLens):
                batch_dict[name + "_lens"] = batch_data[batch_index]
                batch_index = batch_index + 1
    return batch_dict

def process_batch(batch_data, sup_data_types_set):
    batch_dict = {}
    batch_index = 0

    for name, datatype in DATA_STR2DATA_CLASS.items():
        if datatype in MAIN_DATA_TYPES or datatype in sup_data_types_set:
            batch_dict[name] = batch_data[batch_index]
            batch_index = batch_index + 1
            if issubclass(datatype, WithLens):
                batch_dict[name + "_lens"] = batch_data[batch_index]
                batch_index = batch_index + 1

    if len(batch_data)==12 or len(batch_data)==11 or (len(batch_data)==13 and 'energy' in batch_dict.keys()):     # audio, lm embedding, without xvector or with speaker id
        batch_dict['audio_embedding'] = batch_data[-4]
        batch_dict['audio_embedding_lens'] = batch_data[-3]
        batch_dict['lm_embedding'] = batch_data[-2]
        batch_dict['lm_embedding_lens'] = batch_data[-1]
    else:   # audio, lm embedding, with xvector
        batch_dict['audio_embedding'] = batch_data[-5]
        batch_dict['audio_embedding_lens'] = batch_data[-4]
        batch_dict['lm_embedding'] = batch_data[-3]
        batch_dict['lm_embedding_lens'] = batch_data[-2]
        batch_dict['pretrained_spk_embedding'] = batch_data[-1]

    return batch_dict

import torch
import torch.nn.functional as F

from nemo.collections.tts.modules.transformer import mask_from_lens
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import (
    LengthsType,
    LossType,
    MelSpectrogramType,
    RegressionValuesType,
    TokenDurationType,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType

class EnergyLoss(Loss):
    def __init__(self, loss_scale=0.1):
        super().__init__()
        self.loss_scale = loss_scale

    @property
    def input_types(self):
        return {
            "energy_predicted": NeuralType(('B', 'T'), RegressionValuesType()),
            "energy_tgt": NeuralType(('B', 'T'), RegressionValuesType()),
            "length": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, energy_predicted, energy_tgt, length):
        if energy_tgt is None:
            return 0.0
        dur_mask = mask_from_lens(length, max_len=energy_tgt.size(1))
        energy_loss = F.mse_loss(energy_tgt, energy_predicted, reduction='none')
        energy_loss = (energy_loss * dur_mask).sum() / dur_mask.sum()
        energy_loss *= self.loss_scale

        return energy_loss

if __name__=="__main__":
    pitch = torch.load("../test_pitch_1.pt")
    durs = torch.load("../test_durs_1.pt")
    average_features(pitch,durs)