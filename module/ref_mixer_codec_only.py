from typing import List
import torch
import torch.nn as nn
from module.codec_embedding import AudioCodecForEmbedding

from nemo.collections.tts.modules.mixer_tts import MixerTTSBlock, \
    create_time_mix_layer, create_channel_mix_layer,Mix

class RefMixer(nn.Module):
    def __init__(
            self,
            initial_dim,
            gru_out,
            expansion_factor,
            conv_type,
            num_layers,
            kernel_sizes,
            dropout=0.1,
    ):
        super().__init__()
        if num_layers != len(kernel_sizes):
            raise ValueError
        self.feature_dim = initial_dim

        self.mixer_blocks = nn.Sequential(
            *[
                MixerTTSBlock(initial_dim, expansion_factor, kernel_size, conv_type, dropout)
                for kernel_size in kernel_sizes
            ],
        )
        self.norm = nn.LayerNorm(initial_dim, eps=1e-03)

        self.gru = nn.GRU(initial_dim, gru_out, 1, batch_first=True)


    def _get_mask_from_length(self, lengths):
        mask = (
                torch.arange(lengths.max()).to(lengths.device).expand(lengths.shape[0],
                                                                      lengths.max()) < lengths.unsqueeze(
            1)).unsqueeze(2)
        return mask

    def forward(self, codec_embedding, masks):
        # x = multi_embeds * multi_masks
        x = codec_embedding.transpose(1, 2).float()
        for block in self.mixer_blocks:
            x, lens = block(x, masks)
        # x += codec_embedding.transpose(1, 2).float()
        y = self.norm(x)

        self.gru.flatten_parameters()
        _, y = self.gru(y)   # whole output , last hidden state (1, batch, featue dim)
        y = y.transpose(0,1)  # (batch, 1, feature dim)

        return y

if __name__=="__main__":
    c1 = torch.load("../encodec_sample.pt")
    # c1 = torch.sum(c1, dim=0)
    c2 = torch.load("../encodec_sample2.pt")
    # c2 = torch.sum(c2, dim=0)
    codec_length = [c1.size(1), c2.size(1)]
    max_length = max(c1.size(1), c2.size(1))

    c1 = torch.cat([c1, torch.zeros(8, max_length - c1.size(1))], dim=1)
    c2 = torch.cat([c2, torch.zeros(8, max_length - c2.size(1))], dim=1)
    codec_codes = torch.stack([c1.long(),c2.long()])

    codec_length = torch.as_tensor(codec_length)

    ref_enc = RefMixer(
        initial_dim=8,
        gru_out=384,
        expansion_factor=4,
        num_layers=6,
        kernel_sizes=[11, 13, 15, 17, 19, 21],
        conv_type="depth-wise",
    )
    # output = ref_enc(audio_embed, lm_embed, audio_length, lm_length)
    ref_enc = ref_enc.to('cuda')
    output = ref_enc(codec_codes.to('cuda'), codec_length.to('cuda'))
