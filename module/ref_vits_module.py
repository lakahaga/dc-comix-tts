import math

import torch
import torch.nn as nn

from module.monotonic_align import maximum_path
from model.helper import rand_slice_segments, generate_path, get_mask_from_lengths
from module.vits_modules import TextEncoder, PosteriorEncoder, DurationPredictor, ResidualCouplingBlock, Generator, StochasticDurationPredictor

# ref encoders
from module.ref_gst import GlobalStyleTokenForCodec, GlobalStyleTokenForMulti, GlobalStyleToken
from module.ref_mixer_v2 import RefMixer as MultiMixer
from module.ref_mixer_codec_rnn import RefMixer as MultiCodecMixer
from module.ref_mixer_mel import RefMixer as MelMixer
from module.ref_mixer_codec_only import RefMixer as CodecMixer

class RefSynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        padding_idx,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        ref_encoder=None,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        **kwargs
    ):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.padding_idx = padding_idx
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            padding_idx,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels
        )
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

        self.ref_encoder = ref_encoder

    def forward(self, text, text_len, spec, spec_len, speakers=None,
                ref_spec=None, ref_spec_lens=None, ref_codec=None, ref_codec_lens=None, lm_embedding=None, lm_embedding_lens=None):
        x, mean_prior, logscale_prior, text_mask = self.enc_p(text, text_len)
        if self.n_speakers > 1:
            g = self.emb_g(speakers).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        if self.ref_encoder is not None:
            style_embedding = None
            if isinstance(self.ref_encoder, GlobalStyleToken) or isinstance(self.ref_encoder,
                                                                            GlobalStyleTokenForCodec) or isinstance(
                    self.ref_encoder, MelMixer) or isinstance(self.ref_encoder, CodecMixer):
                # mel or codec only
                if ref_spec is not None and ref_spec_lens is not None:
                    ref_spec_mask = (
                                torch.arange(ref_spec_lens.max()).to(ref_spec.device).expand(ref_spec_lens.shape[0],
                                                                                             ref_spec_lens.max()) < ref_spec_lens.unsqueeze(
                            1)).unsqueeze(2)
                    style_embedding = self.ref_encoder(ref_spec, ref_spec_mask)
                else:
                    raise ValueError("GST needs reference spectrogram")
            if isinstance(self.ref_encoder, GlobalStyleTokenForMulti) or isinstance(self.ref_encoder,
                                                                                    MultiMixer) or isinstance(
                    self.ref_encoder, MultiCodecMixer):
                # codec + lm
                if ref_codec is not None and ref_codec_lens is not None and lm_embedding is not None and lm_embedding_lens is not None:
                    style_embedding = self.ref_encoder(ref_codec, lm_embedding, ref_codec_lens, lm_embedding_lens)
                else:
                    raise ValueError(
                        "Multi Reference encoder needs codec, codec length, lm embedding, lm embedding lens")
            g += style_embedding.transpose(1, 2)  # g is [b,h,1] ans style embedding is [b,1,h] => needs transpose

        z, mean_posterior, logscale_posterior, spec_mask = self.enc_q(spec, spec_len, g=g)
        z_p = self.flow(z, spec_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logscale_prior)  # [b, d, t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logscale_prior, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (mean_prior * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (mean_prior ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(text_mask, 2) * torch.unsqueeze(spec_mask, -1)
            attn = maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, text_mask, w, g=g)
            l_length = l_length / torch.sum(text_mask)
        else:
            logw_ = torch.log(w + 1e-6) * text_mask
            logw = self.dp(x, text_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(text_mask)  # for averaging

        # expand prior
        mean_prior = torch.matmul(attn.squeeze(1), mean_prior.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logscale_prior = torch.matmul(attn.squeeze(1), logscale_prior.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_slice, ids_slice = rand_slice_segments(z, spec_len, self.segment_size)
        audio = self.dec(z_slice, g=g)
        return (
            audio,
            l_length,
            attn,
            ids_slice,
            text_mask,
            spec_mask,
            (z, z_p, mean_prior, logscale_prior, mean_posterior, logscale_posterior),
        )

    def infer(self, text, text_len, speakers=None, noise_scale=1, length_scale=1, noise_scale_w=1.0, max_len=None,
              ref_spec=None, ref_spec_lens=None, ref_codec=None, ref_codec_lens=None, lm_embedding=None, lm_embedding_lens=None):
        x, mean_prior, logscale_prior, text_mask = self.enc_p(text, text_len)
        if self.n_speakers > 1 and speakers is not None:
            g = self.emb_g(speakers).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        if self.ref_encoder is not None:
            style_embedding = None
            if isinstance(self.ref_encoder, GlobalStyleToken) or isinstance(self.ref_encoder,
                                                                            GlobalStyleTokenForCodec) or isinstance(
                self.ref_encoder, MelMixer) or isinstance(self.ref_encoder, CodecMixer):
                # mel or codec only
                if ref_spec is not None and ref_spec_lens is not None:
                    ref_spec_mask = (
                            torch.arange(ref_spec_lens.max()).to(ref_spec.device).expand(ref_spec_lens.shape[0],
                                                                                         ref_spec_lens.max()) < ref_spec_lens.unsqueeze(
                        1)).unsqueeze(2)
                    style_embedding = self.ref_encoder(ref_spec, ref_spec_mask)
                else:
                    raise ValueError("GST needs reference spectrogram")
            if isinstance(self.ref_encoder, GlobalStyleTokenForMulti) or isinstance(self.ref_encoder,
                                                                                    MultiMixer) or isinstance(
                self.ref_encoder, MultiCodecMixer):
                # codec + lm
                if ref_codec is not None and ref_codec_lens is not None and lm_embedding is not None and lm_embedding_lens is not None:
                    style_embedding = self.ref_encoder(ref_codec, lm_embedding, ref_codec_lens, lm_embedding_lens)
                else:
                    raise ValueError(
                        "Multi Reference encoder needs codec, codec length, lm embedding, lm embedding lens")
            g += style_embedding.transpose(1, 2)  # g is [b,h,1] ans style embedding is [b,1,h] => needs transpose

        if self.use_sdp:
            logw = self.dp(x, text_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, text_mask, g=g)
        w = torch.exp(logw) * text_mask * length_scale
        w_ceil = torch.ceil(w)
        audio_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        audio_mask = torch.unsqueeze(get_mask_from_lengths(audio_lengths, None), 1).to(text_mask.dtype)
        attn_mask = torch.unsqueeze(text_mask, 2) * torch.unsqueeze(audio_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        mean_prior = torch.matmul(attn.squeeze(1), mean_prior.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logscale_prior = torch.matmul(attn.squeeze(1), logscale_prior.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = mean_prior + torch.randn_like(mean_prior) * torch.exp(logscale_prior) * noise_scale
        z = self.flow(z_p, audio_mask, g=g, reverse=True)
        audio = self.dec((z * audio_mask)[:, :, :max_len], g=g)
        return audio, attn, audio_mask, (z, z_p, mean_prior, logscale_prior)

    # Can be used for emotions
    def voice_conversion(self, y, y_lengths, speaker_src, speaker_tgt):
        assert self.n_speakers > 1, "n_speakers have to be larger than 1."
        g_src = self.emb_g(speaker_src).unsqueeze(-1)
        g_tgt = self.emb_g(speaker_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
