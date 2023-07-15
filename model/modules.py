import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad
from transformer.Models import get_sinusoid_encoding_table
from transformer.Layers import ConvNorm, FFTBlock
from model.blocks import LinearNorm, FiLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    def get_pitch_embedding(self, x, target, mask, control, gammas=None, betas=None):
        prediction = self.pitch_predictor(x, mask, gammas, betas)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control, gammas=None, betas=None):
        prediction = self.energy_predictor(x, mask, gammas, betas)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        gammas=None,
        betas=None
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask, gammas, betas)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, e_control
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, e_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )
        self.film = FiLM()
        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask, gammas=None, betas=None):
        out = self.conv_layer(encoder_output)
        if (gammas is not None ) and (betas is not None):
                out = self.film(out, gammas, betas)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
    
class ReferenceEncoder(nn.Module):
    """ Reference Encoder """

    def __init__(self, preprocess_config, model_config):
        super(ReferenceEncoder, self).__init__()

        self.max_seq_len = model_config["max_seq_len"] + 1
        n_conv_layers = model_config["reference_encoder"]["conv_layer"]
        kernel_size = model_config["reference_encoder"]["conv_kernel_size"]
        n_layers = model_config["reference_encoder"]["encoder_layer"]
        n_head = model_config["reference_encoder"]["encoder_head"]
        self.d_model = model_config["reference_encoder"]["encoder_hidden"]
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        d_k = d_v = (self.d_model // n_head)
        self.filter_size = model_config["reference_encoder"]["conv_filter_size"]
        dropout = model_config["reference_encoder"]["dropout"]

        self.layer_stack = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNorm(
                            n_mel_channels if i == 0 else self.filter_size,
                            self.filter_size,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=(kernel_size - 1) // 2,
                            dilation=1,
                            transform=True,
                        ),
                    nn.ReLU(),
                    nn.LayerNorm(self.filter_size),
                    nn.Dropout(dropout),
                )
                for i in range(n_conv_layers)
            ]
        )

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(self.max_seq_len, self.filter_size).unsqueeze(0),
            requires_grad=False,
        )

        self.fftb_linear = LinearNorm(self.filter_size, self.d_model)
        self.fftb_stack = nn.ModuleList(
            [
                FFTBlock(
                    self.d_model, n_head, d_k, d_v, self.filter_size, [kernel_size, kernel_size], dropout=dropout, film=False
                )
                for _ in range(n_layers)
            ]
        )

        self.feature_wise_affine = LinearNorm(self.d_model, 2 * self.d_model)

    def forward(self, mel, max_len, mask):

        batch_size = mel.shape[0]

        # -- Prepare Input
        enc_seq = mel
        for enc_layer in self.layer_stack:
            enc_seq = enc_layer(enc_seq)

        if mask is not None:
            enc_seq = enc_seq.masked_fill(mask.unsqueeze(-1), 0.0)

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            enc_seq = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.filter_size
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            enc_seq = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        enc_seq = self.fftb_linear(enc_seq)
        for fftb_layer in self.fftb_stack:
            enc_seq, _ = fftb_layer(
                enc_seq, mask=mask, slf_attn_mask=slf_attn_mask
            )

        # -- Avg Pooling
        prosody_vector = enc_seq.mean(dim=1, keepdim=True) # [B, 1, H]


        # -- Feature-wise Affine
        gammas, betas = torch.split(
            self.feature_wise_affine(prosody_vector), self.d_model, dim=-1
        )

        return gammas, betas
