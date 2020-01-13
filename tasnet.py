import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class TasNet:
    def __init__(self, L, N):
        """
        :param L: One frame length
        :param N: Number of vectors for convolution
        """
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(L, N, kernel_size=1, stride=1, bias=False)
        self.conv1d_V = nn.Conv1d(L, N, kernel_size=1, stride=1, bias=False)

    def encoder(self, speech_mixture):
        """
        TasNet Encoder
        It takes mixture audio vectors as input
        Calculate mixture weight matrix by following steps;
            1. L2 normalization to input mixture signal
            2. 1D gated convolution along time dimension
            3. Apply non-linearity (3.1. sigmoid, 3.2. ReLU)
            4. Multiply 3.1 and 3.2

        Args:
            :param speech_mixture: Input mixture signals [B, K, L]
            :return mixture_weight: Non-negative mixture weight matrix
            :return norm_coefficient: Normalize coefficient

        B: Number of items (number of audio files)
        K: Number of frames from one audio
        L: One frame length
        """
        # Get input batch data size
        B, K, L = speech_mixture.size()

        # L2 Norm along L axis
        norm_coefficient = torch.norm(speech_mixture, p=2, dim=2, keepdim=True)  # B x K x 1
        norm_mixture = speech_mixture / (norm_coefficient + EPS)  # B x K x L

        # 1-D gated convolution (along time dimension)
        norm_mixture = torch.unsqueeze(norm_mixture.view(-1, L), 2)  # B*K x L x 1

        # Apply non-linearity
        conv = F.relu(self.conv1d_U(norm_mixture))  # B*K x N x 1
        gate = torch.sigmoid(self.conv1d_V(norm_mixture))  # B*K x N x 1

        # Multiply output matrices from sigmoid and ReLU
        mixture_weight = conv * gate  # B*K x N x 1
        mixture_weight = mixture_weight.view(B, K, self.N)  # B x K x N
        return mixture_weight, norm_coefficient


    def separator(self, mixture_weight):
        """
        TasNet Separator
        :param mixture_weight [B x K x N]
        :return
        """

    def decoder(self):
        pass



