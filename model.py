"""
This file is a modified version of model.py of an ealier version of Open Unmix
https://github.com/sigsep/open-unmix-pytorch
"""

from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def smax(tensor, dim, gamma, keepdim=False):
    exp_gamma = torch.exp(tensor * gamma)
    sum_over_dim = torch.sum(exp_gamma, dim=dim, keepdim=keepdim)
    result = torch.log(sum_over_dim) / gamma
    return result

class MatrixDiagonalIndexIterator:
    '''
    Custom iterator class to return successive diagonal indices of a matrix
    '''

    def __init__(self, m, n, k_start=0, bandwidth=None):
        '''
        __init__(self, m, n, k_start=0, bandwidth=None):

        Arguments:
            m (int)         : number of rows in matrix
            n (int)         : number of columns in matrix
            k_start (int)   : (k_start, k_start) index to begin from
            bandwidth (int) : bandwidth to constrain indices within
        '''
        self.m         = m
        self.n         = n
        self.k         = k_start
        self.k_max     = self.m + self.n - k_start - 1
        self.bandwidth = bandwidth

    def __iter__(self):
        return self

    def __next__(self):
        if hasattr(self, 'i') and hasattr(self, 'j'):

            if self.k == self.k_max:
                raise StopIteration

            elif self.k < self.m and self.k < self.n:
                self.i = self.i + [self.k]
                self.j = [self.k] + self.j
                self.k+=1

            elif self.k >= self.m and self.k < self.n:
                self.j.pop(-1)
                self.j = [self.k] + self.j
                self.k+=1

            elif self.k < self.m and self.k >= self.n:
                self.i.pop(0)
                self.i = self.i + [self.k]
                self.k+=1

            elif self.k >= self.m and self.k >= self.n:
                self.i.pop(0)
                self.j.pop(-1)
                self.k+=1

        else:
            self.i = [self.k]
            self.j = [self.k]
            self.k+=1

        if self.bandwidth:
            i_scb, j_scb = sakoe_chiba_band(self.i.copy(), self.j.copy(), self.m, self.n, bandwidth)
            return i_scb, j_scb
        else:
            return self.i.copy(), self.j.copy()

def sakoe_chiba_band(i_list, j_list, m, n, bandwidth=1):
    i_scb, j_scb = zip(*[(i, j) for i,j in zip(i_list, j_list)
                         if abs(2*(i*(n-1) - j*(m-1))) < max(m, n)*(bandwidth+1)])
    return list(i_scb), list(j_scb)


def dtw_matrix(scores, mode='faster', idx_to_skip=None):
    """
    Computes the accumulated score matrix by the "DTW forward operation"

    Args:
        scores (torch.Tensor): Score matrix of shape (batch_size, seq_len1, seq_len2)
        mode (str): Mode of operation, currently supports 'faster'
        idx_to_skip (list[int], optional): Unused in current mode

    Returns:
        torch.Tensor: Accumulated DTW matrix of shape (batch_size, seq_len1, seq_len2)
    """
    B, N, M = scores.shape
    device = scores.device

    if mode == 'faster':
        dtw_matrix = torch.full((B, N + 1, M + 1), fill_value=-1e5, device=device)
        dtw_matrix[:, 0, 0] = 2e5

        for (m, n), (m_m1, n_m1) in zip(
            MatrixDiagonalIndexIterator(m=M + 1, n=N + 1, k_start=1),
            MatrixDiagonalIndexIterator(m=M, n=N, k_start=0)
        ):
            d1 = dtw_matrix[:, n_m1, m].unsqueeze(-1)
            d2 = dtw_matrix[:, n_m1, m_m1].unsqueeze(-1)
            max_vals = torch.maximum(d1, d2).squeeze(-1)
            dtw_matrix[:, n, m] = scores[:, n_m1, m_m1] + max_vals

        return dtw_matrix[:, 1:N+1, 1:M+1]


def optimal_alignment_path(matrix):
    # matrix is torch.tensor with size (1, sequence_length1, sequence_length2)

    # Forward step DTW
    accumulated_scores = dtw_matrix(matrix, mode='faster')
    accumulated_scores = accumulated_scores.cpu().detach().squeeze(0).numpy()

    N, M = accumulated_scores.shape

    # Optimal path matrix initialization
    optimal_path_matrix = np.zeros((N, M), dtype=int)
    optimal_path_matrix[-1, -1] = 1  # last phoneme is active at last time frame

    # Backtracking: go backwards through time steps
    n, m = N - 2, M - 1

    while m > 0:
        # Fetch the previous scores
        d1, d2 = accumulated_scores[n, m], accumulated_scores[n, m - 1]

        # Find the path with the highest score
        arg_max = np.argmax([d1, d2])  # 0 if same phoneme active, 1 if previous phoneme active

        # Mark the active phoneme in the path
        optimal_path_matrix[n, m - arg_max] = 1

        # Update indices for backtracking
        n -= 1
        m -= arg_max

        # Error handling: Ensure n doesn't go out of bounds
        if n == -2:
            print(f"DTW backward pass failed. n={n}, m={m}")
            break

    # Ensure that the first column of the optimal path is fully marked
    optimal_path_matrix[:n + 1, 0] = 1

    return optimal_path_matrix  # numpy array with shape (N, M)


def pad_for_stft(signal, hop_length):
    # this function pads the given signal so that all samples are taken into account by the stft
    # input and output signal have shape (batch_size, nb_channels, nb_timesteps)

    nb_samples, nb_channels, signal_len = signal.size()
    incomplete_frame_len = signal_len % hop_length

    device = signal.device

    if incomplete_frame_len == 0:
        # no padding needed
        return signal
    else:
        pad_length = hop_length - incomplete_frame_len
        padding = torch.zeros((nb_samples, nb_channels, pad_length)).to(device)
        padded_signal = torch.cat((signal, padding), dim=2)
        return padded_signal


class STFT(nn.Module):
    def __init__(self, n_fft=4096, n_hop=1024, center=False):
        super(STFT, self).__init__()

        # Window initialization: no gradient needed
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output: (nb_samples, nb_channels, nb_bins, nb_frames)
        """
        nb_samples, nb_channels, nb_timesteps = x.shape

        # Merge nb_samples and nb_channels for multichannel stft computation
        x = x.view(nb_samples * nb_channels, nb_timesteps)

        # Compute the STFT (torch.stft is optimized for batch operations)
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            pad_mode='reflect',
            return_complex=True
        )  # shape: (nb_samples * nb_channels, nb_bins, nb_frames)

        # Reshape: (nb_samples, nb_channels, nb_bins, nb_frames)
        nb_bins, nb_frames = stft_f.shape[1], stft_f.shape[2]
        stft_f = stft_f.view(nb_samples, nb_channels, nb_bins, nb_frames)

        return stft_f

class Spectrogram(nn.Module):
    def __init__(
        self,
        power=1,
        mono=True
    ):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_channels, nb_bins, nb_frames)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        stft_f = stft_f.transpose(2, 3)

        # take the magnitude
        stft_f = stft_f.abs().pow(self.power)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        # permute output for LSTM convenience
        return stft_f.permute(2, 0, 1, 3)  # (nb_frames, nb_samples, nb_channels, nb_bins)


def index2one_hot(index_tensor, vocabulary_size):
    """
    Transforms index representation to one hot representation
    :param index_tensor: shape: (batch_size, sequence_length, 1) tensor containing character indices
    :param vocabulary_size: scalar, size of the vocabulary
    :return: chars_one_hot: shape: (batch_size, sequence_length, vocabulary_size)
    """

    device = index_tensor.device
    index_tensor = index_tensor.type(torch.LongTensor).to(device)

    batch_size = index_tensor.size()[0]
    char_sequence_len = index_tensor.size()[1]
    chars_one_hot = torch.zeros((batch_size, char_sequence_len, vocabulary_size), device=device)
    chars_one_hot.scatter_(dim=2, index=index_tensor, value=1)

    return chars_one_hot



class InformedOpenUnmix3(nn.Module):
    """
    Open Unmix with an additional text encoder and attention mechanism
    """
    def __init__(
        self,
        n_fft=512,
        n_hop=256,
        input_is_spectrogram=False,
        hidden_size=512,
        nb_channels=1,
        sample_rate=16000,
        audio_encoder_layers=2,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=257,
        unidirectional=False,
        power=1,
        vocab_size=44,
        audio_transform='STFT'
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        super(InformedOpenUnmix3, self).__init__()

        self.return_alphas = False
        self.optimal_path_alphas = False

        # Text processing
        self.vocab_size = vocab_size
        self.lstm_txt = LSTM(vocab_size, hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True)

        # Attention
        w_s_init = torch.empty(hidden_size, hidden_size)
        nn.init.uniform_(w_s_init, -torch.sqrt(torch.tensor(1., dtype=torch.float32) / hidden_size),
                         torch.sqrt(torch.tensor(1., dtype=torch.float32) / hidden_size))
        self.w_s = nn.Parameter(w_s_init)

        # Connection
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)
        self.bn_c = nn.BatchNorm1d(hidden_size)

        self.nb_output_bins = n_fft // 2 + 1
        self.nb_bins = max_bin if max_bin else self.nb_output_bins
        self.hidden_size = hidden_size

        # Audio transform and STFT
        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        self.transform = nn.Sequential(self.stft, self.spec) if not input_is_spectrogram else nn.Identity()

        # Audio encoder
        self.fc1 = nn.Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        lstm_hidden_size = hidden_size if unidirectional else hidden_size // 2
        self.audio_encoder_lstm = LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size,
                                       num_layers=audio_encoder_layers, bidirectional=not unidirectional,
                                       batch_first=False, dropout=0.4)

        # LSTM layers
        self.lstm = LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, num_layers=nb_layers,
                         bidirectional=not unidirectional, batch_first=False, dropout=0.4)

        self.fc2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.fc3 = nn.Linear(hidden_size, self.nb_output_bins * nb_channels, bias=False)
        self.bn3 = nn.BatchNorm1d(self.nb_output_bins * nb_channels)

        # Input mean/scale handling
        self.input_mean = nn.Parameter(torch.tensor(input_mean[:self.nb_bins] if input_mean is not None else [0.] * self.nb_bins).float())
        self.input_scale = nn.Parameter(torch.tensor(1.0 / input_scale[:self.nb_bins] if input_scale is not None else [1.] * self.nb_bins).float())
        self.output_scale = nn.Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = nn.Parameter(torch.ones(self.nb_output_bins).float())

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            input_mean=config.get('scaler_mean', None),
            input_scale=config.get('scaler_std', None),
            nb_channels=config['nb_channels'],
            hidden_size=config['hidden_size'],
            n_fft=config['nfft'],
            n_hop=config['nhop'],
            max_bin=config['max_bin'],
            sample_rate=config['samplerate'],
            vocab_size=config['vocabulary_size'],
            audio_encoder_layers=config['nb_audio_encoder_layers'],
            attention=config.get('attention', 'general')
        )

    def forward(self, x):
        text_idx = x[1].unsqueeze(dim=2)
        x = x[0]  # mix

        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.shape

        # Text processing
        text_onehot = index2one_hot(text_idx, self.vocab_size)
        h, _ = self.lstm_txt(text_onehot)

        # Audio processing
        mix = x.detach().clone()
        x = x[..., :self.nb_bins]  # Crop to max bin
        x += self.input_mean
        x *= self.input_scale

        # Encode audio features
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        x = self.bn1(x)
        x = torch.tanh(x.reshape(nb_frames, nb_samples, self.hidden_size))

        x, _ = self.audio_encoder_lstm(x)

        # Attention mechanism
        batch_size = h.size(0)
        x = x.transpose(0, 1)  # Shape (nb_samples, nb_frames, hidden_size)
        side_info_transformed = torch.bmm(self.w_s.expand(batch_size, -1, -1), h.transpose(1, 2))
        scores = torch.bmm(x, side_info_transformed)
        dtw_alphas = dtw_matrix(scores, mode='faster')
        alphas = F.softmax(dtw_alphas, dim=2)

        # Compute context vectors
        context = torch.bmm(h.transpose(1, 2), alphas.transpose(1, 2))
        context = context.transpose(1, 2)

        # Connection of audio and text
        concat = torch.cat((context, x), dim=2)
        x = self.fc_c(concat)
        x = self.bn_c(x.transpose(1, 2))
        x = torch.tanh(x).transpose(1, 2).transpose(0, 1)

        # Apply stacked LSTMs
        lstm_out = self.lstm(x)
        x = torch.cat([x, lstm_out[0]], -1)

        # Dense layers with batch normalization
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)

        # Second dense stage
        x = self.fc3(x)
        x = self.bn3(x)

        # Reshape back to original dimensions
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # Output scaling
        x *= self.output_scale
        x += self.output_mean

        # Apply ReLU and mix the result
        return F.relu(x) * mix, alphas, scores