import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math
import os
from utils import config
import pdb


import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import numpy as np


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                       hidden_size, num_heads, bias_mask, attention_dropout)

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding='both',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)
        # self.layer_norm_end = LayerNorm(hidden_size)

    def forward(self, inputs, mask=None):
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        # y = self.layer_norm_end(y)
        return y


class GraphLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(GraphLayer, self).__init__()

        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                               hidden_size, num_heads, None, attention_dropout)

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding='left',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)
        # self.layer_norm_end = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """

        x, encoder_outputs, attention_weight, mask_src = inputs

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x)

        # Multi-head encoder-decoder attention
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs, mask_src)

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)

        # y = self.layer_norm_end(y)

        # Return encoder outputs as well to work with nn.Sequential
        return y, encoder_outputs, attention_weight


class DecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(DecoderLayer, self).__init__()

        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                           hidden_size, num_heads, bias_mask, attention_dropout)

        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                               hidden_size, num_heads, None, attention_dropout)

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding='left',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)
        # self.layer_norm_end = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """
        x, encoder_outputs, attention_weight, mask = inputs
        mask_src, dec_mask = mask

        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)

        # Masked Multi-head attention
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)

        # Dropout and residual after self-attention
        x = self.dropout(x + y)

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x)

        # Multi-head encoder-decoder attention
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs, mask_src)

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)

        # y = self.layer_norm_end(y)

        # Return encoder outputs as well to work with nn.Sequential
        return y, encoder_outputs, attention_weight, mask


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth,
                 num_heads, bias_mask=None, dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        # if total_key_depth % num_heads != 0:
        #     raise ValueError("Key depth (%d) must be divisible by the number of "
        #                      "attention heads (%d)." % (total_key_depth, num_heads))
        # if total_value_depth % num_heads != 0:
        #     raise ValueError("Value depth (%d) must be divisible by the number of "
        #                      "attention heads (%d)." % (total_value_depth, num_heads))

        if total_key_depth % num_heads != 0:
            print("Key depth (%d) must be divisible by the number of "
                  "attention heads (%d)." % (total_key_depth, num_heads))
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print("Value depth (%d) must be divisible by the number of "
                  "attention heads (%d)." % (total_value_depth, num_heads))
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5  ## sqrt
        self.bias_mask = bias_mask

        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.emotion_output_linear = nn.Linear(2 * output_depth, output_depth, bias=False)


        self.W_vad = nn.Parameter(torch.FloatTensor(1))

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values, mask):

        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)  # (bsz, heads, len, key_depth-20)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))  # (bsz, head, tgt_len, src_len)

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)

        ## attention weights
        attetion_weights = logits.sum(dim=1) / self.num_heads  # (bsz, tgt_len, src_len)

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)  # (bsz, 2, tgt_len, src_len)

        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)

        # Merge heads -> context vector
        contexts = self._merge_heads(contexts)
        # contexts = torch.tanh(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)  # 50 -> 300

        # return outputs, attetion_weights
        return outputs, torch.softmax(attetion_weights, dim=-1)



class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data),
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data),
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()

        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)] * (len(layer_config) - 2) +
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def _get_attn_subsequent_mask(size):
    """
    Get an attention mask to avoid using the subsequent info.
    Args:
        size: int
    Returns:
        (`LongTensor`):
        * subsequent_mask `[1 x size x size]`
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if (config.USE_CUDA):
        return subsequent_mask.cuda()
    else:
        return subsequent_mask

def gen_embeddings(vocab):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01
    print('Embeddings: %d x %d' % (vocab.n_words, config.emb_dim))
    if config.emb_file is not None:
        print('Loading embedding file: %s' % config.emb_file)
        pre_trained = 0
        for line in open(config.emb_file).readlines():
            sp = line.split()
            if(len(sp) == config.emb_dim + 1):
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings

class Embeddings(nn.Module):
    def __init__(self,vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

def share_embedding(vocab, pretrain=True):
    embedding = Embeddings(vocab.n_words, config.emb_dim, padding_idx=config.PAD_idx)
    if(pretrain):
        pre_embedding = gen_embeddings(vocab)
        embedding.lut.weight.data.copy_(torch.FloatTensor(pre_embedding))
        embedding.lut.weight.data.requires_grad = True
    return embedding


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.size()[0] > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_input_from_batch(batch):
    enc_batch = batch["input_batch"]
    enc_lens = batch["input_lengths"]
    batch_size, max_enc_len = enc_batch.size()
    assert len(enc_lens) == batch_size

    enc_padding_mask = sequence_mask(enc_lens, max_len=max_enc_len).float()

    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = batch["input_ext_vocab_batch"]
        # max_art_oovs is the max over all the article oov list in the batch
        if batch["max_art_oovs"] > 0:
            extra_zeros = torch.zeros((batch_size, batch["max_art_oovs"]))

    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim))

    coverage = None
    if config.is_coverage:
        coverage = torch.zeros(enc_batch.size())

    if config.USE_CUDA:
        enc_padding_mask = enc_padding_mask.cuda()
        if enc_batch_extend_vocab is not None:
                enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
        c_t_1 = c_t_1.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

def get_output_from_batch(batch):

    dec_batch = batch["target_batch"]

    if(config.pointer_gen):
        target_batch = batch["target_ext_vocab_batch"]
    else:
        target_batch = dec_batch

    dec_lens_var = batch["target_lengths"]
    max_dec_len = max(dec_lens_var)

    assert max_dec_len == target_batch.size(1)

    dec_padding_mask = sequence_mask(dec_lens_var, max_len=max_dec_len).float()

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch



def evaluate(model, data,  ty='valid', max_dec_step=30, write_summary=False):
    emo_map = {
        'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
        'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
        'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23,
        'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}
    emo_map = {v:k for k,v in emo_map.items()}

    model.__id__logger = 0
    dial = []
    ref, hyp_g, hyp_b, hyp_t = [],[],[],[]
    if ty=="test":
        print("testing generation:")
    t = Translator(model, model.vocab)
    l = []
    p = []
    bce = []
    acc = []
    pbar = tqdm(enumerate(data),total=len(data))
    inf_results = []
    try:
        for j, batch in pbar:
            loss, ppl, bce_prog, acc_prog = model.train_one_batch(batch, 0, train=False)

            l.append(loss)
            p.append(ppl)
            bce.append(bce_prog)
            acc.append(acc_prog)
            if(ty =="test"):
                sent_g, vader_score, emotion_id = model.decoder_greedy(batch,max_dec_step=max_dec_step)
                sent_b = t.beam_search(batch, max_dec_step=max_dec_step)
                sent_t = model.decoder_topk(batch, max_dec_step=max_dec_step)
                for i, (greedy_sent, beam_sent, topk_sent)  in enumerate(zip(sent_g, sent_b, sent_t)):
                    rf = " ".join(batch["target_txt"][i])
                    hyp_g.append(greedy_sent)
                    hyp_b.append(beam_sent)
                    hyp_t.append(topk_sent)
                    ref.append(rf)
                    # print_custom
                    temp = write_custum(emotion= batch["program_txt"][i], vader_score=vader_score, emo=emo_map[emotion_id],
                                dial=[" ".join(s) for s in batch['input_txt'][i]] if config.dataset=="empathetic" else " ".join(batch['input_txt'][i]),
                                ref=rf,
                                hyp_t=topk_sent,
                                hyp_g=greedy_sent,
                                hyp_b=beam_sent)
                    inf_results.append(temp)
            pbar.set_description("loss:{:.4f} ppl:{:.1f}".format(np.mean(l),math.exp(np.mean(l))))
    except KeyboardInterrupt:
        print("Only testing for a fraction of testing dataset, do not use this result!")

    loss = np.mean(l)
    ppl = np.mean(p)
    bce = np.mean(bce)
    acc = np.mean(acc)
    bleu_score_g = moses_multi_bleu(np.array(hyp_g), np.array(ref), lowercase=True)
    bleu_score_b = moses_multi_bleu(np.array(hyp_b), np.array(ref), lowercase=True)
    bleu_score_t = moses_multi_bleu(np.array(hyp_t), np.array(ref), lowercase=True)

    print("EVAL\tLoss\tPPL\tAccuracy\tBleu_g\tBleu_b\tBlue_t")
    print("{}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(ty,loss,math.exp(loss), acc, bleu_score_g,bleu_score_b, bleu_score_t))
    if write_summary:
        return loss, math.exp(loss), bce, acc, bleu_score_g, bleu_score_b, bleu_score_t, inf_results
    else:
        return loss, math.exp(loss), bce, acc, bleu_score_g, bleu_score_b, bleu_score_t

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def top_k_top_p_filtering(logits, top_k=0, top_p=0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits