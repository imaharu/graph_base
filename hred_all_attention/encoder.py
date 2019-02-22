from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class WordEncoder(nn.Module):
    def __init__(self, opts):
        super(WordEncoder, self).__init__()
        self.opts = opts
        self.embed = nn.Embedding(source_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=self.opts["bidirectional"])
        self.W_whs = nn.Linear(hidden_size, hidden_size)

    def forward(self, sentences, max_s_len):
        current_gpu = torch.cuda.current_device()
        b = sentences.size(0)
        lengths = sentences.ne(0).sum(-1)
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = sentences.index_select(0, idx_sort)
        for i, _ in enumerate(lengths_sort):
            if lengths_sort[i] == 0:
                lengths_sort[i] = 1
        embed = self.embed(x_sort)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths_sort, batch_first=True)
        o_pack, hx = self.gru(x_pack)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack)
        o_unsort = o.index_select(1, idx_unsort)
        hx = hx.transpose(0,1).index_select(0, idx_unsort)
        if self.opts["bidirectional"]:
            word_outputs = o_unsort[:, :, :hidden_size] + o_unsort[:, :, hidden_size:]
            hx = hx.view(-1, 2 , b, hidden_size).sum(1)
        hx = hx.view(b , -1)
        word_outputs = F.pad(word_outputs, (0, 0, 0, 0, 0, max_s_len - word_outputs.size(0) ), "constant", 0)
        word_features = self.W_whs(word_outputs)
        return word_outputs, word_features, hx

class SentenceEncoder(nn.Module):
    def __init__(self, opts):
        super(SentenceEncoder, self).__init__()
        self.opts = opts
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=self.opts["bidirectional"])
        self.W_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, words_encoder_outputs):
        b = words_encoder_outputs.size(1)
        current_gpu = torch.cuda.current_device()
        input_sentences = words_encoder_outputs.transpose(0,1)
        input_lengths = input_sentences.ne(0).sum(1).t()[0]
        sorted_lengths, indices = torch.sort(input_lengths, descending=True)
        input_sentences = input_sentences[indices]
        sequence = rnn.pack_padded_sequence(input_sentences, sorted_lengths, batch_first=True)
        self.gru.flatten_parameters()
        sentence_outputs, s_hx = self.gru(sequence)
        sentence_outputs, _ = rnn.pad_packed_sequence(
            sentence_outputs
        )

        inverse_indices = indices.sort()[1] # Inverse permutation
        sentence_outputs = sentence_outputs[:, inverse_indices]
        s_hx = s_hx[:, inverse_indices]

        if self.opts["bidirectional"]:
            sentence_outputs = sentence_outputs[:, :, :hidden_size] + sentence_outputs[:, :, hidden_size:]
            s_hx = s_hx.view(-1, 2 , b, hidden_size).sum(1)
        ms_inputs_len = input_lengths.max()
        ms_len = words_encoder_outputs.size(0)
        if ms_inputs_len != ms_len:
            s_pad = torch.zeros(ms_len - ms_inputs_len, b, hidden_size).cuda(current_gpu)
            sentence_outputs = torch.cat((sentence_outputs, s_pad), 0)
        sentence_features = self.W_h(sentence_outputs)
        s_hx = s_hx.view(b, -1)
        return sentence_outputs, sentence_features, s_hx
