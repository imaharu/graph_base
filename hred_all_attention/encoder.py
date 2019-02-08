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
        b = sentences.size(0)
        input_lengths = sentences.ne(0).sum(-1)
        current_gpu = torch.cuda.current_device()
        except_flag = False
        # もし,全て０の時
        print(input_lengths)
        print(input_lengths.max())
        if input_lengths.max().item() == 0:
            p_w_hx = torch.zeros((b, hidden_size)).cuda(current_gpu)
            p_words_hx = torch.zeros((max_s_len, b, hidden_size)).cuda(current_gpu)
            return p_words_hx, p_words_hx, p_w_hx

        sorted_lengths, indices = torch.sort(input_lengths, descending=True)
        sentences = sentences[indices]
        masked_select = sorted_lengths.masked_select(sorted_lengths.ne(0))

        # if all 0 sentence is appeared
        if not torch.equal(sorted_lengths, masked_select):
            s_b = b
            b = masked_select.size(0)
            sentences = sentences.narrow(0, 0, b)
            sorted_lengths = masked_select
            except_flag = True

        embed = self.embed(sentences)
        sequence = rnn.pack_padded_sequence(embed, sorted_lengths, batch_first=True)
        self.gru.flatten_parameters()
        word_outputs, w_hx = self.gru(sequence)
        word_outputs, _ = rnn.pad_packed_sequence(
            word_outputs
        )

        if self.opts["bidirectional"]:
            word_outputs = word_outputs[:, :, :hidden_size] + word_outputs[:, :, hidden_size:]
            w_hx = w_hx.view(-1, 2 , b, hidden_size).sum(1)
        w_hx = w_hx.view(b , -1)
        if except_flag:
            outputs_zeros = torch.zeros((word_outputs.size(0) ,s_b - b, hidden_size)).cuda(current_gpu)
            word_outputs = torch.cat((word_outputs, outputs_zeros), 1)
            zeros = torch.zeros((s_b - b, hidden_size)).cuda(current_gpu)
            w_hx = torch.cat((w_hx, zeros), 0)
        inverse_indices = indices.sort()[1] # Inverse permutation
        word_outputs = word_outputs[:,inverse_indices,:]
        w_hx = w_hx[inverse_indices]
        word_outputs = F.pad(word_outputs, (0, 0, 0, 0, 0, max_s_len - word_outputs.size(0) ), "constant", 0)
        word_features = self.W_whs(word_outputs)
        return word_outputs, word_features, w_hx

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
