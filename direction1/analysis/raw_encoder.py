import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class PrismEncoder(nn.Module):
    def __init__(
            self, lm_name, frq_filter, frq_tuning=False,
            specials=False, cache=None, max_length=512):
        super().__init__()

        self.max_length = max_length
        tokenizer_kwargs = {'use_fast': True}
        if any(m in lm_name for m in ['gpt2', 'bloom', 'Mistral', 'aya']):
            tokenizer_kwargs['add_prefix_space'] = True
        self._tok = transformers.AutoTokenizer.from_pretrained(
            lm_name, **tokenizer_kwargs)

        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        self._lm = transformers.AutoModel.from_pretrained(
            lm_name, return_dict=True, torch_dtype=torch.float32, device_map="auto")
        self._config = self._lm.config

        # load cache
        # {hash: torch.tensor (num_layers, sen_len, emb_dim)}
        self._cache = cache
        # internal variables
        self._lm_name = lm_name
        self._specials = specials
        # frequency filter parameter
        self._frq_filter = None if frq_filter is None else nn.Parameter(
            frq_filter)
        # public variables
        self.emb_dim = self._config.hidden_size
        self.num_layers = self._config.num_hidden_layers

    @staticmethod
    def load(path, frq_filter=None, frq_tuning=False, emb_pooling=None, specials=False, cache=None):
        objects = torch.load(path, weights_only=False)
        # load necessary components
        lm_name = objects['language_model_name']
        emb_pooling = objects.get('embedding_pooling', emb_pooling)
        frq_filter = objects.get('frequency_filter', frq_filter)
        # construct model
        encoder = PrismEncoder(
            lm_name=lm_name, frq_filter=frq_filter, frq_tuning=frq_tuning, specials=specials, cache=cache
        )
        # load LM (if available)
        encoder._lm = objects.get('language_model', encoder._lm)
        return encoder

    def forward(self, sentences, label_type):
        # embed sentences (standard last-layer encoding), removing places
        with torch.no_grad():
            emb_tokens, att_tokens = self.embed(sentences, label_type)

        return emb_tokens, att_tokens

    def embed(self, sentences, label_type):
        # compute embeddings if not in cache
        # tok_sentences = self.tokenize(sentences)
        if label_type == 'token':
            tok_sentences = self.tokenize(
                [s.split(' ') for s in sentences], tokenized=True)
        else:
            tok_sentences = self.tokenize(sentences)

        model_inputs = {
            k: tok_sentences[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']
            if k in tok_sentences
        }

        # perform embedding forward pass
        model_outputs = self._lm(**model_inputs, output_hidden_states=True)
        # (batch_size, max_len, hidden_dim)
        hidden_states = model_outputs.hidden_states[-1]
        emb_tokens, att_tokens = hidden_states, tok_sentences['attention_mask'].bool(
        )

        # remove special tokens
        if (not self._specials):
            # mask any non-padding token that is special
            non_special_mask = ~tok_sentences['special_tokens_mask'].bool(
            ) & att_tokens  # (batch_size, max_len)
            # new maximum length without special tokens
            non_special_max_len = torch.max(torch.sum(non_special_mask, -1))
            non_special_embeddings = torch.zeros_like(
                # (batch_size, new_max_len, hidden_dim)
                emb_tokens)[:, :non_special_max_len, :]
            non_special_attentions = torch.zeros_like(
                # (batch_size, new_max_len)
                att_tokens)[:, :non_special_max_len]
            # iterate over each sequence due to variable length
            for sidx in range(emb_tokens.shape[0]):
                # current sequence length up to which values should be filled
                seq_len = torch.sum(non_special_mask[sidx])
                # store non-special token embeddings up to new sequence length (rest is zeros)
                non_special_embeddings[sidx,
                                       :seq_len] = emb_tokens[sidx, non_special_mask[sidx], :]
                non_special_attentions[sidx,
                                       :seq_len] = att_tokens[sidx, non_special_mask[sidx]]
            # keep non-special embeddings only
            emb_tokens = non_special_embeddings
            att_tokens = non_special_attentions

        return emb_tokens, att_tokens

    def tokenize(self, sentences, tokenized=False):
        # tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]], special_tokens_mask: [[]]}
        # check for multi-sentence inputs
        if (len(sentences) > 0) and (type(sentences[0]) is tuple):
            sentences1, sentences2 = [s[0]
                                      for s in sentences], [s[1] for s in sentences]
            tok_sentences = self._tok(
                sentences1, sentences2,
                padding=True, truncation=True, max_length=self.max_length,
                return_tensors='pt', return_special_tokens_mask=True,
                return_offsets_mapping=True, is_split_into_words=tokenized
            )
        # use standard tokenization otherwise
        else:
            tok_sentences = self._tok(
                sentences,
                padding=True, truncation=True, max_length=self.max_length,
                return_tensors='pt', return_special_tokens_mask=True,
                return_offsets_mapping=True, is_split_into_words=tokenized
            )

        # move input to GPU (if available)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            for k, v in tok_sentences.items():
                if isinstance(v, torch.Tensor):
                    tok_sentences[k] = v.to(device)

        return tok_sentences


def dct(embeddings, norm=None):
    """
    (Implemented as in https://github.com/zh217/torch-dct)
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    orig_shape = embeddings.shape
    N = embeddings.shape[-1]
    embeddings = embeddings.contiguous().view(-1, N)

    v = torch.cat(
        [embeddings[:, ::2], embeddings[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    k = - torch.arange(N, dtype=embeddings.dtype,
                       device=embeddings.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*orig_shape)

    return V


def idct(frequencies, norm=None):
    """
    (Implemented as in https://github.com/zh217/torch-dct)
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    orig_shape = frequencies.shape
    N = orig_shape[-1]

    X_v = frequencies.contiguous().view(-1, orig_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(orig_shape[-1], dtype=frequencies.dtype,
                     device=frequencies.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    frequencies = v.new_zeros(v.shape)
    frequencies[:, ::2] += v[:, :N - (N // 2)]
    frequencies[:, 1::2] += v.flip([1])[:, :N // 2]

    return frequencies.view(*orig_shape)
