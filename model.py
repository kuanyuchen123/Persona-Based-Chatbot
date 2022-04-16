from base64 import decode, encode
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Info
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class Transformer_Encoder(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        dropout_p
    ):
        super().__init__()
        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src, src_pad_mask=None):
        """
        src: (batch_size, sequence length)
        src embeded + pos: (batch_size, sequence length, dim_model)
        encoder_out: (sequence length, batch_size, dim_model)
        """

        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        
        # convert to (sequence length, batch_size, dim_model)
        src = src.permute(1,0,2)
        encoder_hidden = self.encoder(src, src_key_padding_mask=src_pad_mask)
        return encoder_hidden
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)

class Transformer_Decoder(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_decoder_layers,
        dropout_p
        ):

        super().__init__()
        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.output_layer = nn.Linear(dim_model, num_tokens)

    def forward(self, encoder_hidden, tgt, tgt_mask=None, tgt_pad_mask=None, src_pad_mask=None):
        """
        tgt: (batch_size, sequence length)
        tgt embeded + pos: (batch_size, sequence length, dim_model)
        decoder_hidden: (sequence length, batch_size, dim_model)
        output: (sequence, batch_size, num_tokens)
        """

        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        tgt = self.positional_encoder(tgt)

        # convert to (sequence length, batch_size, dim_model)
        tgt = tgt.permute(1,0,2)
        decoder_hidden = self.decoder(tgt, encoder_hidden, tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.output_layer(decoder_hidden)
        output = F.log_softmax(output, dim=2)
        return decoder_hidden, output

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)

class Knowledge_Manager(nn.Module):
    def __init__(self):
        super().__init__()
