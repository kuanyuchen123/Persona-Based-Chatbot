from data import load_data, PersonaDataset
from model import Transformer_Encoder, Transformer_Decoder
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import params
import os
import tokenizer

def train(encoder, decoder, optimizer, data_loader):
    NLLLoss = nn.NLLLoss(ignore_index=params.PAD)
    for epoch in range(params.n_epoch):
        for step, (src_X, src_Y, tgt_Y, src_K) in enumerate(data_loader):
            src_X = src_X.to(device)
            src_Y = src_Y.to(device)
            tgt_Y = tgt_Y.to(device)
            src_K = src_K.to(device)

            sequence_length = src_Y.size(1)
            src_pad_mask = encoder.create_pad_mask(src_X, params.PAD).to(device)
            tgt_pad_mask = decoder.create_pad_mask(src_Y, params.PAD).to(device)
            tgt_mask = decoder.get_tgt_mask(sequence_length).to(device)
            optimizer.zero_grad()
            encoder_hidden = encoder(src_X, src_pad_mask)
            hidden, outputs = decoder(encoder_hidden, src_Y, tgt_mask, tgt_pad_mask, src_pad_mask)
            nll_loss = NLLLoss(outputs.permute(1,2,0), tgt_Y)
            nll_loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:
                nll_loss /= 50
                print(
                    "Epoch [%.2d/%.2d] Step [%.4d/%.4d]: nll_loss=%.4f"
                    % (
                        epoch + 1,
                        params.n_epoch,
                        step + 1,
                        len(data_loader),
                        nll_loss,
                    )
                )
                nll_loss = 0

        torch.save(encoder.state_dict(), "./checkpoint/encoder_{}.pt".format(epoch))
        torch.save(decoder.state_dict(), "./checkpoint/decoder_{}.pt".format(epoch))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print( "Set device to {}".format( device ) )

if os.path.exists("vocab.json"):
    print("vocab dictionary built...")
else :
    print("building vocab dictionary...")
    tokenizer.tokenize()

print("loading model...")
encoder = Transformer_Encoder(
    num_tokens=params.n_vocab, dim_model=params.model_d, num_heads=params.n_head, \
    num_encoder_layers=params.n_encode_layer, dropout_p=params.dropout
).to(device)

decoder = Transformer_Decoder(
    num_tokens=params.n_vocab, dim_model=params.model_d, num_heads=params.n_head, 
    num_decoder_layers=params.n_decode_layer, dropout_p=params.dropout
).to(device)

parameters = (
    list(encoder.parameters())
    + list(decoder.parameters())
)

print("loading data...")
X, Y, K = load_data("./data/train.txt")
dataset = PersonaDataset(X, Y, K)
data_loader = DataLoader(dataset=dataset, batch_size=params.n_batch, shuffle=True)
optimizer = optim.Adam(parameters, lr=params.lr)

print("start training...")
train(encoder, decoder, optimizer, data_loader)