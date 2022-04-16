import torch
import params
from tokenizers import Tokenizer
from model import Transformer_Encoder, Transformer_Decoder
from beam import beam_decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print( "Set device to {}".format( device ) )

print("loading model...")
encoder = Transformer_Encoder(
    num_tokens=params.n_vocab, dim_model=params.model_d, num_heads=params.n_head, \
    num_encoder_layers=params.n_encode_layer, dropout_p=params.dropout
).to(device)

decoder = Transformer_Decoder(
    num_tokens=params.n_vocab, dim_model=params.model_d, num_heads=params.n_head, 
    num_decoder_layers=params.n_decode_layer, dropout_p=params.dropout
).to(device)

encoder.load_state_dict(torch.load("./checkpoint/encoder_14.pt"))
decoder.load_state_dict(torch.load("./checkpoint/decoder_14.pt"))
tokenizer = Tokenizer.from_file("./vocab.json")

print( "Start chatting with our chatbot!" )
while True :
    utterence = input("you: ")
    if utterence == "exit" : break 
    tokenized_utterence = torch.tensor(tokenizer.encode(utterence.lower()).ids).unsqueeze(0).to(device)
    src_pad_mask = encoder.create_pad_mask(tokenized_utterence, params.PAD).to(device)
    encoder_hidden = encoder(tokenized_utterence, src_pad_mask)
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        result = beam_decode(decoder, encoder_hidden, src_pad_mask, device)

    result = tokenizer.decode(result[0])
    print("bot: {}".format(result))