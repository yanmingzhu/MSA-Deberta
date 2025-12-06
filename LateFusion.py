from transformers import DebertaV2Model
import torch
import torch.nn as nn
from config import *

class LateDebertaMM(nn.Module):
  def __init__(self, hidden_dim=512, nhead=8, nlayer=4, modal='av'):
    super(LateDebertaMM, self).__init__()
    self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')

    audio_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
    self.withAudio = 'a' in modal
    self.withVision = 'v' in modal
    print(f"audio {self.withAudio} vision {self.withVision}")
    if self.withAudio:
        self.audio_encoder = nn.Sequential(
            nn.Linear(AUDIO_DIM, hidden_dim),
            nn.TransformerEncoder(audio_encoder_layer, num_layers=nlayer)
        )

    if self.withVision:
        vision_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.vision_encoder = nn.Sequential(
            nn.Linear(VISION_DIM, hidden_dim),
            nn.TransformerEncoder(vision_encoder_layer, num_layers=nlayer)
        )



    self.fc1 = nn.Linear(TEXT_DIM + hidden_dim*(self.withAudio+self.withVision), hidden_dim)
    self.dropout = nn.Dropout(0.2)
    self.fc2 = nn.Linear(hidden_dim, 1)

  def forward(self, text, audio, vision, masks=None):
    bert_output = self.deberta(input_ids=text, attention_mask=masks)
    last_hidden = bert_output.last_hidden_state
    cls = last_hidden[:, 0]

    audio_out = torch.empty(0).to(text.device)
    if self.withAudio:
        audio_out = self.audio_encoder(audio)
        audio_out = torch.mean(audio_out, dim=1)

    vision_out = torch.empty(0).to(text.device)
    if self.withVision:
        vision_out = self.vision_encoder(vision)
        vision_out = torch.mean(vision_out, dim=1)

    pred = self.fc1(torch.cat((cls, audio_out, vision_out), dim=1))
    pred = torch.relu(pred)
    pred = self.dropout(pred)
    pred = self.fc2(pred)
    #print(f"final pred ==== {pred[2:4, :].squeeze()}")

    return pred