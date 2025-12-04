from transformers import DebertaV2Model

class TextDebertaMM(nn.Module):
  def __init__(self, dim=768):
    super(TextDebertaMM, self).__init__()
    self.dim = dim

    self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')

    self.fc1 = nn.Linear(dim, 512)
    self.dropout = nn.Dropout(0.2)
    self.fc2 = nn.Linear(512, 1)

  def forward(self, text, audio, vision, masks=None):
    '''
    text[torch.isinf(text)] = 0.0
    audio[torch.isinf(audio)] = 0.0
    vision[torch.isinf(vision)] = 0.0

    x_concat = torch.cat((text, audio, vision), dim=2)

    if torch.isinf(x_concat).any():
      inf_indices = torch.nonzero(torch.isinf(x_concat), as_tuple=False)
      print(f"infinities {inf_indices}")
      print(x_concat[inf_indices[0], inf_indices[1], :])
    '''
    bert_output = self.deberta(input_ids=text, attention_mask=masks)

    last_hidden = bert_output.last_hidden_state
    cls = last_hidden[:, 0]

    pred = self.fc1(cls)
    pred = torch.relu(pred)
    pred = self.dropout(pred)
    pred = self.fc2(pred)
    #print(f"final pred ==== {pred[2:4, :].squeeze()}")

    return pred