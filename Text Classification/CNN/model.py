import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_channels, 
                 kernel_size, max_sen_len, output_size, dropout):
        super(TextCNN, self).__init__()
        # https://github.com/AnubhavGupta3377/Text-Classification-
        # Models-Pytorch/blob/master/Model_TextCNN/model.py
        self.model_type = 'CNN'

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        self.convs = nn.ModuleList([nn.Sequential(
                                    nn.Conv1d(in_channels=embed_size, out_channels=num_channels, kernel_size=kernel),
                                    nn.ReLU(),
                                    nn.MaxPool1d(max_sen_len - kernel+1)) for kernel in kernel_size])
        
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels*len(kernel_size), output_size)
        
        
    def forward(self, x):
        embedded_sent = self.embeddings(x.T).permute(1,2,0)
        conv_outs = [conv(embedded_sent).squeeze(2) for conv in self.convs]

        all_out = torch.cat(conv_outs, dim=1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)
        return final_out