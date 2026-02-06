class FusionModel():
    def __init__(self, cnn_model, rnn_model, num_classes):
        pass
    #     super(FusionModel, self).__init__()
    #     self.cnn_model = cnn_model
    #     self.rnn_model = rnn_model
    #     self.fc = nn.Linear(cnn_model.fc_layers[-1].out_features + rnn_model.fc.out_features, num_classes)

    # def forward(self, image, text):
    #     cnn_features = self.cnn_model.fc_layers[:-1](self.cnn_model.conv_layers(image))
    #     cnn_features = cnn_features.view(cnn_features.size(0), -1)

    #     rnn_features = self.rnn_model.embedding(text)
    #     _, (h, _) = self.rnn_model.rnn(rnn_features)
    #     rnn_final_hidden_state = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
    #     rnn_final_hidden_state = self.rnn_model.dropout(rnn_final_hidden_state)

    #     combined_features = torch.cat((cnn_features, rnn_final_hidden_state), dim=1)
    #     out = self.fc(combined_features)
    #     return out