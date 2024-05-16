## Shiryoku (思慮, eyesight)

_Still in development...._

A vision model(Image caption) using a Convnet (interchanges with Vision transformer) for the encoder and a standard RNN/LSTM decoder. It was built using Pytorch and trained on the Moondream dataset. 

### Architecture
- Encoder -> Custom ConvNet for image feature extraction
- Decoder -> LSTM model(RNN) for caption generation/prediction

### Version information
v0 -> Basic CNN and RNN fusion