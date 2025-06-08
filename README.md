# Models

This repo contains various models developed with the purpose of extracting relevant text from an image.
Two different approaches were tried:

- [Text Detector Model](./src/text_detector_model/README.md), uses an ocr to detect the bounding boxes in the image containing relevant text
- [Text Extractor Model](./src/text_extractor_model/README.md) [PROTOTYPE], uses an autoencoder to generate a new image contianing only the relevant text 

### Code Style
```bash
black .
```