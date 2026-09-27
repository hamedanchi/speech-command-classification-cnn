# Speech Command Classification with CNN and Spectrograms

A deep learning project for recognizing spoken commands from raw audio using **Short-Time Fourier Transform (STFT) spectrograms** and a **Convolutional Neural Network (CNN)** implemented with TensorFlow/Keras.

The project demonstrates an end-to-end audio classification pipeline, from raw waveform processing to spectrogram generation, CNN training, evaluation, and inference on individual audio files.

---

## Overview

Speech command recognition is a fundamental task in audio machine learning and is commonly used in voice-controlled systems, embedded devices, virtual assistants, and human-computer interaction applications.

In this project, short audio recordings containing spoken commands are transformed from their raw waveform representation into time-frequency spectrograms.

A convolutional neural network then learns discriminative patterns within these spectrograms to classify the spoken command.

The model recognizes eight commands:

```text
down
go
left
no
right
stop
up
yes
```

---

## Project Pipeline

The complete workflow is:

```text
Raw Audio (.wav)
       │
       ▼
Waveform
       │
       ▼
Short-Time Fourier Transform (STFT)
       │
       ▼
Magnitude Spectrogram
       │
       ▼
Normalization
       │
       ▼
Convolutional Neural Network
       │
       ▼
Class Scores
       │
       ▼
Softmax Probabilities
       │
       ▼
Predicted Speech Command
```

---

## Dataset

This project uses the **Mini Speech Commands** dataset provided by TensorFlow.

The dataset contains:

- **8,000 audio recordings**
- **8 speech command classes**
- approximately one-second audio clips
- audio sampled at **16 kHz**

The eight target classes are:

| Class |
|---|
| down |
| go |
| left |
| no |
| right |
| stop |
| up |
| yes |

The dataset is automatically downloaded when the notebook is executed.

---

## Dataset Split

The data is divided into approximately:

| Dataset | Percentage |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Test | 10% |

The initial dataset loader creates an 80/20 training-validation split. The validation subset is subsequently divided into separate validation and test datasets.

---

## Audio Representation

Each audio sample is represented as a waveform containing up to:

```text
16,000 samples
```

corresponding to approximately one second of audio sampled at:

```text
16 kHz
```

Example waveform shape:

```text
(16000,)
```

---

## Spectrogram Generation

Raw audio waveforms are converted into spectrograms using the **Short-Time Fourier Transform (STFT)**.

The transformation is implemented with:

```python
tf.signal.stft(
    waveform,
    frame_length=255,
    frame_step=128
)
```

The magnitude of the complex STFT output is then calculated:

```python
spectrogram = tf.abs(spectrogram)
```

A channel dimension is added so that the spectrogram can be processed as image-like input by convolutional layers.

Example spectrogram shape:

```text
124 × 129 × 1
```

This representation allows the neural network to learn both frequency and temporal characteristics of the spoken commands.

---

## Model Architecture

The classifier is implemented using a Convolutional Neural Network.

The original architecture contains:

```text
Input Spectrogram
       │
       ▼
Resize
       │
       ▼
Normalization
       │
       ▼
Conv2D — 32 filters
       │
       ▼
Conv2D — 64 filters
       │
       ▼
MaxPooling2D
       │
       ▼
Dropout
       │
       ▼
Flatten
       │
       ▼
Dense — 128 units
       │
       ▼
Dropout
       │
       ▼
Dense — 8 output units
```

The final layer produces logits for the eight speech command classes.

---

## Training

The model is trained using the Adam optimizer:

```python
optimizer=tf.keras.optimizers.Adam()
```

The loss function is:

```python
SparseCategoricalCrossentropy(from_logits=True)
```

Sparse categorical cross-entropy is appropriate because the labels are represented as integer class indices rather than one-hot encoded vectors.

Training is performed with early stopping to reduce unnecessary training once validation performance stops improving.

---

## Evaluation

The model achieved approximately:

| Metric | Result |
|---|---:|
| Test Accuracy | **84.01%** |
| Test Loss | **0.4972** |

These results demonstrate that even a relatively compact CNN can learn useful time-frequency patterns from speech spectrograms.

---

## Confusion Matrix

A confusion matrix is generated after evaluation to analyze classification errors across the eight speech commands.

This makes it possible to identify which commands are most frequently confused by the model.

For example, acoustically similar commands may produce similar spectrogram patterns and therefore be more difficult to distinguish.

---

## Single-Audio Prediction

The trained model can also classify individual `.wav` files.

The inference pipeline is:

```text
Audio File
    │
    ▼
Decode WAV
    │
    ▼
Waveform
    │
    ▼
Spectrogram
    │
    ▼
CNN
    │
    ▼
Softmax
    │
    ▼
Predicted Command
```

The predicted probabilities can also be visualized as a bar chart for all eight classes.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Digital Signal Processing
- Short-Time Fourier Transform
- Convolutional Neural Networks
- Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/speech-command-classification-cnn.git
```

Move into the project directory:

```bash
cd speech-command-classification-cnn
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Suggested Requirements

```text
tensorflow
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

For reproducible environments, exact package versions can be pinned in `requirements.txt`.

---

## Running the Project

1. Clone the repository.
2. Install the required dependencies.
3. Open the Jupyter notebook.
4. Run all cells sequentially.
5. The Mini Speech Commands dataset will be downloaded automatically.
6. Audio waveforms will be transformed into spectrograms.
7. The CNN will be trained on the generated spectrograms.
8. Evaluate the model on the test dataset.
9. Inspect the confusion matrix.
10. Test the trained model on individual audio samples.

---

## Possible Improvements

Several improvements could further increase the performance and robustness of the project:

- Use deeper CNN architectures
- Replace `Flatten` with `GlobalAveragePooling2D`
- Experiment with different STFT parameters
- Compare different spectrogram resolutions
- Use Mel spectrograms
- Use MFCC features
- Apply audio data augmentation
- Add background noise augmentation
- Add learning-rate scheduling
- Save the best model using `ModelCheckpoint`
- Report precision, recall, and F1-score for each class
- Compare CNN performance with recurrent neural networks
- Experiment with CRNN architectures
- Compare the model with pretrained audio networks

---

## Future Work

A production-oriented version of this project could expose the speech classifier through an API.

For example:

```text
Audio File
    │
    ▼
FastAPI Endpoint
    │
    ▼
Audio Preprocessing
    │
    ▼
Spectrogram Generation
    │
    ▼
CNN Model
    │
    ▼
Predicted Command
    │
    ▼
JSON Response
```

The model could also be deployed as:

- a FastAPI service
- a Docker container
- a TensorFlow Lite model
- a mobile application
- an embedded voice-command system

---

## What I Learned

This project provided practical experience with:

- processing raw audio data
- waveform visualization
- Short-Time Fourier Transform
- spectrogram generation
- audio classification
- convolutional neural networks
- TensorFlow data pipelines
- multi-class classification
- confusion matrix analysis
- inference on individual audio files

---

## Author

**Omid Nobari**

Machine Learning & AI Projects

---

## License

This project is intended for educational and portfolio purposes.
