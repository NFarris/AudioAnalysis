# Georgia Tech Robotic Musicianship Prosody Audio Analysis

## Background
The Robotic Musicianship group aims to facilitate meaningful musical interactions between humans and machines, leading to novel musical experiences and outcomes. In our research, we combine computational modeling approaches for perception, interaction, and improvisation, with novel approaches for generating acoustic responses in physical and visual manners.

The motivation for this work is based on the hypothesis that real-time collaboration between human and robotic players can capitalize on the combination of their unique strengths to produce new and compelling music. Our goal is to combine human qualities, such as musical expression and emotions, with robotic traits, such as powerful processing, the ability to perform sophisticated mathematical transformations, robust long-term memory, and the capacity to play accurately without practice.

## Purpose
The purpose of this repository is to provide documentation for the work performed by the Prosody Machine Learning Team. Human emotion is conveyed via many mediums, one of which is prosody. In linguistics, prosody is concerned with those elements of speech that are not individual phonetic segments but are properties of syllables and larger units of speech, including linguistic functions such as intonation, tone, stress, and rhythm. The main goals for this lab are 1. to classify emotion based on prosody and 2. to generate prosody for specific emotions.

## Methodology
During the summer of 2020, a 4.4 hour long dataset containing hundreds of audio clips labeled with a specific emotion was collected. This dataset, and more to come, will serve as the backbone for the models and algorithms generated this semester of Fall 2020. 

## Results from PyAudioAnalysis Feature Extraction and Model Training
During the first week with this dataset, our team decided to explore the accuracy of various classification models implemented by the popular python library PyAudioAnalysis. We trained and tested the following types of models: KNN, SVM, Random Forests, Extra Trees, and Gradient Boosting. Additionally we trained with two types of classification: All 20 emotions separated individually and the 20 emotions divided into 4 categories determined by the intersection of control and valance. The categories can be visualized as follows: 
| High Control, Negative Valance | High Control, Positive Valance | Low Control, Negative Valance | Low Control, Positive Valance |
|---|---|---|---|
| Anger | Amusement | Disappointment | Admiration |
| Contempt | Interest | Fear | Compassion |
| Disgust | Joy | Guilt | Contentment |
| Hate | Pleasure | Sadness | Love |
| Regret | Pride | Shame | Relief |

All models were trained with the following parameters consistent:
- Mid-term Window Step = 1.0 seconds
- Mid-term Window Size = 1.0 seconds
- Short-term Window Step = 0.05 seconds
- Short-term Window Size = 0.05 seconds
- computeBeat = False

The models achieved the following accuracies and f1 scores:
| Model | Classification | Best Accuracy | Best F1 | Best Hyperparameter |
|---|---|---|---|---|
| KNN | Big4 | 56.1 | 56.2 | C=11 |
| SVM | Big4  | 66.5 | 65.3 | C=1.0 |
| Extra Trees |  Big4 | 64.6 | 64.3 | C=100 |
| Gradient Boosting | Big4  | 67.0 | 66.7 | C=500 |
| Random Forest | Big4  | 63.5 | 63.2 | C=200 |
| KNN | Individual | 33.8 | 32.1 | C=15 |
| SVM | Individual | 49.1 | 48.1 | C=5.0 |
| Extra Trees | Individual | 44.3 | 42.8 | C=500 |
| Gradient Boosting | Individual | 47.2 | 46.6 | C=200 |
| Random Forest | Individual | 43.8 | 42.3 | C=200 |
|SVM|Love Vs Disgust|98.9|98.9|C=.01|

### Big4 Classification
The confusion matrix for Gradient Boosting on the Big4 Classification has been shown below. Notice the difficulty the model has in distinguishing between Low Control Positive Valance and High Control Positive Valance.

||HCN|HCP|LCN|LCP|
|---|---|---|---|---|
|HCN|20.82|2.38|2.46|1.39|
|HCP|2.21|15.16|2.38|5.66|
|LCN|2.87|2.13|15.66|2.30|
|LCP|1.39|5.16|2.62|15.41|

### Individual Classification
The confusion matrix for SVM on the Individual Classification has been shown below. Although this can be hard to read, it is useful to visualize which emotions are being confused for each other. A few interesting observations include: 

- Disgust is rarely confused with other emotions. 
- Fear and Guilt are the two most common pair of emotions to confuse for one another. 
- Pleasure is the hardest emotion to correctly classify.

||Adm|Amu|Ang|Com|Con|Con|Disa|Disg|Fear|Gui|Hate|Int|Joy|Love|Ple|Pri|Reg|Rel|Sad|Sha|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Adm|2.29|0.41|0.08|0.00|0.08|0.00|0.24|0.00|0.00|0.00|0.16|0.24|0.24|0.08|0.16|0.00|0.00|0.08|0.00|0.00|
|Amu|0.16|3.18|0.00|0.00|0.00|0.00|0.00|0.16|0.08|0.00|0.08|0.33|0.33|0.00|0.00|0.08|0.00|0.08|0.00|0.00|
|Ang|0.00|0.08|2.94|0.00|0.41|0.08|0.00|0.16|0.16|0.08|0.41|0.00|0.08|0.08|0.08|0.08|0.00|0.08|0.00|0.16|
|Com|0.08|0.00|0.00|2.20|0.08|0.41|0.49|0.00|0.16|0.16|0.00|0.16|0.00|0.16|0.00|0.41|0.00|0.00|0.00|0.16|
|Con|0.33|0.08|0.65|0.00|1.88|0.00|0.16|0.16|0.49|0.33|0.33|0.08|0.08|0.00|0.16|0.00|0.41|0.08|0.00|0.08|
|Con|0.49|0.16|0.08|0.98|0.08|1.71|0.08|0.00|0.08|0.16|0.00|0.08|0.00|0.08|0.33|0.08|0.16|0.00|0.24|0.08|
|Disa|0.08|0.00|0.16|0.16|0.00|0.00|2.45|0.16|0.08|0.33|0.00|0.08|0.08|0.00|0.08|0.08|0.00|0.16|0.00|0.16|
|Disg|0.08|0.33|0.24|0.00|0.08|0.00|0.16|5.31|0.16|0.00|0.08|0.00|0.00|0.00|0.00|0.00|0.00|0.08|0.00|0.00|
|Fear|0.08|0.00|0.24|0.08|0.33|0.00|0.08|0.08|2.53|0.49|0.16|0.00|0.00|0.08|0.24|0.08|0.08|0.08|0.08|0.16|
|Gui|0.08|0.08|0.08|0.08|0.08|0.24|0.08|0.00|1.14|2.45|0.00|0.00|0.00|0.00|0.24|0.24|0.24|0.00|0.08|0.16|
|Hate|0.00|0.16|0.57|0.00|0.33|0.00|0.00|0.24|0.16|0.00|3.35|0.08|0.33|0.24|0.00|0.00|0.00|0.08|0.16|0.00|
|Int|0.24|0.82|0.00|0.24|0.16|0.65|0.16|0.00|0.00|0.00|0.00|2.12|0.00|0.08|0.24|0.16|0.08|0.08|0.16|0.08|
|Joy|0.49|0.57|0.08|0.16|0.16|0.00|0.08|0.00|0.08|0.00|0.33|0.08|2.53|0.33|0.41|0.08|0.08|0.24|0.00|0.00|
|Love|0.16|0.16|0.08|0.82|0.08|0.41|0.00|0.00|0.16|0.24|0.08|0.24|0.16|1.96|0.33|0.33|0.00|0.00|0.08|0.00|
|Ple|0.41|0.00|0.00|0.73|0.16|0.49|0.24|0.08|0.00|0.41|0.00|0.41|0.00|0.82|1.06|0.16|0.24|0.16|0.08|0.24|
|Pri|0.24|0.00|0.16|0.16|0.08|0.16|0.08|0.00|0.08|0.08|0.00|0.00|0.08|0.33|0.08|2.37|0.00|0.16|0.00|0.00|
|Reg|0.08|0.00|0.24|0.41|0.08|0.16|0.16|0.08|0.08|0.49|0.16|0.08|0.00|0.00|0.41|0.16|1.47|0.16|0.08|0.16|
|Rel|0.16|0.00|0.08|0.16|0.16|0.08|0.16|0.33|0.00|0.08|0.24|0.16|0.16|0.08|0.08|0.00|0.00|4.00|0.16|0.00|
|Sad|0.00|0.00|0.33|0.08|0.00|0.16|0.24|0.16|0.00|0.00|0.24|0.24|0.00|0.08|0.00|0.00|0.00|0.24|1.71|0.16|
|Sha|0.08|0.00|0.08|0.16|0.16|0.33|0.24|0.08|0.65|0.33|0.16|0.00|0.00|0.16|0.08|0.33|0.33|0.08|0.08|1.55|

### Disgust vs Love Classification
When tasked with categorizing between two emotions, our model performs extremely well. Achieving accuracies as high as 98.9% with a f1 of 98.9 in the distinction between Love and Disgust using a SVM. This reinforces the intuition that by reducing the number of emotional categories we can achieve exceptionally high accuracies for identification. The confusion matrix is shown below:
||Dis|Love|
|---|---|---|
|Dis|54.03|1.14|
|Love|0.00|44.83|


## Understanding Feature Extraction 
//TODO: Add the following sections to each of the features listed below: Description, Why is this feature important?, Algorithm for extraction
### Zero Crossing Rate
#### Description

#### Why is this important?

#### Algorithm

### Energy
#### Description

#### Why is this important?

#### Algorithm

#### Example Extraction

### Entropy of Energy
#### Description

#### Why is this important?

#### Algorithm

#### Example Extraction

### Spectral Centroid
#### Description

#### Why is this important?

#### Algorithm

#### Example Extraction

### Spectral Entropy
#### Description

#### Why is this important?

#### Algorithm

#### Example Extraction

### Spectral Flux
#### Description

#### Why is this important?

#### Algorithm

#### Example Extraction

### Spectral Rolloff
#### Description

#### Why is this important?

#### Algorithm

#### Example Extraction

### Mel Frequency Cepstral Coefficients
#### Description

#### Why is this important?

#### Algorithm

#### Example Extraction

### Chroma Vector
#### Description

#### Why is this important?

#### Algorithm

#### Example Extraction

### Chroma Deviation
#### Description

#### Why is this important?

#### Algorithm

#### Example Extraction
