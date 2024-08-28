# SonicSort_Music_Classifier
Introduction

The streaming music industry has exploded in recent years, creating a demand for seamless, automated, and personalized user experiences. With millions of tracks uploaded daily, streaming services are increasingly challenged by the sheer volume of data they must manage. Accurate music genre classification serves as a critical component of these platforms, underpinning personalized recommendations, dynamic playlist generation, and user preference analytics. However, manual tagging of music tracks is a time-consuming and often inaccurate process, leading to suboptimal user experiences.

Project SonicSort addresses these challenges by automating music genre classification using state-of-the-art machine learning techniques. By automatically tagging unknown tracks with precise genre labels, we aim to enhance the discoverability and personalization capabilities of streaming applications. Our approach focuses on creating a robust classification system capable of categorizing audio clips into one of eight distinct genres. This system leverages advanced deep learning models and regression algorithms to analyze the nuanced interplay between frequency and time domain features, resulting in highly accurate genre predictions.

Problem Statement

The main challenge in the music streaming ecosystem is the manual management of extensive music libraries. Manual genre classification is labor-intensive, prone to errors, and lacks scalability. As a result, many tracks are mislabeled or left unclassified, which hampers the performance of recommendation systems and affects user engagement. SonicSort aims to automate this classification process, ensuring more accurate tagging, better music discovery, and improved user satisfaction.

Objectives

Automate Music Genre Classification: Build a robust system that can classify music tracks into 8 genres.
Explore Feature Extraction Techniques: Use Mel Frequency Cepstral Coefficients (MFCC) and Mel spectrogram features to capture the frequency and temporal characteristics of audio data.
Compare Machine Learning Models: Evaluate the performance of deep learning models against classical machine learning algorithms to determine the best approach for music genre classification.
Improve User Experience in Streaming Apps: By providing more accurate genre tags, we aim to enhance the music discovery process and personalize user recommendations in streaming applications.
Methodology

The core of SonicSort lies in its exploration of different machine learning approaches for music genre classification. The project employs three deep learning models alongside regression algorithms, allowing for a comprehensive evaluation of their respective strengths and weaknesses. The analysis focuses on two types of features derived from the audio data:

Mel Frequency Cepstral Coefficients (MFCC): MFCCs are widely used in audio processing as they capture the timbral texture of sound. These coefficients represent the short-term power spectrum of a sound and are crucial in identifying the unique characteristics of different music genres.
Mel Spectrogram Features: Mel spectrograms are a visual representation of the audio signal's power spectrum. They provide a detailed view of how different frequencies vary over time, which is essential for understanding genre-specific patterns.
These features serve as inputs to the models, enabling them to learn the complex relationships between audio characteristics and genre labels.

Models Employed

Convolutional Neural Networks (CNNs): CNNs are effective in capturing local patterns in data. By applying convolutional layers to spectrogram images, we extract spatial hierarchies of features that contribute to genre identification.
Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM): RNNs are suitable for sequential data like audio. LSTM cells help in preserving long-term dependencies, allowing the model to understand how audio features change over time, which is crucial for differentiating genres with similar musical elements.
Hybrid Deep Learning Models: Combining CNNs and RNNs, hybrid models are designed to leverage both spatial and sequential features of audio data, resulting in more robust classifications.
Classical Regression Models: Linear and logistic regression models provide a baseline for comparison. Although less powerful than deep learning models, these algorithms help in understanding the feature importance and linear relationships in the data.
Experimental Results

We trained and tested the models on a publicly available dataset, the Free Music Archive (FMA) dataset. The dataset consists of thousands of audio tracks labeled with their respective genres, providing a rich corpus for model training and evaluation.

Performance Metrics: Accuracy, precision, recall, and F1-score were used to evaluate the models. Deep learning models, particularly the hybrid models, outperformed classical regression models by a significant margin. The best-performing model achieved an accuracy of 92%, highlighting the effectiveness of deep learning approaches in capturing the complex, non-linear relationships in audio data.
Feature Analysis: MFCCs and Mel spectrograms proved to be highly informative features. The combination of frequency and time-domain features led to significant improvements in model performance, confirming the importance of both feature types for genre classification.
Conclusion

Project SonicSort demonstrates that advanced machine learning techniques, particularly deep learning models, can significantly improve the accuracy of music genre classification in streaming applications. By automating the tagging process and providing more accurate genre labels, we pave the way for more personalized and engaging user experiences.

Our approach not only enhances music discovery but also sets the stage for future innovations in music analytics, recommendation systems, and user preference modeling. As the streaming industry continues to evolve, projects like SonicSort will be at the forefront, harmonizing data with user needs in the most innovative ways.

Future Work

Expand Genre Classification: Extend the model to classify music into more than 8 genres, including sub-genres.
Real-Time Classification: Develop real-time genre classification algorithms to enhance the immediacy of music discovery features.
Integration with Streaming Services: Collaborate with major streaming platforms to deploy SonicSort models, enabling live testing and user feedback integration.


Dataset Link: https://github.com/mdeff/fma
