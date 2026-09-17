# Project

**Academic year: 2026-2027**

## **Project Description: Build a Deep Learning Pipeline with TensorFlow**

**Objective**  
Develop a complete deep learning pipeline using TensorFlow. Choose one domain—text, audio, or images—select a suitable dataset, and define a specific subject for your project. Develop a complete pipeline, including data preprocessing, model design, training, evaluation, and deployment.  Enhance your project by integrating Symbolic AI components for added functionality or interpretability.

---

**Steps to Complete the Project**  

1. **Choose Your Domain and Subject**
   - Select a domain: text, audio, or images.
   - Define a clear and specific subject related to your chosen domain. Examples include:
     - Text: emotion analysis, paraphrasing, question answering.
     - Images: image classification, object detection.
     - Audio: speech recognition, emotion detection, music vs. speech classification. 

3. **Select a Dataset**  
   - Pick a dataset from the provided sources or propose your own.  
   - Ensure the dataset is relevant to your chosen domain and task.  

4. **Preprocess the Data**  
   - Clean, transform, and augment the data as needed.  
   - Use TensorFlow tools like `tf.data` or `tf.keras.preprocessing` for efficient pipelines.  
   - For text, consider tokenization or embedding; for images, apply normalization or augmentation; for audio, extract features like spectrograms or MFCCs.

5. **Design and Train Your Model**  
   - Build a model suitable for your task:
     - For text: use RNNs, LSTMs, or Transformers.  
     - For images: use CNNs or pre-trained architectures like ResNet.  
     - For audio: combine feature extraction layers with RNNs or CNNs.  
   - Experiment with hyperparameters, activation functions, and layers.  
   - Train your model using TensorFlow and evaluate its performance on a validation set.  

6. **Integrate Symbolic AI (Optional Bonus)**  
   - Combine your model with rule-based or logic-driven systems to improve interpretability or accuracy.  
   - For example:
     - Use knowledge graphs in text analysis.  
     - Add reasoning components for emotion recognition in audio.  
     - Implement rule-based constraints for object detection in images.  

7. **Evaluate and Deploy**  
   - Assess your model using metrics appropriate to your task (e.g., accuracy, precision, recall).  
   - Deploy your model as an interactive application or notebook.  

---


**Deliverables**  

1. A complete TensorFlow implementation of your pipeline.  
2. A detailed report (the README of your submission) covering:
   - The chosen subject, problem statement, and objectives.
   - The dataset used (source, size, licence) and how to obtain it.
   - Preprocessing methods.  
   - Model architecture, training process and the experiments carried out.  
   - Evaluation results, error analysis, limitations and potential improvements.  
   - The sources used (documentation, papers, repositories, AI assistants) and the contribution of each member of the pair.  
3. A deployed demo or app (runnable notebook, web application, command-line interface, etc.).  

---

**Evaluation and submission**  

The project accounts for **100% of the course grade**; the practicals are not graded and are not to be submitted.
Submission is online (e-campus) following the [submission instructions](../README.md) (folder `group_N1_N2` with README, CONTRIBUTORS and `src/`). The deadline is given on e-campus.

**Grading criteria (out of 20)**

| Criterion | Points | What is expected |
|---|---|---|
| 1. Problem statement and data | 4 | Clear and feasible subject; suitable and well-described dataset; justified preprocessing (cleaning, train/validation/test split, augmentation, tokenization, spectrograms, …). |
| 2. Model and training | 6 | Architecture suited to the task and justified with respect to the course (MLP, CNN, RNN/LSTM/GRU, Transformer, transfer learning); documented experiments on hyperparameters (learning rate, batch size, epochs, regularization); learning curves. |
| 3. Evaluation and analysis | 4 | Metrics suited to the task (precision, recall, F1, confusion matrix, …); comparison of configurations; error analysis and limitations; honest discussion of the results. |
| 4. Code and report | 4 | Readable, commented and reproducible code (runs without errors, installation instructions); complete and well-structured report; sources cited; CONTRIBUTORS filled in. |
| 5. Demo and deployment | 2 | A working demonstration that lets the model be tried on new inputs. |
| Bonus: symbolic AI | +2 | Relevant integration of a symbolic component (rules, logic, knowledge graph, Prolog, Z3, …) that improves interpretability or results. The total grade is capped at 20. |

The report and the code must be the work of the pair; any reuse of code, pre-trained models or generated text must be cited. A short oral presentation may be requested to check the understanding of the project.


## Example Notebooks
 - [Data Processing in Tensorflow](Data.ipynb)
 - [Handwriting recognition using MNIST dataset](Introduction.ipynb)
 - [Text classification based on IMDB reviews](Texts.ipynb)
 - [Understanding Property Translation of Wikidata](miniproject-notebook.ipynb)

## Project domains 
 - Text
 - Images
 - Audio

## Datasets 
### Existing catalogues
 - https://www.kaggle.com/datasets
 - https://www.tensorflow.org/datasets
 - https://wordnet.princeton.edu/download
 - http://www.image-net.org/ 

### Domains
- **Text**:  
  - Start with datasets like IMDB reviews, SQuAD, or CoNLL-2003.  
  - Use pre-trained embeddings like GloVe, Word2Vec, or BERT.  

- **Images**:  
  - Use datasets such as CIFAR-10, ImageNet, or Oxford Flowers.  
  - Try transfer learning with TensorFlow’s pre-trained models.  

- **Audio**:  
  - Choose datasets like LibriSpeech or UrbanSound8K.  
  - Preprocess with audio-specific techniques like spectrograms.  


## Possible topics
 - Text
   - Language identification
   - Speaker identification
   - Question answering
      - yes or no answering
      - answers to questions related to multiline paragraphs
      - mathematical question answering
   - Analysis of citations
   - Analysis of reviews
   - Paraphrasing
   - Common knowledge facts
   - Common sense explanation 
   - Analysis of emotions
 - Images
   - Object detection
   - Image classification
 - Audio
   - Detection of music genre
   - Analysis of musical notes
     - pitch, timbre, envelope, etc.
   - Analysis of sentiments
   - Speech recognition 
     - Single speaker
     - Multiple speakers
     - Accents
   - Emotion recognition 
   - Distinction between speech and music 
   - Speech commands
   - Transcription

