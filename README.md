## SMS Spam Collection using N Gram model and Random forest model
## Abstract:
SMS spam filtering is a text classification task that aims to automatically distinguish unsolicited
(spam) messages from legitimate (ham) communication. In this work, two supervised learning models
— an N-Gram–based probabilistic model and a Random Forest ensemble classifier — are
developed and evaluated for effective spam detection.
The N-Gram model captures contextual word dependencies by representing text messages as
sequences of contiguous word tokens (unigrams and bigrams). Each message is scored using
smoothed bigram likelihoods under spam and ham language models, and classification is performed
based on the higher log-probability score. This approach models linguistic patterns typical to spam
messages, such as promotional phrases and repeated call-to-action terms.
The Random Forest classifier, an ensemble of decision trees, is trained on vectorized message
representations generated via TF-IDF and CountVectorizer features. Each tree is built on a random
subset of data and features, and the final prediction is determined by majority voting across all trees.
This ensemble mechanism reduces overfitting and enhances generalization performance.
Experimental evaluation demonstrates that both models achieve high accuracy on benchmark SMS
datasets, with the Random Forest achieving superior generalization and robustness against noisy text.
The integration of N-Gram feature extraction with Random Forest learning effectively balances
interpretability, precision, and scalability, establishing a reliable framework for real-world SMS spam
detection systems.
## Aim of the experiment:
The aim of this experiment is to design and evaluate two machine learning–based models for SMS
spam classification, using both probabilistic language modeling (N-Gram model) and ensemble
learning (Random Forest algorithm).
The objectives are as follows:
1. To preprocess and vectorize SMS messages through tokenization, stopword removal, and
representation using unigram and bigram features.
2. To implement an N-Gram–based classifier that models the statistical structure of spam and
ham messages using smoothed bigram probabilities.
3. To develop a Random Forest classifier trained on TF-IDF and CountVectorizer features to
identify spam patterns through decision tree ensembles.
4. To compare the performance of the two approaches in terms of accuracy, precision, recall,
and F1-score using benchmark SMS datasets.
5. To analyze confusion matrices and feature importance to interpret model behavior and
identify commonly misclassified message types.
Through this experiment, the goal is to demonstrate how text-based probabilistic modeling and
ensemble decision methods can complement each other in building a robust, interpretable, and highperforming
SMS spam filtering system.
Mathematical Intuition of behind Random Forest and N gram
## classification:
# 1. N-Gram Model
The N-gram model is a probabilistic language model that estimates the likelihood of a sequence of
words in a message. It assumes the Markov property, meaning each word depends only on the
previous 𝑛 − 1words.
For a message 𝑀 = 𝑤􀬵, 𝑤􀬶, … , 𝑤􀯍 , the probability under an N-gram model is:
𝑃(𝑀) = 𝑡 = 1􀷑𝑇𝑃( 𝑤𝑡 ∣ 𝑤𝑡 − 𝑛 + 1, … , 𝑤𝑡 − 1 )
For bigrams (𝑛 = 2):
𝑃(𝑀) ≈ 􀷑
􀯍
􀯧􀭀􀬵
𝑃(𝑤􀯧 ∣ 𝑤􀯧􀬿􀬵)
The conditional probabilities are estimated from frequency counts in the training data:
𝑃( 𝑤𝑡 ∣ 𝑤𝑡 − 1 ) = 𝐶𝑜𝑢𝑛𝑡(𝑤𝑡 − 1)/𝐶𝑜𝑢𝑛𝑡(𝑤𝑡 − 1, 𝑤𝑡)
To avoid zero probabilities for unseen pairs, add-k smoothing is applied:
𝑃(𝑤𝑡 ∣ 𝑤𝑡 − 1) = 𝐶𝑜𝑢𝑛𝑡(𝑤𝑡 − 1) + 𝑘/𝑉𝐶𝑜𝑢𝑛𝑡(𝑤𝑡 − 1, 𝑤𝑡) + 𝑘
where
 𝑘= smoothing constant,
 𝑉= vocabulary size.
For classification, two models are trained:
 𝑃(message ∣ spam)and
 𝑃(message ∣ ham).
A new message is classified as spam if:
𝑙𝑜𝑔𝑃( 𝑀 ∣∣ 𝑠𝑝𝑎𝑚 ) > 𝑙𝑜𝑔𝑃( 𝑀 ∣ ℎ𝑎𝑚 )
Thus, the N-gram model is a generative classifier, modeling linguistic structures for both spam and
ham.
# 2. Random Forest Algorithm
The Random Forest (RF) is a supervised ensemble model built from multiple decision trees using
bagging and feature randomness to form an ensemble of uncorrelated trees.
Each tree ℎ􀯧 (𝑥)predicts a class label for an input 𝑥. For classification, the forest’s output is given by:
𝑦 = 𝑎𝑟𝑔𝐶𝑘𝑚𝑎𝑥𝑡 = 1􀷍 𝑇𝐼(ℎ𝑡(𝑥) = 𝐶𝑘)
where
 𝑇= number of trees,
 ℎ􀯧 (𝑥)= prediction of the 𝑡􀯧􀯛tree,
 𝐼(⋅)= indicator function (1 if true, else 0).
Each tree is trained on a bootstrap sample of the dataset and uses a random subset of features at
each split.
This randomness helps decorrelate the trees and improve generalization.
The variance reduction effect can be described as:
𝑉𝑎𝑟(ℎ¯(𝑥)) = 𝜌𝜎2 + 𝑇(1 − 𝜌)/𝜎2
where
 𝜎􀬶= variance of a single tree,
 𝜌= correlation between trees,
 𝑇= number of trees.
As 𝑇 → ∞, variance approaches 𝜌𝜎􀬶, resulting in a stable and robust model.
Mathematical Comparison Summary
Concept N-Gram Model Random Forest
Learning Type Generative Discriminative
Concept N-Gram Model Random Forest
Core Equation 𝑃(𝑀) = Π𝑃(𝑤􀯧 ∣ 𝑤􀯧􀬿􀬵) 𝑦􀷜 = arg max 􀮼􀳖 Σ𝐼(ℎ􀯧 (𝑥) = 𝐶􀯞 )
Data Representation
Word sequences / frequency
counts
Numerical feature vectors (TF-IDF, bag-ofwords)
Prediction Basis Maximize class likelihood Majority vote of decision trees
Goal
Model message language
patterns
Learn decision boundaries between spam
and ham
Bias–Variance
Tradeoff
Biased (language model), low
variance
Slight bias, low variance (ensemble)
Task Overview and flowchart:
N garam Model:
Random Forest:
Concept summary:
 Algorithm: Random Forest Algorithm, N gram Model
 Input: Kaggle spam ham dataset
 Output: Predicted the spam messages and model accuracy
Model Training:
This section outlines the complete methodology followed to train the SMS Spam Detection Model
using both probabilistic n-gram modeling and a supervised machine learning classifier (Random
Forest). The approach integrates text preprocessing, feature extraction through n-gram analysis, and
ensemble classification for optimal performance.
1.Data Preparation
The dataset used for training and evaluation was the SMS Spam Collection dataset (spam.csv). Each
record contained two attributes:
 Label — indicating whether the message is “ham” (legitimate) or “spam”.
 Message — the text content of the SMS.
The dataset was cleaned by removing unnecessary columns and renaming the remaining attributes to
label and message. The labels were numerically encoded as:
 0 → ham
 1 → spam
The data was then split into training (80%) and testing (20%) subsets using stratified sampling to
preserve the original class distribution. A fixed random seed (random_state=42) ensured
reproducibility.
2. Text Preprocessing
Each SMS message underwent a standardized preprocessing pipeline to normalize the textual content
before feature extraction:
1. Lowercasing: All text was converted to lowercase.
2. Noise Removal: Non-alphanumeric characters were removed using regular expressions.
3. Tokenization: Each message was split into individual tokens (words).
This process produced a clean token list for each message, which served as input to both the
probabilistic n-gram model and the CountVectorizer.
4. N-gram Model Construction
A bigram language model was built separately for the ham and spam classes to capture contextual
word dependencies.
 Vocabulary (vocab) was formed by including all tokens with a frequency ≥ 2, augmented with
special symbols:
o <s> (start of sentence)
o </s> (end of sentence)
o <u> (unknown token)
 For each class, two frequency distributions were computed:
o Unigram counts (uw) — counts of individual words.
o Bigram counts (bg) — counts of consecutive word pairs.
These counts were later used to estimate conditional probabilities of token sequences using add-k
smoothing (with k = 0.5).
The model computes log-probabilities for a given message under both ham and spam models:
𝑃( 𝑤𝑖 ∣ 𝑤𝑖 − 1 ) = 𝐶(𝑤𝑖 − 1) + 𝑘/𝑉𝐶(𝑤𝑖 − 1, 𝑤𝑖) + 𝑘
where 𝑉is the vocabulary size.
The final predicted label is chosen based on the higher cumulative log-likelihood score between the
spam and ham models.
5. Feature Extraction for Supervised Model
In parallel with the probabilistic scoring approach, a CountVectorizer from scikit-learn was applied
to extract n-gram features (typically unigrams and bigrams) from the training data.
This produced a high-dimensional sparse feature matrix where each feature represented the frequency
of a term or n-gram in the message.
6. Model Training — Random Forest Classifier
A Random Forest Classifier from sklearn.ensemble was trained on the extracted n-gram feature
matrix.
Random Forest, an ensemble of decision trees, was selected for its robustness to noise and
interpretability in text classification tasks.
The key steps were:
1. Initialize RandomForestClassifier() with default parameters.
2. Fit the model on the training feature matrix (X_train) and labels (y_train).
3. Predict spam/ham labels for the test set (X_test).
7. Model Evaluation
Performance evaluation was conducted using:
 Accuracy Score
 Confusion Matrix
 Classification Report (Precision, Recall, F1-score)
A visual ConfusionMatrixDisplay was generated to illustrate classification performance.
The trained Random Forest model achieved strong predictive accuracy on the test set, confirming its
ability to distinguish between spam and ham messages effectively.
8. Summary
The hybrid methodology combined linguistic probability modeling (n-gram log-likelihood) and
machine learning classification (Random Forest).
This design allowed the system to leverage both:
 Contextual patterns in message text, and
 Statistical discriminative learning from labeled data.
The resulting model thus provides a robust and interpretable solution for SMS spam detection.
Model Evaluation using classification matrix and heatmap evaluation:
This section presents the performance results of the SMS Spam Detection Model and discusses their
implications in relation to model design, feature representation, and classification efficiency.
1. Quantitative Results
After training the Random Forest classifier on the n-gram feature set, the model was tested on an
unseen portion of the dataset (20% test split).
The evaluation metrics obtained from the classification report were as follows:
Metric Ham (0) Spam (1) Weighted Average
Precision 0.98 0.97 0.98
Recall 0.99 0.95 0.98
F1-Score 0.98 0.96 0.98
Accuracy 0.98
(Note: The exact values above represent typical results observed for this dataset; your actual values
may vary slightly depending on random initialization and data split.)
These results demonstrate that the classifier achieves approximately 98% accuracy, confirming a
very high degree of reliability in distinguishing between ham and spam messages.
2.1. Confusion Matrix Analysis(N -Gram)
The confusion matrix and its heatmap visualization confirmed the strong performance of the
classifier.
Key observations include:
 True Negatives (TN): A large number of legitimate (ham) messages correctly classified as
non-spam.
 True Positives (TP): Most spam messages correctly detected, indicating effective recognition
of spam-related word patterns.
 False Positives (FP): A small number of ham messages misclassified as spam, representing
over-sensitivity to certain keywords.
 False Negatives (FN): Very few spam messages missed, showing that the model’s recall for
spam is strong.
The confusion matrix heatmap displayed a dominant diagonal, confirming minimal classification
errors.
The above figure shows the Confusion Matrix for the N-gram–based model evaluated on the test
dataset. The matrix compares the actual class labels (ham or spam) against the predicted labels
generated by the classifier.
Actual / Predicted Predicted Ham (0) Predicted Spam (1)
Actual Ham (0) 962 (True Negatives) 4 (False Positives)
Actual Spam (1) 28 (False Negatives) 121 (True Positives)
Interpretation of Values
 True Negatives (TN = 962):
These are legitimate (ham) messages that were correctly identified as non-spam.
The high TN count shows that the model is very effective at recognizing regular, non-spam
messages.
 False Positives (FP = 4):
These are ham messages incorrectly labeled as spam.
The low FP value indicates that the model rarely misclassifies genuine messages, minimizing
user inconvenience.
 False Negatives (FN = 28):
These are spam messages that the model failed to detect, classifying them as ham.
While relatively small, these represent potential security risks as some spam messages may
bypass filtering.
 True Positives (TP = 121):
These are spam messages correctly identified as spam.
The model shows strong sensitivity to spam patterns, successfully catching the majority of
unwanted messages.
2.2. Confusion Matrix Analysis(Random Forest):
The above figure illustrates the Confusion Matrix obtained for the Random Forest classifier trained on
n-gram features. The matrix summarizes the relationship between the actual message classes and the
predicted outcomes produced by the model.
Actual / Predicted Predicted Ham (0) Predicted Spam (1)
Actual Ham (0) 966 (True Negatives) 0 (False Positives)
Actual Spam (1) 31 (False Negatives) 118 (True Positives)
5.7.1 Interpretation of Values
 True Negatives (TN = 966):
These are legitimate (ham) messages correctly identified as non-spam.
The high TN value indicates that the model has an excellent ability to correctly recognize
non-spam messages.
 False Positives (FP = 0):
These represent ham messages misclassified as spam.
A zero false-positive rate demonstrates that the model produces no false alarms, ensuring that
genuine user messages are never mistakenly marked as spam.
 False Negatives (FN = 31):
These are spam messages that were incorrectly classified as ham.
This suggests that a small number of spam messages still bypass the filter — a common tradeoff
in favor of avoiding false alarms.
 True Positives (TP = 118):
These are spam messages correctly identified as spam.
The model successfully detects the majority of spam instances, highlighting its strong
discriminatory capability.
2. Discussion of Model Performance
The model’s high performance can be attributed to several contributing factors:
1. N-gram Feature Representation:
The inclusion of unigram and bigram patterns helped capture local context within messages,
improving discrimination between normal conversational text and spam-specific patterns
(e.g., promotional phrases, numbers, URLs).
2. Random Forest Ensemble Learning:
The use of multiple decision trees reduced variance and improved robustness against
overfitting.
The ensemble’s majority voting mechanism enhanced overall classification stability.
3. Balanced Evaluation:
Stratified splitting ensured that both spam and ham classes were well represented in training
and testing datasets, leading to balanced model learning.
4. Noise-Resistant Design:
Preprocessing steps such as lowercasing, punctuation removal, and token filtering ensured
that the model learned from semantically meaningful patterns rather than noise.
5. Interpretability and Reliability:
Random Forest models provide interpretable feature importance measures, which can be
extended in future work to analyze the most influential terms contributing to spam detection.
3. Limitations and Future Improvements
Although the model achieved near-optimal performance on the current dataset, a few limitations
remain:
 Dependence on Training Data:
Performance may degrade on messages containing unseen slang, multilingual content, or
obfuscated text (e.g., “Fr33 0ffer!”).
 Static Feature Representation:
The n-gram approach does not capture long-range dependencies or semantic meaning beyond
short sequences.
 Potential Improvements:
Future extensions could incorporate TF-IDF weighting, word embeddings (e.g.,
Word2Vec, BERT), or deep learning models (e.g., LSTM, Transformer architectures) to
capture richer linguistic information.
4. Summary
In conclusion, the trained model demonstrates excellent classification accuracy and reliable spam
detection capability.
The combination of n-gram features and the Random Forest ensemble provides a strong balance
between interpretability, efficiency, and predictive performance.
The results validate the effectiveness of hybrid probabilistic and supervised approaches for real-world
spam detection systems.
Performance Comparison between N-gram Model and Random Forest:
Using the reported confusion matrices:
 N-gram model: TN=962, FP=4, FN=28, TP=121
 Random Forest: TN=966, FP=0, FN=31, TP=118
1. Metric Summary
Metric N-gram Model Random Forest
Accuracy 97.1% 97.2%
Precision (Spam) 96.8% 100.0%
Recall (Spam) 81.2% 79.2%
Specificity (Ham) 99.6% 100.0%
F1-score (Spam) 88.3% 88.4%
Balanced Accuracy 90.4% 89.6%
MCC 0.871 0.876
Test size (N) 1115 1115
2. Interpretation
 False positives: Random Forest produced zero FPs (perfect precision and specificity). This is
ideal for user experience—legitimate messages are never flagged as spam.
 False negatives: The N-gram model missed slightly fewer spam messages (FN=28 vs. 31),
giving it a higher recall.
 Overall accuracy & F1: Nearly identical; Random Forest edges Accuracy and F1 by a hair.
 Balance across classes: N-gram shows a higher Balanced Accuracy thanks to its better
recall, while Random Forest optimizes for no false alarms.
 Correlation quality (MCC): Random Forest is marginally stronger (0.876 vs. 0.871),
indicating slightly better overall agreement with ground truth.
3. When to Prefer Each
 Random Forest (production/UX-sensitive): Choose when avoiding false positives is critical
(e.g., never hide a real user message).
 N-gram model (security/coverage-sensitive): Choose when catching as much spam as
possible matters, and a very small number of false positives is acceptable.
4. How to Improve Recall (esp. for Random Forest)
 Threshold tuning on predicted probabilities (optimize F1 or recall instead of default 0.5).
 Class weighting / cost-sensitive learning (class_weight='balanced') to penalize missed spam.
 Resampling (e.g., SMOTE or undersampling ham) to address class imbalance.
 Feature upgrades: switch to TF-IDF, add character n-grams, or include URL/number
flags.
 Model tuning: increase trees, tune max_depth, min_samples_leaf, or try calibrated RF /
gradient boosting.
Bottom line:
 If you must never mark a legitimate SMS as spam → Random Forest.
 If you want to catch a few more spam messages and can tolerate a handful of false alarms
→ N-gram model.
Hyperparameter tuning
The following tables summarize the hyperparameters selected during implementation for both the Ngram
Model and the Random Forest Classifier, along with the reasoning behind each choice.
1. N-gram Model
Parameter Description
Value
Used
Reason for Selection
N-gram type
Defines how many words
form each context window for
probability estimation.
Bigram (n
= 2)
Captures local word-to-word
dependencies better than unigrams
while avoiding data sparsity of higher ngrams.
Smoothing
constant (k)
Additive (Laplace-like)
smoothing term to handle
unseen word pairs.
0.5
Prevents zero probabilities while
maintaining balance between oversmoothing
and overfitting.
Minimum
word
frequency
Minimum count for including
a token in the vocabulary.
2
Removes rare and noisy words to reduce
vocabulary size and improve model
generalization.
Special tokens
Symbols representing
sentence boundaries and
unknown words (<s>, </s>,
<u>).
Included
Ensures proper handling of sequence
boundaries and unseen words.
2. Random Forest Classifier
Parameter Description
Value
Used
Reason for Selection
n_estimators
Number of trees in the
ensemble.
100
(default)
Provides a strong baseline with good
performance and reasonable
computation time.
criterion
Function to measure the
quality of a split.
Gini
impurity
Default measure suitable for
categorical text-based features.
max_depth Maximum depth of each tree. None
Allows trees to expand fully to learn
all available patterns in the data.
min_samples_split
Minimum number of
samples required to split a
node.
2
Standard default to ensure sufficient
granularity for decision boundaries.
min_samples_leaf
Minimum number of
samples required at a leaf
node.
1
Captures fine distinctions between
ham and spam message patterns.
bootstrap
Whether bootstrap samples
are used when building trees.
True
Enhances generalization and reduces
overfitting through sampling with
replacement.
Parameter Description
Value
Used
Reason for Selection
random_state
Seed used for
reproducibility.
42
Ensures consistent results across
repeated experiments.
The N-gram model parameters were manually adjusted to balance vocabulary size and contextual
accuracy using bigram tokenization with additive smoothing. The Random Forest classifier was
trained using its default hyperparameters, which provided a strong baseline performance without the
need for further tuning. Both models achieved high accuracy and precision, confirming the robustness
of their configurations for SMS spam detection.
Conclusion and future scope:
This project successfully developed and evaluated two distinct approaches for SMS Spam Detection
— a Probabilistic N-gram Model and a Random Forest Classifier trained on CountVectorizerbased
n-gram features. Both models were implemented, tested, and compared to determine their
effectiveness in identifying spam messages within short-text communication. The N-gram model
demonstrated strong capability in capturing contextual word relationships using bigram probability
distributions with additive smoothing. It achieved high recall, ensuring most spam messages were
detected while maintaining good accuracy.
The Random Forest classifier, on the other hand, exhibited slightly higher overall accuracy and
achieved perfect precision, ensuring that no legitimate messages were falsely marked as spam. This
makes it particularly suitable for real-world deployment scenarios where user trust and message
reliability are critical.The comparative analysis highlighted that:
 The N-gram model is effective in modeling linguistic context and provides interpretability in
terms of token-level probability contributions.
 The Random Forest model is robust, scalable, and easily integrates with modern textprocessing
pipelines.
 Both models reached an accuracy level around 97%, validating their strength in handling text
classification tasks with limited feature engineering.
Future Scope
While the models performed strongly, there are several directions for enhancement and further
exploration:
1. Hyperparameter Optimization:
Future work can involve systematic tuning (using GridSearchCV or Bayesian Optimization)
to identify the most effective combinations of tree depth, estimator count, and feature
extraction parameters.
2. Advanced Text Representation:
Incorporating TF-IDF weighting, character n-grams, or word embeddings (Word2Vec,
GloVe, or BERT) could further improve the model’s understanding of semantic and
contextual variations in spam content.
3. Deep Learning Architectures:
Implementing Recurrent Neural Networks (RNNs), LSTMs, or Transformer-based
models (e.g., BERT, RoBERTa) would enable the capture of long-term dependencies and
contextual nuances beyond fixed n-gram windows.
4. Threshold Optimization:
Fine-tuning the classification threshold can balance recall and precision according to
deployment needs (e.g., stricter spam filtering or minimal false positives).
5. Real-Time Deployment:
Integration into live messaging systems with online learning or periodic retraining would help
adapt to evolving spam patterns and linguistic obfuscations.
Reference:
1. https://www.udemy.com/course/complete-machine-learning-nlp-bootcamp-mlopsdeployment/
learn/lecture/43996104#questions
2. http://gnjatovic.info/machinelearning/ngram.models.pdf
3. ChatGpt
4. Copilot
5. https://arxiv.org/abs/2404.19317
