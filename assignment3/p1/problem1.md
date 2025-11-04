1. Next-Word Prediction using MLP [5 marks]
In this question, you will extend the next-character prediction notebook (discussed in class) to a next-word prediction problem.That is you will create a MLP based text generator. You will train the model, visualize learned word embeddings, and finally deploy a Streamlit app for interactive text generation. It is recommended to refer to Andrej Karpathy’s blog post on the Effectiveness of RNNs.
You must complete this task for two datasets: one from Category I (Natural Language) and one from Category II (Structured/Domain Text).
1.1 Preprocessing and Vocabulary Construction [0.5 mark]
For text-based datasets, you can remove special characters except “full stop (.)” so that it can be used to split sentences. However, you cannot ignore special characters for other datasets like for C++ code. You will have to treat text between newlines as a statement. To remove special characters from a line, you can use the following code snippet:
import re
line = re.sub('[^a-zA-Z0-9 \.]', '', line)
It will remove everything except alphanumeric characters, space and full-stop.
Convert the text to lowercase and use unique words to create the vocabulary.
• Report:
	- Vocabulary size
	- 10 most frequent and 10 least frequent words
To create X, and y pairs for training, you can use a similar approach used for next-character prediction. For example:
![training data](./training-data.png)


You will get something like “. . . . . ---> to” whenever there is a paragraph change.
1.2 Model Design and Training [1 marks]
Build an MLP-based text generator with the following structure:
	- Embedding dimension: 32 or 64
	- Hidden layers: 1–2 (1024 neurons each)
	- Activation: ReLU or Tanh
	- Output: Softmax over vocabulary
Use Google Colab or Kaggle for training (use maximum 500-1000 epochs). Start the assignment early, as training takes time. 
 Report in notebook:
	- Training vs validation loss plot
	- Final validation loss/accuracy
	- Example predictions and commentary on learning behavior.
1.3 Embedding Visualization and Interpretation [1 mark]
Visualize the embeddings using t-SNE if using more than 2 dimensions or using a scatter plot if using 2 dimensions and write your observations. For visualizations, you may have to select words with relations like synonyms, antonyms, names and pronouns, verbs and adverbs, words with no relations, and so on. Discuss your observations on clustering patterns and semantic relationships.

1.4 Streamlit Application [1.5 marks]
Write a streamlit application that asks users for an input text, and it then predicts the next k words or lines. In the streamlit app, you should have controls for modifying context length, embedding dimension, activation function, random seed, etc. You can use any one of the datasets mentioned. Incorporate temperature control in your streamlit app to control the randomness of predicted words. Refer to this article. 
Think how you would handle the case where words provided by the user in the streamlit app are not in the vocabulary. There is no need to re-train the model based on the user input. Train two to three variants and accordingly give options to the user.
1.5 Comparative Analysis [1 mark]
• Compare your two trained models (Category I vs Category II):
	- Dataset size, vocabulary, context predictability
	- Model performance (loss curves, qualitative generations)
	- Embedding visualizations
• Summarize insights on how natural vs structured language differs in learnability.
Datasets:
Category I
- Paul Graham essays
- Wikipedia (English)
- Shakespeare
- Leo Tolstoy's War and Peace
- The Adventures of Sherlock Holmes, by Arthur Conan Doyle

Category II
- Maths texbook
- Python or C++ code (Linux Kernel Code)
- IITGN advisory generation
- IITGN website generation
- Generate sklearn docs 
- Notes generation
- Image generation (ascii art, 0-255)
- Music Generation
Something comparable in spirit but of your choice (do confirm with TA Neerja)

Submission Format: Share a GitHub repo with your training notebooks named “question<number>.ipynb”.  Include textual answers in the notebook itself. For this question, put the link to the streamlit app at the top of the notebook.