# Language Modelling

### Q1. What are generative models of language, and how do they differ from discriminative models? [4]
**a) What are generative models of language? Explain any one model in detail.[4]**
**a) What are generative models of language, and how do they differ from discriminative models? Provide an example of a generative model and describe how it can be used in NLP. [9]**

### **Generative Model – Definition:**
- Generative models are a class of models in Natural Language Processing (NLP)
- It models the **joint probability distribution** $P(x, y)$ and can generate new data instances by learning how the data is distributed.
* $x$ = input data (e.g., features, observed data),
* $y$ = label or target output (e.g., class label, sentence, etc.).

#### **Key Points:**

* Models both input $x$ and output $y$ together.
* Can **generate new data** similar to the training set.
* Learns how data is **generated** in the real world.
* Example: **Naive Bayes**, **Hidden Markov Model (HMM)**, **GPT**.
* Used in **text generation**, **machine translation**, **speech synthesis**, etc.

---

### **Discriminative Model – Definition:**

A **discriminative model** models the **conditional probability distribution** $P(y|x)$ and focuses on predicting the correct output label $y$ for a given input $x$.

#### **Key Points:**

* Models the **boundary** between classes.
* Cannot generate new data, only classifies.
* Used for **classification** and **regression** tasks.
* Example: **Logistic Regression**, **Support Vector Machine (SVM)**, **BERT**.
* Used in **spam detection**, **sentiment analysis**, **named entity recognition**, etc.

---

**Difference between Generative and Discriminative Models:**

| **Point**                      | **Generative Model**                           | **Discriminative Model**                                 |      |
| ------------------------------ | ---------------------------------------------- | -------------------------------------------------------- | ---- |
| **Definition**                 | Models the **joint probability** $P(x, y)$     | Models the **conditional probability** ( P(y             | x) ) |
| **Objective**                  | Learns how the data is **generated**           | Learns to **classify** or **predict labels**             |      |
| **Functionality**              | Can **generate** new data samples              | Can only **predict** output labels                       |      |
| **Data Modeling**              | Models distribution of both input and output   | Models decision boundary between classes                 |      |
| **Complexity**                 | Usually more complex, models more information  | Usually simpler and focused on prediction                |      |
| **Examples**                   | Naive Bayes, Hidden Markov Model (HMM), GPT    | Logistic Regression, Support Vector Machine (SVM), BERT  |      |
| **Use Cases**                  | Text generation, speech synthesis, translation | Spam detection, sentiment analysis, image classification |      |
| **Accuracy in Classification** | May be less accurate for classification tasks  | Generally more accurate for classification               |      |
| **Data Requirement**           | Needs more data to model full distribution     | Requires less data compared to generative models         |      |


**Example of Generative Model:**

**GPT (Generative Pre-trained Transformer)**

* GPT is a **transformer-based** generative language model trained on massive text corpora.
* It can **generate text**, complete sentences, answer questions, translate languages, or even write code.

**Use in NLP:**

* **Text Generation:** Generate human-like text for dialogue systems, story writing, content creation.
* **Summarization:** Summarize long articles.
* **Translation:** Translate text from one language to another.
* **Question Answering:** Answer questions by generating relevant responses.

### Q2.  Describe the process of building a simple Markov model for predicting the next word in a sentence with the help of example. [6]


### **Markov Model – Explanation:**

- A Markov model predicts the next state (in our case, the next word) based only on the current state (the current word). 
- It assumes the **Markov property**:
> *The future state depends only on the present state and not on the sequence of events that preceded it.*

### **Types of Markov Models:**

* **Unigram Model:** Assumes each word is independent.
* **Bigram Model:** Assumes the current word depends only on the previous word.
* **Trigram Model:** Assumes the current word depends on the previous two words.

---

### **Building a Simple Markov Model (Bigram) – Step-by-Step:**
1. **Prepare Your Training Data**: You need a corpus of text (a collection of sentences) to train your model. 
2. **Tokenization**: Break down your training data into individual words (tokens). This usually involves splitting sentences by spaces and punctuation.
3. **Create the Transition Probabilities**: This is the heart of the Markov model. You need to calculate the probability of each word following another word in your training data.
4. **Represent the Model**: You can represent these transition probabilities in a data structure, such as a dictionary or a matrix.
5. **Prediction**: To predict the next word, given a current word:
      * Look up the current word in your model.
      * Find the probabilities of all possible next words.
      * You can then choose the word with the highest probability as your prediction

#### **Step 1: Prepare Training Data**

Example corpus:

```
"I love NLP", "I love coding", "NLP is fun"
```
### **Step 2: Tokenization**

["I", "love", "NLP"]
["I", "love", "coding"]
["NLP", "is", "fun"]

#### **Step 3: Transition Probabilities**

$$
P(\text{word}_2 \mid \text{word}_1) = \frac{\text{Count}(\text{word}_1\ \text{word}_2)}{\text{Count}(\text{word}_1)}
$$

P("love" | "i") = Count("i love") / Count("i") = 2 / 2 = 1.0
P("nlp" | "love") = Count("love nlp") / Count("love") = 1 / 2 = 0.5
P("coding" | "love") = Count("love coding") / Count("love") = 1 / 2 = 0.5
P("is" | "nlp") = Count("nlp is") / Count("nlp") = 1 / 1 = 1.0
P("fun" | "is") = Count("is fun") / Count("is") = 1 / 1 = 1.0

#### **Step 4: Represent the Model**

simple_nlp_model = {
    "i": {"love": 1.0},
    "love": {"nlp": 0.5, "coding": 0.5},
    "nlp": {"is": 1.0},
    "is": {"fun": 1.0}
}

#### **Step 5: Prediction**
* If the current word is "i": The model would predict "love" with a 100% probability.
* If the current word is "love": The model would predict "nlp" with a 50% probability and "coding" with a 50% probability.
* If the current word is "nlp": The model would predict "is" with a 100% probability.
* If the current word is "is": The model would predict "fun" with a 100% probability.

---


### Q3.Explain Probabilistic Language Modeling

**Definition:**

A **Probabilistic Language Model** assigns a **probability to a sequence of words**, estimating how likely that sentence is to occur in a language. It helps machines understand, generate, or predict text in natural language processing (NLP).
- Markov Models are a type of Probabilistic Language Model. But not all probabilistic models are Markov models.

| Aspect         | **Markov Model**                                                                                       | **Probabilistic Language Model**                                        |
| -------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| **Definition** | Assumes the next state (word) depends only on a limited number of previous states (Markov assumption). | Assigns probability to a sequence of words using statistical methods.   |
| **Type**       | A **subset** of probabilistic models with a specific memory limitation (n-gram).                       | A **broad category** that includes Markov models, neural models, etc.   |
| **Examples**   | Bigram, Trigram models                                                                                 | Markov models, Hidden Markov Models, Neural Language Models (e.g., GPT) |

---

### **Key Concepts:**

* It uses **statistical methods** to compute the probability of a word given the previous word(s).

* The probability of a sentence $P(w_1, w_2, ..., w_n)$ is calculated using the **chain rule**:

  $$
  P(w_1, w_2, ..., w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1, w_2) \cdots P(w_n|w_1,...,w_{n-1})
  $$

* Due to data sparsity, we simplify it using **n-gram models**:

  * **Unigram:** $P(w_i)$
  * **Bigram:** $P(w_i|w_{i-1})$
  * **Trigram:** $P(w_i|w_{i-2}, w_{i-1})$

---

### **Applications:**

* **Speech recognition**
* **Machine translation**
* **Text generation**
* **Autocomplete / predictive text**


---

### Q4. Consider the following small corpus: [8]
* Training corpus:
```cpp
<s> I am from Pune </s>
<s> I am a teacher </s>
<s> students are good and are from various cities </s>
<s> students from Pune do engineering </s>
```
* Test data:
`<s> students are from Pune </s>`
Find the Bigram probability of the given test sentence.

To calculate the **Bigram probability** of the test sentence:

### **Step 1: Training Corpus**

Let's extract all bigrams from the training data. Each sentence is surrounded by special tokens `<s>` (start) and `</s>` (end):

1. `<s> I am from Pune </s>`
   Bigrams: (`<s>`, I), (I, am), (am, from), (from, Pune), (Pune, `</s>`)

2. `<s> I am a teacher </s>`
   Bigrams: (`<s>`, I), (I, am), (am, a), (a, teacher), (teacher, `</s>`)

3. `<s> students are good and are from various cities </s>`
   Bigrams: (`<s>`, students), (students, are), (are, good), (good, and), (and, are), (are, from), (from, various), (various, cities), (cities, `</s>`)

4. `<s> students from Pune do engineering </s>`
   Bigrams: (`<s>`, students), (students, from), (from, Pune), (Pune, do), (do, engineering), (engineering, `</s>`)

---

### **Step 2: Count Bigram Frequencies**

| Bigram                | Count |
| --------------------- | ----- |
| (`<s>`, I)            | 2     |
| (I, am)               | 2     |
| (am, from)            | 1     |
| (from, Pune)          | 2     |
| (Pune, `</s>`)        | 1     |
| (am, a)               | 1     |
| (a, teacher)          | 1     |
| (teacher, `</s>`)     | 1     |
| (`<s>`, students)     | 2     |
| (students, are)       | 1     |
| (are, good)           | 1     |
| (good, and)           | 1     |
| (and, are)            | 1     |
| (are, from)           | 1     |
| (from, various)       | 1     |
| (various, cities)     | 1     |
| (cities, `</s>`)      | 1     |
| (students, from)      | 1     |
| (Pune, do)            | 1     |
| (do, engineering)     | 1     |
| (engineering, `</s>`) | 1     |

Also collect **unigram counts** (for denominator):

| Word        | Count |
| ----------- | ----- |
| `<s>`       | 4     |
| I           | 2     |
| am          | 2     |
| from        | 3     |
| Pune        | 2     |
| `</s>`      | 4     |
| a           | 1     |
| teacher     | 1     |
| students    | 2     |
| are         | 2     |
| good        | 1     |
| and         | 1     |
| various     | 1     |
| cities      | 1     |
| do          | 1     |
| engineering | 1     |

---

### **Step 3: Test Sentence**

`<s> students are from Pune </s>`

Bigrams:

1. (`<s>`, students)
2. (students, are)
3. (are, from)
4. (from, Pune)
5. (Pune, `</s>`)

---

### **Step 4: Bigram Probabilities (using MLE)**

Formula:

$$
P(w_n | w_{n-1}) = \frac{\text{Count}(w_{n-1}, w_n)}{\text{Count}(w_{n-1})}
$$

1. $P(students | <s>) = \frac{2}{4} = 0.5$
2. $P(are | students) = \frac{1}{2} = 0.5$
3. $P(from | are) = \frac{1}{2} = 0.5$
4. $P(Pune | from) = \frac{2}{3} \approx 0.6667$
5. $P(</s> | Pune) = \frac{1}{2} = 0.5$

---

### **Step 5: Final Bigram Probability**

Multiply all probabilities:

$$
P = 0.5 \times 0.5 \times 0.5 \times 0.6667 \times 0.5 = 0.0417
$$

---

### ✅ **Final Answer:**

**Bigram probability = 0.0417 (approx)**

---

### Q5. Suppose you have a text corpus of 10,000 words, and you want to build a bigram model from this corpus. The vocabulary size of the corpus is 5,000. After counting the bigrams in the corpus, you found that the bigram “the cat” appears 50 times, while the unigram “the” appears 1000 times and the unigram “cat” appears 100 times. Using the add-k smoothing method with k=0.5, what is the probability of the sentence “the cat sat on the mat”? 

- Add-k smoothing is a technique used in probabilistic language models to handle zero-frequency problems by adding a small constant 𝑘 to all n-gram counts.
- It ensures that every possible word sequence has a non-zero probability, even if it never appeared in the training data.

### ✅ **Add-k Smoothing Method (Laplace Smoothing – Generalized)**

**Definition:**

Add-k smoothing is a technique used in **n-gram language models** to handle **zero-frequency** problems by adding a small constant $k$ to each count.

---

### ✅ **Formula (Bigram Model with Add-k Smoothing):**

For bigram $P(w_i | w_{i-1})$:

$$
P(w_i | w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i) + k}{\text{Count}(w_{i-1}) + k \cdot V}
$$

Where:

* $k$ = smoothing constant
* $V$ = vocabulary size
* $\text{Count}(w_{i-1}, w_i)$ = bigram count
* $\text{Count}(w_{i-1})$ = unigram count of previous word

---

### ✅ **Given:**

* Corpus size = 10,000 words (not directly used)
* Vocabulary size $V = 5000$
* Bigram count $\text{Count}(\text{"the cat"}) = 50$
* Unigram count $\text{Count}(\text{"the"}) = 1000$
* $k = 0.5$


Tokenized with start:
`<s> the cat sat on the mat </s>`

**Bigrams:**

* (`<s>`, the)
* (the, cat)
* (cat, sat)
* (sat, on)
* (on, the)
* (the, mat)
* (mat, `</s>`)

We will assume all **bigram and unigram counts are 0** unless specified (only “the cat” and “the” are given).


1. **General formula**

   $$
   P(w_i\,|\,w_{i-1}) \;=\; \frac{\mathrm{Count}(w_{i-1},w_i) + k}{\mathrm{Count}(w_{i-1}) + k\cdot V}
   $$

2. **Given counts**

   * Count(“the cat”) = 50
   * Count(“the”) = 1000
   * Count(“cat”) = 100
   * All other bigrams/unigrams (“sat”, “on”, “mat” as previous words) have count = 0.

3. **Denominators**

   * For “the”: 1000 + 0.5×5000 = 1000 + 2500 = 3500
   * For “cat”: 100  + 2500           = 2600
   * For “sat”, “on”, “mat”: 0 + 2500 = 2500 each
   * Count of all unseen unigrams = 0 → denominator = $0 + 0.5 \cdot 5000 = 2500$
   * Count of unseen bigrams = 0 → numerator = 0.5

4. **Bigram probabilities**

   $$
   \begin{aligned}
   P(\text{cat}\mid\text{the}) &= \frac{50 + 0.5}{3500} \;\approx\; 0.01443,\\
   P(\text{sat}\mid\text{cat}) &= \frac{0.5}{2600}      \;\approx\; 0.0001923,\\
   P(\text{on}\mid\text{sat})  &= \frac{0.5}{2500}      \;=\;    0.0002,\\
   P(\text{the}\mid\text{on})  &= \frac{0.5}{2500}      \;=\;    0.0002,\\
   P(\text{mat}\mid\text{the}) &= \frac{0.5}{3500}      \;\approx\; 0.0001429.
   \end{aligned}
   $$

5. **Sentence probability**

   $$
   P = \prod P(w_i\mid w_{i-1})
     = 0.01443 \times 0.0001923 \times 0.0002 \times 0.0002 \times 0.0001429
     \;\approx\; 1.59 \times 10^{-17}.
   $$

---

**Answer:**

$$
P(\text{“the cat sat on the mat”}) \approx 1.6 \times 10^{-17}.
$$

---

### Q6. Explain in detail Latent Semantic Analysis for topic modelling (LSA)
 Write a short note on Latent Semantic Analysis (LSA). [4]


### 🔷 What is Latent Semantic Analysis (LSA)?

Latent Semantic Analysis (LSA), also known as **Latent Semantic Indexing (LSI)** when used in information retrieval, is an **unsupervised NLP technique** used to discover **hidden (latent) relationships between words and documents**.

It is based on the idea that **words that are used in similar contexts tend to have similar meanings**.

---

### 🔶 Purpose in NLP:

* **Dimensionality reduction**
* **Capturing synonymy and polysemy**
* **Topic extraction from documents**

---

### 🔷 Steps Involved in LSA for Topic Modelling:

#### ✅ 1. Text Preprocessing

* Lowercasing
* Tokenization
* Removing stop words
* (Optional) Lemmatization or stemming

#### ✅ 2. Create Term-Document Matrix (TDM)

A matrix is created where:

* Rows = terms (words)
* Columns = documents
* Values = frequency or TF-IDF scores

Example:

| Term / Doc | Doc1 | Doc2 | Doc3 |
| ---------- | ---- | ---- | ---- |
| data       | 3    | 0    | 2    |
| mining     | 2    | 1    | 0    |
| science    | 0    | 2    | 3    |

#### ✅ 3. Apply TF-IDF (Optional but recommended)

This is used to give more weight to terms that are more important in a document.

* **Term Frequency (TF)** – importance in document
* **Inverse Document Frequency (IDF)** – uniqueness across corpus

#### ✅ 4. Apply **Singular Value Decomposition (SVD)**

SVD allows dimensionality reduction by capturing the most important aspects of the data.
SVD factorizes the matrix into three matrices:

$$
A \approx U \cdot \Sigma \cdot V^T
$$

Where:

* **U**: term-topic matrix
* **Σ**: singular values (importance of each topic)
* **Vᵗ**: topic-document matrix

You can reduce dimensionality by **keeping only top-k singular values**, capturing major topics.

#### ✅ 5. Topic Extraction


From the decomposed matrix:

* Rows of **U** tell us which words are associated with which **topics**
* Rows of **Vᵗ** tell us which documents are associated with which **topics**

---

### 🔷 Simple Example

Let’s say you have 3 documents:

1. “Cats are small animals.”
2. “Dogs are friendly animals.”
3. “Cats and dogs are pets.”

LSA might identify:

* Topic 1: {“cats”, “animals”, “pets”}
* Topic 2: {“dogs”, “friendly”, “pets”}

Even though “pets” may not co-occur with “friendly” directly, the latent structure reveals this link.

---

### 🔶 Advantages of LSA:

* Captures **semantic meaning** via context
* Handles **synonymy and polysemy**
* Simple to implement with SVD libraries

---

### 🔶 Limitations:

* Assumes linear relationships (no non-linearity)
* Not probabilistic like LDA
* Poor scalability for large corpora


---

### Q7. Given a document-term matrix with the following counts: [6]

```
Document 1 Document 2 Document 3
Term 1 10 5 0
Term 2 2 0 8
Term 3 1 3 6
Calculate the TF-IDF score of “Term 1” in “Document 1”.
```
Here is a complete **theory + example explanation** of **TF-IDF** and how to compute it, ideal for your **theory exam**:

---

### 📌 **What is TF-IDF?**

**TF-IDF (Term Frequency–Inverse Document Frequency)** is a statistical measure used to evaluate how important a word is to a document in a collection (or corpus).

* It reflects the **importance of a term** based on how frequently it appears in a **specific document** (TF) and how **rare** it is across the entire corpus (IDF).
* The goal is to **highlight important and unique words** in a document and **down-weight common words** like “the”, “is”, “and”, etc.

---

### 📘 **TF-IDF Formula**

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

Where:

* $t$ = term (word)
* $d$ = document
* $\text{TF}(t, d)$ = Term Frequency of term *t* in document *d*
* $\text{IDF}(t)$ = Inverse Document Frequency of term *t*

---

### 🔹 **Step-by-Step Calculation**

#### 1. **Term Frequency (TF)**

It measures how frequently a term appears in a document.

$$
\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
$$

#### 2. **Inverse Document Frequency (IDF)**

It measures how rare a term is across all documents.

$$
\text{IDF}(t) = \log\left(\frac{N}{df(t)}\right)
$$

Where:

* $N$ = total number of documents
* $df(t)$ = number of documents containing term *t*

---

### 🧮 **Example Calculation**

#### 🔸 Given Document-Term Matrix:

|        | Document 1 | Document 2 | Document 3 |
| ------ | ---------- | ---------- | ---------- |
| Term 1 | 10         | 5          | 0          |
| Term 2 | 2          | 0          | 8          |
| Term 3 | 1          | 3          | 6          |

👉 **Find TF-IDF score of “Term 1” in “Document 1”**

---

#### 🔹 Step 1: Compute TF of Term 1 in Document 1

$$
\text{TF}(\text{Term 1}, \text{Doc 1}) = \frac{10}{10+2+1} = \frac{10}{13} ≈ 0.7692
$$

---

#### 🔹 Step 2: Compute IDF of Term 1

* Total documents $N = 3$
* Term 1 appears in Document 1 and 2 → $df = 2$

$$
\text{IDF}(\text{Term 1}) = \log\left(\frac{3}{2}\right) ≈ 0.4055
$$

---

#### 🔹 Step 3: Compute TF-IDF

$$
\text{TF-IDF} = 0.7692 × 0.4055 ≈ \boxed{0.3119}
$$

---

### ✅ **Interpretation**

* A **higher TF-IDF score** means the term is **important to that document** but **not too common in other documents**.
* Here, **“Term 1” is highly relevant to Document 1**, as seen from the high TF and moderate IDF.

---

### Q8. Define Latent Dirichlet Allocation (LDA) and explain how it is used for topic modeling in text data. Discuss the key components of LDA, including topics, documents, and word distributions. [9]

### 🔷 Latent Dirichlet Allocation (LDA) – Explained in Detail

**Latent Dirichlet Allocation (LDA)** is a **generative probabilistic model** used for **topic modeling** — the task of discovering the abstract "topics" that occur in a collection of documents.

---

### 🔶 Key Concepts

* **Document**: A mixture of topics.
* **Topic**: A distribution over words.
* **Word**: An observed word in a document.
* **Latent**: Hidden structures (topics) are inferred from the data.
* **Dirichlet**: A type of distribution used to represent probabilities over probabilities.

---

### 🔷 How LDA Works (Step-by-Step Intuition)

1. **Assumptions**:

   * Each document is composed of a mix of topics.
   * Each topic is a probability distribution over words.

2. **LDA Process** (Generative process):

   * Choose a distribution over topics for each document (θ).
   * For each word in the document:

     * Choose a topic `z` from the topic distribution.
     * Choose a word `w` from the selected topic’s word distribution.

3. **Goal**: Reverse-engineer this process using observed words and infer:

   * The topics in the corpus.
   * Topic distribution per document.
   * Word distribution per topic.

---

### 🔶 LDA Visualization

Imagine a document as a **smoothie**, and each **fruit** in the smoothie is a word.
The **recipe** (topic distribution) tells you what fruits (topics) are mixed in what proportions.

---

### 🔷 Example

#### Corpus (3 documents):

1. “I love to eat broccoli and bananas.”
2. “I ate a banana and spinach smoothie.”
3. “Broccoli is rich in nutrients.”

#### LDA Output (assuming 2 topics):

* **Topic 1** (Health): broccoli, spinach, nutrients
* **Topic 2** (Food): banana, eat, smoothie

Now each document is tagged with topic probabilities:

* Doc1: 40% Topic1, 60% Topic2
* Doc2: 20% Topic1, 80% Topic2
* Doc3: 90% Topic1, 10% Topic2

---

### 🔷 Applications of LDA

* **Document Classification**
* **Recommender Systems**
* **Search Engine Optimization**
* **Customer Feedback Analysis**
* **News Article Clustering**

---

### 🔶 Advantages of LDA

* **Unsupervised**: No labeled data required.
* **Interpretable**: Topics are human-readable.
* **Flexible**: Can be used for a variety of document collections.

---

### ❌ Disadvantages

* Needs to predefine the number of topics (K).
* May not capture complex semantic relationships.
* Assumes "bag-of-words" — ignores word order.

---



### Q9. What is BERT,Describe the concept of contextualized representations, such as those generated by BERT, and how they are used in natual language processing. Discuss the advantages and disadvantages of contextualized representations. [10]
(Write short note on . [4])



## ✅ **What is BERT?**

**BERT (Bidirectional Encoder Representations from Transformers)** is a **pre-trained language model** developed by Google in 2018. It is based on the **Transformer architecture** and is designed to understand the **context of a word in a sentence by looking at both its left and right surroundings (bidirectional).**

---

## 📘 **Contextualized Representations:**

In traditional word embeddings (like Word2Vec), each word has **one fixed vector**, regardless of context.
But in **contextualized representations**, the meaning of a word changes depending on its surrounding words.

🔹 Example:

* "bank" in "river bank" vs. "bank" in "open a bank account"
* BERT gives **different vectors** for each, based on context.

---

## 🔍 **How BERT Generates Contextual Representations:**

1. **Uses Transformer Architecture** – BERT is built on the Transformer encoder.
2. **Bidirectional Attention** – It looks at **both left and right words** in a sentence at the same time.
3. **Word Meaning Depends on Context** – Each word is represented **differently** depending on the surrounding words.
4. BERT is pre-trained on two tasks:

   * **Masked Language Modeling (MLM)** – Predict missing words.
   * **Next Sentence Prediction (NSP)** – Understand sentence relationships.
5. **Outputs Contextual Vectors** – For each word, BERT gives a **unique vector** that captures its meaning in that specific sentence.


## 🛠️ **Use and **Advantages of Contextualized Representations:****

* **Text classification** (e.g., spam detection, sentiment)
* **Named Entity Recognition (NER)** ( e.g., identifying people, places, organizations)
* **Question Answering systems** (e.g., understanding questions and providing answers)
* **Machine Translation**
* **Text summarization**
* **Language generation** (e.g., chatbots, text completion)
* **Captures polysemy**: Same word gets different meanings in different contexts.

## ❌ **Disadvantages:**

1. **Computationally expensive** (large memory and GPU usage).
2. **Slower inference** compared to simpler models.
3. **Large model size**, hard to deploy on low-resource devices.
4. **Opaque**: Interpretability of deep models like BERT is still a challenge.

---

