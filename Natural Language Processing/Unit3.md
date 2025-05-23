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

Markov Models are a type of Probabilistic Language Model.
But not all probabilistic models are Markov models.

### **Markov Model – Explanation:**

- A **Markov Model** is a statistical model that predicts future states based on the **current state only**,
- It assumes the **Markov property**:
> *The future state depends only on the present state and not on the sequence of events that preceded it.*
- In **Natural Language Processing (NLP)**, Markov models are used to model sequences of words or characters where the probability of a word depends on a fixed number of previous words.

---

### **Types of Markov Models:**

* **Unigram Model:** Assumes each word is independent.
* **Bigram Model:** Assumes the current word depends only on the previous word.
* **Trigram Model:** Assumes the current word depends on the previous two words.

---

### **Building a Simple Markov Model (Bigram) – Step-by-Step:**

#### **Step 1: Collect Corpus**

Example corpus:

```
"I love NLP", "I love coding", "NLP is fun"
```

#### **Step 2: Build Bigram Frequencies**

Break sentences into word pairs (bigrams):

* (I, love) → 2 times
* (love, NLP) → 1 time
* (love, coding) → 1 time
* (NLP, is) → 1 time
* (is, fun) → 1 time

#### **Step 3: Calculate Probabilities**

For word "I":

* $P(love|I) = \frac{2}{2} = 1.0$

For word "love":

* $P(NLP|love) = \frac{1}{2} = 0.5$
* $P(coding|love) = \frac{1}{2} = 0.5$

#### **Step 4: Predict Next Word**

If input is: **"I"** → most probable next word is **"love"**
If input is: **"love"** → next word could be **"NLP"** or **"coding"**

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
<s> I am from Pune </s>
<s> I am a teacher </s>
<s> students are good and are from various cities </s>
<s> students from Pune do engineering </s>

* Test data:
<s> students are from Pune </s>
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

### 🔹 **Definition**

**Latent Semantic Analysis (LSA)** is a mathematical and statistical technique used in Natural Language Processing (NLP) to discover hidden (latent) relationships between words and documents.
- It reduces high-dimensional data into a lower-dimensional representation using **Singular Value Decomposition (SVD)**.

---

### 🔹 **Purpose in Topic Modeling**

In topic modeling, LSA uncovers **underlying topics** from a large corpus of text by finding patterns in the usage of words across documents.

---

### 🔹 **Steps in LSA for Topic Modeling**

1. **Create a Term-Document Matrix (TDM)**

   * Rows: words/terms
   * Columns: documents
   * Cells: frequency of the word in a document (can be TF or TF-IDF)

   Example:

   ```
          D1  D2  D3
   word1   1   0   3
   word2   2   1   0
   word3   0   2   1
   ```

2. **Apply TF-IDF (optional but improves results)**
   This scales down frequent but less informative words and highlights important ones.

3. **Interpret Topics**
   Each column in the reduced matrix corresponds to a **topic**, which is a set of terms with high weights.

### 🔹 **Applications**

* Topic detection in large text corpora
* Document clustering and classification
* Information retrieval
* Query expansion in search engines

---

### 🔹 **Example (Simplified)**

Let’s say we have 3 documents:

1. "Cats like milk"
2. "Dogs like bones"
3. "Cats and dogs are pets"

After building a term-document matrix and applying SVD, LSA might find:

* **Topic 1**: cats, milk, pets
* **Topic 2**: dogs, bones, pets

These topics reflect the underlying themes in the corpus without explicit labeling.

---

### Q7. Given a document-term matrix with the following counts: [6]

Document 1 Document 2 Document 3
Term 1 10 5 0
Term 2 2 0 8
Term 3 1 3 6
Calculate the TF-IDF score of “Term 1” in “Document 1”.

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

## ✅ **Latent Dirichlet Allocation (LDA)** – Definition and Explanation

### 📘 **Definition:**

**Latent Dirichlet Allocation (LDA)** is a **generative probabilistic model** used in **Natural Language Processing (NLP)** for **topic modeling**. It assumes that each document in a corpus is a mixture of topics, and each topic is a mixture of words.

---

## 📚 **How LDA is Used for Topic Modeling:**

LDA helps to discover **hidden (latent) topics** in large collections of text. It groups words that frequently occur together and assigns **probabilities** of topics to each document and words to each topic.

For example, in a collection of news articles, LDA might identify topics like "sports", "politics", or "technology", each represented by a set of top words like:

* Sports → {game, team, score, win}
* Politics → {election, party, vote, government}

---

## 🔑 **Key Components of LDA:**

### 1. **Documents:**

* A document is a **collection of words** (e.g., a news article).
* Each document is **assumed to be generated from a mixture of topics**.

### 2. **Topics:**

* A topic is a **distribution over words**.
* Each topic gives **higher probabilities to some words** (e.g., “data”, “model”, “AI” for a tech topic).

### 3. **Word Distributions:**

* For each topic, LDA learns a **probability distribution over the vocabulary**.
* For each document, LDA assigns **topic probabilities**.
* For each word, LDA assumes it was generated by **first picking a topic**, then picking a word from that topic’s word distribution.



## 📌 **Advantages of LDA in NLP:**

* Automatically **discovers hidden topics** in documents.
* Helps in **organizing, summarizing, and searching** large text corpora.
* Used in **recommendation systems**, **content classification**, and **information retrieval**.

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
4. **Outputs Contextual Vectors** – For each word, BERT gives a **unique vector** that captures its meaning in that specific sentence.


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

