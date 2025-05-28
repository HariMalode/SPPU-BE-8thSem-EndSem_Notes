# NLP Tools and Techniques

### Q1. List the tools available for the development of NLP applications? Write the features of any 3 tools. [7]
### Compare and constrast the natural Language Toolkit (NLTK), spaCy, and TexBlob. what are their main features and in what use cases are they most suitable? [9]

Tools are the tools that are used to develop NLP applications. They are used to perform various NLP tasks such as tokenization, stemming, lemmatization, POS tagging, parsing, etc.

### ✅ **Popular NLP Tools:**

1. **NLTK (Natural Language Toolkit)**
2. **spaCy**
3. **Stanford CoreNLP**
4. **TextBlob**
5. **Gensim**
6. **Transformers (by Hugging Face)**
7. **OpenNLP**
8. **Flair (by Zalando)**
9. **AllenNLP**
10. **Polyglot**

---

### 🔍 Features of Any 3 NLP Tools:

Sure! Here's a detailed explanation of **NLTK**, **spaCy**, and **TextBlob**, including their **main features** and **use cases**:

---

## 🔷 1. NLTK (Natural Language Toolkit)

### 📌 What it is:

* A powerful Python library developed mainly for **teaching, research, and prototyping** in NLP.
* Rich in linguistic data, corpora, and tutorials.

### 🌟 Main Features:

* **Tokenization** (sentence and word level)
* **Stemming and Lemmatization** 
         - Stemming is the process of reducing a word to its root or base form by removing suffixes.
         - Lemmatization reduces a word to its dictionary form (lemma), considering the part of speech and meaning.


* **Part-of-Speech (POS) tagging**
* **Parsing and chunking**
* **Named Entity Recognition (NER)**
* **Text classification**
* **Sentiment analysis (basic)**
* Extensive **corpus and lexical resources** (like WordNet)

### ✅ Use Cases:

* Academic and educational projects
* Prototyping NLP models
* Linguistic and language structure analysis
* Teaching NLP fundamentals

---

## 🔷 2. spaCy

### 📌 What it is:

* An **industrial-strength**, fast, and modern NLP library written in Python and Cython.
* Designed for real-world applications and production use.

### 🌟 Main Features:

* **Fast and accurate tokenization**
* **POS tagging**, **Dependency Parsing**, **NER**
* **Pre-trained language models** for multiple languages
* **Word vectors** and similarity computation
* **Entity linking and sentence segmentation**
* Integration with **deep learning libraries** like PyTorch & TensorFlow
* Supports **custom pipelines** and extensions

### ✅ Use Cases:

* **Production-grade NLP applications**
* **Chatbots and virtual assistants**
* **Information extraction and document classification**
* **Text summarization and parsing at scale**
* Building pipelines with machine learning models

---

## 🔷 3. TextBlob

### 📌 What it is:

* A beginner-friendly NLP library built on top of **NLTK** and **Pattern**.
* Provides a simplified API for common NLP tasks.

### 🌟 Main Features:

* **Sentiment analysis** (positive/negative/neutral)
* **Tokenization**, **POS tagging**, **noun phrase extraction**
* **Translation** and **language detection**
* **Spelling correction**
* Easy-to-use, concise syntax

### ✅ Use Cases:

* Quick NLP prototyping and scripts
* **Social media sentiment analysis**
* Basic language processing tasks with minimal code
* **Content filtering** and **text clean-up tools**
* Educational and lightweight NLP tasks

4. **OpenNLP**
### 📌 What it is:
* A Java-based NLP library developed by the Apache Software Foundation.
* Focused on **text analysis** and **information extraction**.

### 🌟 Main Features:
* **Tokenization**, **POS tagging**, **NER**

5. **Gensim library**
### 📌 What it is:
* A Python library for **topic modeling** and **document similarity**.
* Built on top of **NumPy** and **SciPy**.
### 🌟 Main Features:
* **Word2Vec** and **Doc2Vec** for vector representations of words and documents.

---

### 🔁 Comparison Table:

| Feature             | NLTK                  | spaCy                      | TextBlob               |
| ------------------- | --------------------- | -------------------------- | ---------------------- |
| Target audience     | Researchers, students | Developers, production use | Beginners, rapid dev   |
| Speed               | Moderate              | Very Fast                  | Moderate               |
| Built-in ML support | Basic                 | Integrates with ML libs    | No (uses NLTK/pattern) |
| Sentiment Analysis  | Basic (customizable)  | Requires add-ons           | Built-in               |
| Language Support    | Rich corpus           | Multi-language pipelines   | English + limited      |
| Ease of Use         | Medium                | Medium                     | Very Easy              |

---

### Q2. Describe in detail the Lesk algorithm and Walker’s algorithm for word sense disambiguation. [10]

* *The* **Lesk algorithm** is a classic algorithm used for **Word Sense Disambiguation (WSD)** in **Natural Language Processing (NLP)**.
* It helps determine the correct **meaning (sense)** of a word based on its context.


## 📘 What is Word Sense Disambiguation (WSD)?

WSD is the task of identifying which sense of a word is used in a given context, especially when a word has **multiple meanings**.

### Example:

In the sentence:
**“I went to the bank to deposit money,”**
“bank” means **financial institution**, not **riverbank**.

---

## 🔍 Lesk Algorithm: Overview


### 🧠 **Core Idea:**

> **"Words in a given context will share a common meaning."**
> So, the sense of a word whose **definition has the most words in common with the context** is chosen.

---

## 🔧 Steps of the Lesk Algorithm:

1. **Identify the ambiguous word** and the context (surrounding words).
2. **Retrieve dictionary definitions (glosses)** of the target word and the context words.
3. **Count the number of overlapping words** between the glosses of the target word and the glosses of the context.
4. **Choose the sense with the highest overlap.**

---

## 📄 Example:

### Sentence:

**“He sat by the bank of the river.”**

### Ambiguous word:

**“bank”**

### Possible senses from dictionary:

1. **bank (financial)** → “An institution for receiving, lending, and safeguarding money.”
2. **bank (river)** → “The land alongside or sloping down to a river or lake.”

### Context words:

“sat,” “river”

### Gloss overlaps:

* **Gloss 1 (financial):** "money, institution, lending" → No overlap with “river”
* **Gloss 2 (river):** "land, river, lake" → **Overlap with "river"**

✅ **Selected Sense:** “river bank” (Sense 2)

---

### Q3. Describe Walker’s algorithm for word sense disambiguation. How does it differ from other disambiguation techniques like Lesk’s Algorithm, and what are the scenarios where it can be most effective? [9]

- Unlike **Lesk’s Algorithm**, **Walker’s Algorithm** is a **coreference resolution** techniquein NLP.
- Coreference resolution is the task of finding all expressions (like pronouns or noun phrases) that refer to the **same entity** in a text.
- It focuses on identifying **coreferential expressions** (words that refer to the same entity) in a text.
- It is particularly useful in situations where **pronouns** or **noun phrases** are used to refer to a previously mentioned entity.
- It is **not** typically used for **Word Sense Disambiguation** (WSD) like **Lesk’s Algorithm**.


### Example:

> "Hariom went to the market. **He** bought some apples."

Here, "**He**" and "**Hariom**" refer to the same person — they are **coreferent**.

---

## 📘 Walker’s Algorithm (Centering Theory-Based Coreference Resolution)

Walker’s algorithm is a procedural interpretation of **Centering Theory**, which focuses on **local coherence** in discourse. It helps resolve references by **ranking and tracking possible antecedents** (the noun phrases referred to).



## ✅ Example:
### 📘 **Example:**

**Sentence:**
*"He caught a bat in the cave."*

**Ambiguous Word:** `bat`

**Possible Senses:**

* **Sense 1:** Flying mammal
* **Sense 2:** Sports equipment

---

1. Build the graph using semantic relations from WordNet.
2. Run random walks from context words like `caught` and `cave`.
3. These may frequently lead to the sense: **flying mammal**, due to proximity to `cave` and animal-related nodes.

✅ **Chosen sense:** *bat as a flying mammal*

---


### Q4. Compare the Indo Word Net with the traditional WordNet. What are the key differences and advantages of IndoWordNet, especially in the context of Indian languages? [8]
###  Write a note on : WordNet. [10]


### 📚 What is WordNet?

**WordNet** is a **lexical database** for the English language that groups words into sets of **synonyms** called **synsets**, and records the **semantic relationships** between them.

---

### 🔑 Key Features:

1. **Synsets**:
   Groups of synonymous words that represent a single concept (e.g., *car, auto, automobile*).

2. **Semantic Relations**:

   * **Synonymy** (same meaning)
   * **Antonymy** (opposite meaning)
   * **Hypernymy** (general term)
   * **Hyponymy** (specific term)
   * **Meronymy** (part-whole)
   * **Holonymy** (whole-part)

3. **POS Categorization**:
   Covers **nouns**, **verbs**, **adjectives**, and **adverbs** separately.

4. **Used in NLP Tasks**:

   * Word Sense Disambiguation
   * Semantic Similarity
   * Information Retrieval
   * Text Classification

---

### 🧠 Example:

For the word **"bank"**, WordNet lists multiple synsets:

* *bank (financial institution)*
* *bank (river side)*

Each with definitions, examples, and relationships to other words.

---

Let me know if you want examples of how to use WordNet in Python with NLTK!


## ✅ What is IndoWordNet?

IndoWordNet is a lexical database of Indian languages developed on the model of English WordNet, aimed at capturing semantic and lexical relations across multiple Indian languages. It was initiated at IIT Bombay.
* **IndoWordNet extends WordNet** to suit the multilingual and cultural landscape of India.
* It is an essential **NLP resource for Indian languages**, helping build scalable, culturally aware AI and translation tools.


---

## 🔍 Key Differences: IndoWordNet vs Traditional WordNet

| Feature                | **Traditional WordNet (English)** | **IndoWordNet (Indian Languages)**                                   |
| ---------------------- | --------------------------------- | -------------------------------------------------------------------- |
| **Language Support**   | English only                      | 18+ Indian languages (Hindi, Marathi, Bengali, etc.)                 |
| **Structure**          | Single-language lexical network   | Multilingual, interconnected synsets across languages                |
| **Conceptual Base**    | Princeton WordNet                 | Based on a **common concept synset structure** across languages      |
| **Cultural Context**   | Western-centric concepts          | Incorporates **Indian cultural, social, and linguistic** nuances     |
| **Semantic Relations** | Synonymy, hyponymy, etc.          | All semantic relations of WordNet + **language-specific relations**  |
| **Development**        | Manual + expert curation          | **Collaborative** development across institutions                    |
| **Applications**       | English NLP tools                 | **Machine Translation, CLIR, POS tagging, WSD** for Indian languages |

---

## ✅ Advantages of IndoWordNet in the Indian Context

1. **Multilingual Support**

   * Supports multiple Indian languages using a **common synset ID** for alignment.

2. **Cultural Relevance**

   * Includes words, idioms, and concepts relevant to **Indian culture and traditions**.

3. **Cross-Lingual Applications**

   * Enables **machine translation** and **cross-lingual information retrieval (CLIR)** across Indian languages.

4. **Word Sense Disambiguation (WSD)**

   * Aids in **accurate sense mapping** due to shared concepts across languages.

5. **Language Resource for NLP**

   * Acts as a crucial **lexical resource** for building NLP tools like POS taggers, lemmatizers, and translators.

6. **Educational and Linguistic Research**

   * Supports **comparative linguistics**, bilingual dictionaries, and regional language education.

---

## 🧪 Example

Let’s say the concept of "teacher" exists in:

* English WordNet: `teacher` = "a person whose occupation is teaching"
* IndoWordNet:

  * Hindi: **शिक्षक**
  * Marathi: **शिक्षक**
  * Bengali: **শিক্ষক**
  * Tamil: **அசிரியர்**

➡ All are **mapped to the same synset ID**, which allows **cross-lingual operations**.

---

### Q5. What is the significance of PropBank and VerbNet in linguistic resources? Provide examples of how these resoures can be used to extract semantic information from text. [8]

**Skipped** 

### Q6. Explain the following lexical knowledge networks? [10]

i) WordNet
ii) Indo WordNet
iii) VerbNets
iv) PropBank
v) Treebanks

Here’s a clear explanation of the **lexical knowledge networks** listed:

---

### 🟦 **i) WordNet**

**WordNet** is a large lexical database of English where:

* Words are grouped into **synsets** (sets of cognitive synonyms).
* Each synset expresses a **unique concept**.
* Synsets are connected through **semantic relations** like:

  * **Synonymy** (similar meaning),
  * **Antonymy** (opposites),
  * **Hyponymy** (is-a),
  * **Meronymy** (part-of).

🔹 **Use Case**: Word Sense Disambiguation, Information Retrieval, Text Classification.

**Example**:
For the word *car*, synset includes: `{car, auto, automobile, machine, motorcar}`

---

### 🟩 **ii) Indo WordNet**

**IndoWordNet** is a multilingual lexical database for **Indian languages**, modeled after WordNet.

* Supports over **18 Indian languages** (Hindi, Marathi, Tamil, etc.).
* Uses a **common concept synset** shared across languages.
* Adds **cultural and linguistic relevance** specific to Indian contexts.

🔹 **Use Case**: Machine Translation, Cross-Lingual Information Retrieval.

**Example**:
Synset for *water*:

* Hindi: पानी
* Marathi: पाणी
* Tamil: தண்ணீர்

---

### 🟨 **iii) VerbNet**

**VerbNet** is a **lexicon of English verbs** grouped into **classes** based on **syntactic and semantic behavior** (based on Levin’s classification).

* Assigns **semantic roles** like Agent, Theme, Recipient.
* Provides **syntactic frames** (like NP V NP NP).

🔹 **Use Case**: Verb classification, Semantic Role Labeling (SRL), Grammar Checking.

**Example**:
Verb Class: *give-13.1*

* Roles: Agent (giver), Theme (thing given), Recipient (receiver)

---

### 🟧 **iv) PropBank**

**PropBank** is a **corpus** annotated with **predicate-argument structures** (semantic roles) for verbs.

* Adds a **layer of meaning** to syntactic structure.
* Assigns **rolesets** for different verb senses.

🔹 **Use Case**: SRL, Event Extraction, Question Answering.

**Example**:
Sentence: "Alice baked a cake."

* Predicate: bake.01
* Arg0: Alice (baker)
* Arg1: a cake (thing baked)

---

### 🟥 **v) Treebanks**

**Treebanks** are corpora where **sentences are syntactically annotated** with **parse trees**.

* Can be **constituency-based** (phrase structure) or **dependency-based** (word-to-word relations).
* Based on **linguistic rules** or automatically parsed and corrected by linguists.

🔹 **Use Case**: Syntax Parsing, Grammar Learning, NLP Model Training.

**Example**:
Sentence: "The boy eats an apple."
→ Parse Tree showing subject (boy), verb (eats), object (apple)

---

### Q7. Write Python code using NLTK library to split the text into tokens using whitespace, punctuation-based and default tokenization methods. 

1. **Whitespace-based tokenization**
2. **Punctuation-based tokenization**
3. **Default NLTK tokenizer (word\_tokenize)**

```python
import nltk
from nltk.tokenize import word_tokenize, WhitespaceTokenizer, RegexpTokenizer

# Download required resources
nltk.download('punkt')

# Sample text
text = "Hello! I'm learning NLP using the NLTK library. It's quite interesting."

# 1. Whitespace-based tokenization
whitespace_tokenizer = WhitespaceTokenizer()
whitespace_tokens = whitespace_tokenizer.tokenize(text)
print("Whitespace Tokenization:")
print(whitespace_tokens)

# 2. Punctuation-based tokenization (using regex to split on non-word characters)
punct_tokenizer = RegexpTokenizer(r'\w+')
punct_tokens = punct_tokenizer.tokenize(text)
print("\nPunctuation-based Tokenization (removes punctuation):")
print(punct_tokens)

# 3. Default tokenization using NLTK (word_tokenize)
default_tokens = word_tokenize(text)
print("\nDefault NLTK Tokenization (word_tokenize):")
print(default_tokens)
```

### 🔹 Output Example:

```python
Whitespace Tokenization:
['Hello!', "I'm", 'learning', 'NLP', 'using', 'the', 'NLTK', 'library.', "It's", 'quite', 'interesting.']

Punctuation-based Tokenization (removes punctuation):
['Hello', 'I', 'm', 'learning', 'NLP', 'using', 'the', 'NLTK', 'library', 'It', 's', 'quite', 'interesting']

Default NLTK Tokenization (word_tokenize):
['Hello', '!', 'I', "'m", 'learning', 'NLP', 'using', 'the', 'NLTK', 'library', '.', 'It', "'s", 'quite', 'interesting', '.']
```

Let me know if you want sentence tokenization too!
