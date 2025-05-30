# Impact of Machine learning in Business Intelligence Process

### Q1. State different formulae for Evaluation of classification models. [6]

Here is your content properly formatted:

---

**To evaluate how well a classification model is performing**, we use several key metrics & formulae based on values like:

### Confusion Matrix Terms:

* **(a) TP (True Positive)** – Actual outcome positive & predicted positive
* **(b) TN (True Negative)** – Actual outcome negative & predicted negative
* **(c) FP (False Positive)** – Actual outcome negative, but predicted positive
* **(d) FN (False Negative)** – Actual outcome positive, but predicted negative

---

### **Evaluation Formulae:**

---

**1) Accuracy**
Tells how many predictions the model got correct out of all predictions made.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

---

**2) Precision**
Tells how many of the positive predictions made by the model were actually correct.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

*Eg: Predicting someone has a disease when they don’t (false positive).*

---

**3) Recall (Sensitivity)**
Tells how many of the actual positive cases the model was able to correctly identify.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

*Eg: Missing a disease (false negative).*

---

**4) F1-Score**
Harmonic mean of Precision and Recall.

$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---

### Q2.Explain Bayes theorem in detail. [5]
Sure! Here's a **detailed explanation of Bayes’ Theorem** with examples:

---

### **Bayes’ Theorem: Definition**

* Bayes' Theorem is a fundamental concept in statistics used to find the probability of an event based on prior knowledge of related events.

* It helps us update our belief about an event as we get more information.
* Bayes' Theorem is widely used in areas like:
Medical diagnosis
Spam filtering
Machine learning
* This makes it very powerful for reasoning and decision-making in uncertain environments.
---


### **Bayes’ Theorem Formula:**

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

---

### **Where:**

* $P(A|B)$: Posterior probability — the probability of event **A** given that **B** is true.
* $P(B|A)$: Likelihood — the probability of event **B** given that **A** is true.
* $P(A)$: Prior probability — the initial probability of event **A**.
* $P(B)$: Marginal probability — the total probability of event **B**.



### **Example: Medical Diagnosis**

Suppose a disease affects **1% of the population**. A test detects the disease with:

* **True positive rate**: 99% → $P(\text{Positive} | \text{Disease}) = 0.99$
* **False positive rate**: 5% → $P(\text{Positive} | \text{No Disease}) = 0.05$

We want to find:

> **If a person tests positive, what is the probability they actually have the disease?**
> That is: $P(\text{Disease} | \text{Positive})$

---

### **Apply Bayes’ Theorem:**

$$
P(\text{Disease} | \text{Positive}) = \frac{P(\text{Positive} | \text{Disease}) \cdot P(\text{Disease})}{P(\text{Positive})}
$$

#### Step 1: Assign known values

* $P(\text{Disease}) = 0.01$
* $P(\text{No Disease}) = 0.99$
* $P(\text{Positive} | \text{Disease}) = 0.99$
* $P(\text{Positive} | \text{No Disease}) = 0.05$

#### Step 2: Calculate $P(\text{Positive})$

$$
P(\text{Positive}) = P(\text{Positive} | \text{Disease}) \cdot P(\text{Disease}) + P(\text{Positive} | \text{No Disease}) \cdot P(\text{No Disease})
$$

$$
= 0.99 \cdot 0.01 + 0.05 \cdot 0.99 = 0.0099 + 0.0495 = 0.0594
$$

#### Step 3: Plug into Bayes’ Theorem

$$
P(\text{Disease} | \text{Positive}) = \frac{0.99 \cdot 0.01}{0.0594} = \frac{0.0099}{0.0594} ≈ 0.1667
$$

So, even with a positive test, there's only a **\~16.67%** chance the person actually has the disease.

---

### **Applications of Bayes’ Theorem:**

1. **Medical diagnosis**
2. **Spam filtering** (e.g., Naive Bayes algorithm)
3. **Fraud detection**
4. **Weather forecasting**
5. **Search engines and recommendation systems**

---

### Q3. **Naive Bayes Classifier**:

1. **Based on Bayes’ Theorem**:
   It applies Bayes’ Theorem to classify data points by calculating the probability of each class given the input features.

2. **Assumes Feature Independence**:
   It assumes all input features are **independent** of each other, which is a "naive" assumption but works well in practice.

3. **Fast and Efficient**:
   It is computationally **simple and fast**, making it suitable for **large datasets** and real-time predictions.

4. **Used in Text Classification**:
   Widely used in **spam detection**, **sentiment analysis**, and **document categorization**.

---

### Q4. What is logistic regression with example? Discuss the types of logistic regression.[8]

Here is your provided content, rewritten properly and clearly formatted while keeping the original meaning intact:

---

### **Logistic Regression**

Logistic regression is a popular machine learning method used for **classification problems**, especially when the **target variable is binary** (i.e., Yes/No, 0/1, etc.).

* Unlike linear regression, which predicts **continuous values**, logistic regression predicts the **probability** that a given input belongs to a particular class.
* It uses the **logistic (sigmoid) function** to squeeze predicted output between 0 and 1.
* This output is interpreted as **probability**, and based on a **threshold**, the final class label is assigned.

---

### **Example:**

Suppose we want to predict whether a **student will pass or fail** based on the number of hours they studied.

* Logistic regression takes the input (hours studied) and outputs the **probability** (say 0.85) that the student will pass.
* Since 0.85 > 0.5 (assumed threshold), we classify the student as **likely to pass**.

The model learns from past data by adjusting **weights of input features** to best match the outcomes using a technique called **Maximum Likelihood Estimation**.

---

### **Types of Logistic Regression:**

**i) Binary Logistic Regression**

* This is the most common type used when the **outcome variable has only two classes**.
* *Example:* Predicting whether an email is **spam or not**.

**ii) Multinomial Logistic Regression**

* Used when the **dependent variable has more than two categories** that are **not ordered**.
* *Example:* Predicting type of fruit (**Apple**, **Banana**, **Orange**) based on features like color and weight.

**iii) Ordinal Logistic Regression**

* Used when the **output variable has more than two categories that are ordered**.
* *Example:* Predicting **customer satisfaction level** (Low, Medium, High).

---

### Q5. What is the K-mean clustering? Explain the step-by-step working of the k-mean algorithm. [6]

Here’s your provided content formatted properly:

---

### **K-means Clustering**

* K-means is one of the simplest and most widely used clustering algorithms.
* The term **"K"** represents the number of clusters we want to form.
* K-means works by dividing the dataset into **K groups based on similarity** of data points.
* It tries to **minimize the distance** between data points and the center (mean) of their assigned cluster, which is called the **centroid**.

**Example:**
Imagine a shop wants to segment customers based on their annual income and spending habits.
By using K-means clustering, the business can group customers into different types like **low income/low spending**, **high income/high spending**, etc.

---

### **Working**

1. **Choose the number of clusters (K)**
   Decide how many clusters you want to divide the data into.

2. **Initialize centroids**
   Randomly select **K** data points from the dataset as initial centroids (centers of clusters).

3. **Assign data points to nearest centroid**
   For each data point, calculate the distance to each centroid and assign the point to the closest one.

4. **Recalculate centroids**
   Once all data points are assigned to clusters, compute the **mean** of all points in each cluster to find the new centroid.

5. **Repeat steps 3 & 4**
   Continue until the centroids stop changing significantly.

6. **Final clusters**
   The final clusters show groups of similar data points, which can now be analyzed for insights.

---

### Q6. Define - Frequent item set, Minimum Support Count, Hierarchical Clustering, Regression. [6]

### Frequent Item Set

A **frequent item set** refers to a group of items that appear together often in a dataset.

* It is mainly used in market basket analysis where businesses try to find products that customers frequently buy together.
* For example, if many customers buy bread and butter together, then {bread, butter} becomes a frequent itemset.
* This concept is very useful in retail to understand buying patterns and to make decisions.
* Algorithms like **Apriori** or **FP-Growth** are used to discover these itemsets from large databases.

---

### Minimum Support Count

**Minimum support count** is the threshold used in frequent itemset mining.

* It defines the minimum number of times an itemset must appear in the dataset to be considered frequent.
* If the support count of an itemset is lower than this threshold, it is ignored.
* For example, if minimum support count is set to 3, only those itemsets that appear in 3 or more transactions will be selected.

---

### Hierarchical Clustering

Hierarchical clustering is a method of clustering data where groups are formed in a tree-like structure known as a **Dendrogram**.

* It can be done in two ways:
  a) **Agglomerative (bottom-up):**

  * Starts with each data point as its own cluster and gradually merges the closest ones.
    b) **Divisive (top-down):**
  * Starts with all data points in one big cluster and splits them into smaller clusters.
* The algorithm continues merging or splitting based on distance (or similarity) between data points.

---

### Q7. What is the Apriori Algorithm? Discuss the applications of this algorithm with example. [6]

Here’s a clear explanation for **Apriori Algorithm** along with its applications and example:

---

### What is Apriori Algorithm?

* The **Apriori Algorithm** is a classic data mining algorithm used to **find frequent itemsets** and generate **association rules** from large transactional databases.
* It works on the principle that **if an itemset is frequent, then all of its subsets must also be frequent** (Apriori property).
* The algorithm iteratively identifies frequent individual items and extends them to larger itemsets as long as those itemsets appear frequently enough in the dataset (above a minimum support threshold).
* It helps discover interesting relations and patterns between items in large datasets.



### Working of Apriori Algorithm

1. **Scan dataset to find frequent individual items (1-itemsets)** based on a minimum support threshold.

2. Then, **combine frequent 1-itemsets to form 2-itemsets**, check which of them meet the support threshold, and continue this process to generate larger itemsets.

3. Any itemset that **does not meet the support threshold is removed (pruned)**, because their larger combinations will also be infrequent.

4. Repeat until no more frequent itemsets can be found.

5. From the frequent itemsets, the algorithm **creates association rules** like "If A is bought, then B is also bought."

---

### Example

Suppose we have transactions:

* T1: {milk, bread}
* T2: {milk, bread, butter}
* T3: {bread, butter}
* T4: {milk, butter}

Steps:

* First, count how often individual items appear. For example, **milk = 3 times**.

* Then check pairs, like {milk, bread}, {bread, butter}, and see if they appear enough times (meet support threshold).

* If {milk, bread} appears 2 or more times (above threshold), it is considered frequent.

* Finally, generate rules such as:
  **milk → bread** (If someone buys milk, they also buy bread).


---

### Applications of Apriori Algorithm

1. **Market Basket Analysis:**

  * Identifies products that customers frequently buy together.
  * **Example:** If many customers buy milk and bread, the algorithm might suggest a discount on bread when milk is bought.

2. **Cross-Selling:**

   * Recommends related products to customers based on their purchase history.
   * **Example:** If a customer buys a laptop, the algorithm might suggest a mouse or keyboard.

3. **Fraud Detection:**

   * Identifies unusual patterns in transactions to detect fraudulent activities.
   **Example:** If a customer usually buys small items but suddenly buys a large item, it might be suspicious.


4. **Medical Diagnosis:**

   * Analyzes patient records to identify patterns that suggest a disease.
   **Example:** If a patient has a history of heart disease and high cholesterol, the algorithm might suggest a heart scan.

---

### Q8. What are association rules? What is the association rule mining? Explain the terms support, confidence, lift. Explain with Example.[5]

### Association Rules

* **Association rules** are if-then statements that help find relationships between variables in large datasets.
* They describe how the occurrence of one set of items is associated with another set of items.
* For example: For eg, if many customers who buy bread also buy butter, an association rule like {bread} → {butter} can be formed.
* These sules help businesses make decisions Such as promotions, recammendations, etc

---

### Association Rule Mining

* **Association Rule Mining** is the process of discovering interesting relationships, patterns, or correlations among large sets of data items.
* It is widely used in market basket analysis, web usage mining, and recommendation systems.
* The goal is to find rules that have strong support and confidence.

---

### Important Terms

1. **Support**

   * Support measures how frequently an itemset appears in the dataset.
   * It tells popularity and frequency of an itemset
   * Formula:

     $$
     \text{Support}(A \rightarrow B) = \frac{\text{Number of transactions containing both } A \text{ and } B}{\text{Total number of transactions}}
     $$


2. **Confidence**

   * Confidence measures how often the association rule has been found to be true.
   * It is the likelihood that item B bought when Item A is bought
   * Formula:

     $$
     \text{Confidence}(A \rightarrow B) = \frac{\text{Support (A and B)}}{ Support (A)}
     $$
   * If 4 out of 5, who bought milk also bought bread, then confidence is 4/5 = 0.8

3. **Lift**

   * Lift measures the strength of the association between two items.
   * Lift tells us how much moxe likely item в is bought when item A is bought compared to general probability of buying B
   * Formula:

     $$
     \text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}
     $$
   * Lift > 1 indicates a positive association between $A$ and $B$.
   * Lift = 1 indicates independence, and lift < 1 indicates a negative association.
   * Lift < 1 indicates No association

---

### Example

Consider a dataset of 5 transactions:

| Transaction ID | Items Bought        |
| -------------- | ------------------- |
| 1              | Bread, Butter       |
| 2              | Bread, Milk         |
| 3              | Bread, Butter, Milk |
| 4              | Butter, Milk        |
| 5              | Bread, Butter       |

* Let’s analyze the rule: **Bread → Butter**

* **Support** = Transactions containing both Bread and Butter / Total transactions
  \= 3 / 5 = 0.6 (60%)

* **Confidence** = Transactions with both Bread and Butter / Transactions with Bread
  Transactions with Bread = 4 (T1, T2, T3, T5)
  Confidence = 3 / 4 = 0.75 (75%)

* **Support of Butter** = Transactions with Butter / Total transactions
  \= 4 / 5 = 0.8 (80%)

* **Lift** = Confidence / Support(Butter) = 0.75 / 0.8 = 0.9375

* Interpretation: Since lift is slightly less than 1, Bread and Butter appear together a bit less frequently than expected if independent — so no strong positive association.

---

### Q9. What is a decision tree? Explain with a case study. [6]

Here’s a proper explanation of a **Decision Tree** along with a **case study example**:

---

### 🌳 What is a Decision Tree?

* A **Decision Tree** is a **supervised machine learning algorithm** used for both **classification** and **regression** tasks.
* It splits data into **branches** based on feature values, forming a **tree-like structure** of decisions.
* It works like flowchar where, Each **internal node** represents a decision on a feature, each **branch** represents an outcome of the decision, and each **leaf node** represents a final output or class label.
* The selection of features and their order is based on information gain or other criteria.
* A tree continues to split until a stopping criterion is met, such as reaching a maximum depth or a minimum number of samples in a leaf node.


#### ✅ Advantages:

* Easy to understand and interpret
* Works for both categorical and numerical data
* No need for feature scaling

#### ❌ Disadvantages:

* Prone to overfitting on noisy data
* Can be biased if one class dominates

---

### 🧪 Case Study: Student Pass/Fail Prediction

**Problem Statement**:
A school wants to predict whether a student will **pass** or **fail** based on their **study hours** and **attendance**.

#### 🎯 Dataset (simplified):

| Study Hours | Attendance (%) | Result |
| ----------- | -------------- | ------ |
| 2           | 50             | Fail   |
| 4           | 70             | Fail   |
| 6           | 80             | Pass   |
| 7           | 90             | Pass   |
| 8           | 85             | Pass   |

#### 🧠 How the Decision Tree Works:

1. The algorithm finds the **best feature** (e.g., "Study Hours") to split the data using metrics like **Gini Index** or **Information Gain**.

2. Then, it creates a question like:

   ```
   Is Study Hours > 5?
   ```

   * If Yes → Go to next node (e.g., check Attendance)
   * If No → Likely to **Fail**

3. It continues this process until it reaches a **leaf node** (Pass or Fail).

#### 📌 Final Tree (Simplified):

```
           [Study Hours > 5?]
           /              \
        No (Fail)      Yes [Attendance > 75%?]
                            /           \
                         No (Fail)     Yes (Pass)
```

---

### 📈 Application Areas

* Credit scoring
* Loan approval
* Disease diagnosis
* Customer churn prediction

---

### Q10. Write a difference between classification and clustering with applications
Here’s a clear comparison between **Classification** and **Clustering**, along with their applications:

---

### 🔍 Difference between Classification and Clustering

| Feature               | Classification                                           | Clustering                                   |
| --------------------- | -------------------------------------------------------- | -------------------------------------------- |
| **Type of Learning**  | Supervised Learning                                      | Unsupervised Learning                        |
| **Label Requirement** | Requires labeled data                                    | Does not require labeled data                |
| **Goal**              | Predict predefined categories                            | Discover hidden patterns/groups in data      |
| **Output**            | Known label (e.g., "spam" or "not spam")                 | Group/cluster label (e.g., Group A, Group B) |
| **Techniques**        | Decision Tree, Naive Bayes, Logistic Regression          | K-means, Hierarchical Clustering             |
| **Applications**      | Email spam detection, fraud detection, disease diagnosis | Customer segmentation, image grouping        |

---

### 🛠 Applications

#### ✅ **Classification Applications:**

* Email spam detection
* Disease diagnosis (e.g., predicting if a patient has diabetes)
* Sentiment analysis (positive/negative review)
* Fraud detection in banking

#### 🔵 **Clustering Applications:**

* Customer segmentation in marketing
* Social network analysis
* Document or news article grouping
* Image compression

---

#### Q11. What is the difference between hierarchical clustering and partitioning method? [5]

| Feature                        | **Hierarchical Clustering**                            | **Partitioning Method**                                      |
| ------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------ |
| **Clustering Approach**        | Builds a hierarchy of clusters (bottom-up or top-down) | Divides data into a fixed number of non-overlapping clusters |
| **Technique Used**             | Agglomerative or Divisive                              | Iterative relocation (e.g., K-means)                         |
| **Need to Predefine Clusters** | Not required                                           | Must be specified in advance (e.g., value of k in K-means)   |
| **Output Structure**           | Dendrogram (tree-like structure of clusters)           | Flat set of clusters                                         |
| **Scalability**                | Slower and less efficient for large datasets           | Faster and more efficient for large datasets                 |
| **Examples**                   | Agglomerative Clustering, Divisive Clustering          | K-means, K-medoids                                           |


### Q12. Suppose we have group of visitors to the website using their age as following 16, 16, 17, 20, 20, 21, 21, 22, 23, 29, 36, 41, 42, 43, 44, 45, 61, 62, 66 Find out the clusters considering K = 2. [6]

To find **clusters with K = 2** using the **K-means algorithm** on the given age data:

### 🧮 Given Data (Ages):

`16, 16, 17, 20, 20, 21, 21, 22, 23, 29, 36, 41, 42, 43, 44, 45, 61, 62, 66`

---

### ✅ Step-by-Step K-Means Clustering (K = 2)

#### **Step 1: Initialize Centroids**

Choose any two initial centroids. Let’s pick:

* Centroid 1 (C1) = 20
* Centroid 2 (C2) = 60

#### **Step 2: Assign Points to Nearest Centroid**

| Age | Distance to C1 (20) | Distance to C2 (60) | Assigned to |
| --- | ------------------- | ------------------- | ----------- |
| 16  | 4                   | 44                  | C1          |
| 16  | 4                   | 44                  | C1          |
| 17  | 3                   | 43                  | C1          |
| 20  | 0                   | 40                  | C1          |
| 20  | 0                   | 40                  | C1          |
| 21  | 1                   | 39                  | C1          |
| 21  | 1                   | 39                  | C1          |
| 22  | 2                   | 38                  | C1          |
| 23  | 3                   | 37                  | C1          |
| 29  | 9                   | 31                  | C1          |
| 36  | 16                  | 24                  | C1          |
| 41  | 21                  | 19                  | C2          |
| 42  | 22                  | 18                  | C2          |
| 43  | 23                  | 17                  | C2          |
| 44  | 24                  | 16                  | C2          |
| 45  | 25                  | 15                  | C2          |
| 61  | 41                  | 1                   | C2          |
| 62  | 42                  | 2                   | C2          |
| 66  | 46                  | 6                   | C2          |

* Cluster 1 (C1): `16, 16, 17, 20, 20, 21, 21, 22, 23, 29, 36`
* Cluster 2 (C2): `41, 42, 43, 44, 45, 61, 62, 66`

#### **Step 3: Recalculate Centroids**

* New C1 = Average of C1 = (16+16+17+20+20+21+21+22+23+29+36)/11 = **22.7**
* New C2 = Average of C2 = (41+42+43+44+45+61+62+66)/8 = **50.5**

#### **Step 4: Reassign Based on New Centroids**

Repeat assignment based on new centroids (22.7 and 50.5). If assignments don’t change, stop. If changed, recalculate centroids again.

---

### ✅ Final Clusters:

After convergence:

* **Cluster 1 (Young Group)**: `16, 16, 17, 20, 20, 21, 21, 22, 23, 29, 36`
* **Cluster 2 (Older Group)**: `41, 42, 43, 44, 45, 61, 62, 66`

---

### Q13. The database has shown transactions. Let min_support = 02 and min_confidence = 70%. Find all frequent item set using Apriori algorithm and generate strong association rules [6]

| TID  | List of Items  |
| ---- | -------------- |
| T100 | I1, I2, I5     |
| T100 | I2, I4         |
| T100 | I2, I3         |
| T100 | I1, I2, I4     |
| T100 | I1, I3         |
| T100 | I2, I3         |
| T100 | I1, I3         |
| T100 | I1, I2, I3, I5 |
| T100 | I1, I2, I3     |


Let's solve this step-by-step using the **Apriori Algorithm**.

---

### **Given:**

* **Transactions (9 total)**
* **min\_support = 2**
* **min\_confidence = 70%**

---

## 🧮 Step 1: **Transaction Table**

| TID | Items          |
| --- | -------------- |
| T1  | I1, I2, I5     |
| T2  | I2, I4         |
| T3  | I2, I3         |
| T4  | I1, I2, I4     |
| T5  | I1, I3         |
| T6  | I2, I3         |
| T7  | I1, I3         |
| T8  | I1, I2, I3, I5 |
| T9  | I1, I2, I3     |

---

## ✅ Step 2: **Find Frequent Itemsets (support ≥ 2)**

### **L1 – 1-itemsets**

| Item | Support |
| ---- | ------- |
| I1   | 6       |
| I2   | 7       |
| I3   | 6       |
| I4   | 2       |
| I5   | 2       |

**Frequent 1-itemsets**:
`{I1}, {I2}, {I3}, {I4}, {I5}` ✅

---

### **L2 – 2-itemsets**

Generate candidate pairs from L1:

* {I1, I2}, {I1, I3}, {I1, I4}, {I1, I5}
* {I2, I3}, {I2, I4}, {I2, I5}
* {I3, I4}, {I3, I5}
* {I4, I5}

Check support counts:

| Itemset  | Support   |
| -------- | --------- |
| {I1, I2} | 4         |
| {I1, I3} | 4         |
| {I1, I4} | 1       ❌ |
| {I1, I5} | 2         |
| {I2, I3} | 4         |
| {I2, I4} | 2         |
| {I2, I5} | 2         |
| {I3, I4} | 0       ❌ |
| {I3, I5} | 1       ❌ |
| {I4, I5} | 0       ❌ |

**Frequent 2-itemsets**:
`{I1,I2}, {I1,I3}, {I1,I5}, {I2,I3}, {I2,I4}, {I2,I5}` ✅

---

### **L3 – 3-itemsets**

Valid combinations from frequent 2-itemsets:

* {I1, I2, I3}, support = 2 ✅
* {I1, I2, I5}, support = 2 ✅

**Frequent 3-itemsets**:
`{I1, I2, I3}, {I1, I2, I5}` ✅

---

## ✅ Step 3: **Generate Association Rules (confidence ≥ 70%)**

### From `{I1, I2, I3}` (support = 2)

* I1, I2 → I3 = 2/4 = **50%** ❌
* I1, I3 → I2 = 2/4 = **50%** ❌
* I2, I3 → I1 = 2/4 = **50%** ❌
* I1 → I2, I3 = 2/6 = **33.3%** ❌
* I2 → I1, I3 = 2/7 = **28.6%** ❌
* I3 → I1, I2 = 2/6 = **33.3%** ❌

**No strong rules from {I1, I2, I3}**

---

### From `{I1, I2, I5}` (support = 2)

* I1, I2 → I5 = 2/4 = **50%** ❌
* I1, I5 → I2 = 2/2 = **100%** ✅
* I2, I5 → I1 = 2/2 = **100%** ✅
* I1 → I2, I5 = 2/6 = **33.3%** ❌
* I2 → I1, I5 = 2/7 = **28.6%** ❌
* I5 → I1, I2 = 2/2 = **100%** ✅

✅ **Strong Rules (confidence ≥ 70%)**:

1. **I1, I5 → I2** (100%)
2. **I2, I5 → I1** (100%)
3. **I5 → I1, I2** (100%)

---

### ✅ Final Answer

**Frequent Itemsets**:

* 1-item: `{I1}, {I2}, {I3}, {I4}, {I5}`
* 2-item: `{I1,I2}, {I1,I3}, {I1,I5}, {I2,I3}, {I2,I4}, {I2,I5}`
* 3-item: `{I1,I2,I3}, {I1,I2,I5}`

**Strong Association Rules (≥70% confidence)**:

* `I1, I5 → I2` (100%)
* `I2, I5 → I1` (100%)
* `I5 → I1, I2` (100%)

