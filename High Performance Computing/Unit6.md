# High Performance Computing Applications

### Q1. Explain Parallel Depth First Search algorithm in detail? [6]


### ✅ **Parallel Depth First Search (DFS) Algorithm – Explained in Detail**

---

### 🔹 **What is DFS?**

DFS (Depth First Search) is a graph traversal algorithm that explores a graph by going as deep as possible along each branch before backtracking.
---

### 🔹 **Why Parallel DFS?**

Standard DFS is **sequential** in nature (due to its recursive or stack-based design). But with **Parallel DFS**, we aim to:

* Speed up traversal
* Utilize multiple processors or threads

Paxallel DFS attempts to divide a graph traversal among multiple pracessors, so that multiple branches of graph can be exploxed at same time, Speeding up piacess

---

### 🔹 **Challenges in Parallel DFS**

* DFS has **sequential dependencies** (a node must be visited before its children).
* Requires **careful management of visited nodes** to avoid duplicate work or race conditions.

---

### ✅ **Steps of Parallel DFS Algorithm**

1. **Start from root node** (or multiple root nodes if disconnected graph).
2. **Distribute neighbors** of root node across available threads.
3. Each thread performs DFS on its assigned neighbor **independently**.
4. Use a **shared visited\[] array** (with synchronization) to ensure a node is not visited twice.
5. When a thread finishes a subtree, it returns.
6. All threads work concurrently to explore separate branches of the graph.

---

### 🔹 **Example**

Graph:

```
A - B - C  
|     |  
D     E
```

Start DFS from node A. Parallelize children (B, D):

* Thread 1: DFS on B → C → E
* Thread 2: DFS on D

All threads update the `visited[]` array in a thread-safe manner.

---

### ✅ **Advantages**

* Faster traversal for large graphs
* Efficient use of multicore processors
* Better performance in sparse graphs

---

### ❌ **Limitations**

* Complex synchronization (mutexes, atomic operations)
* Overhead of thread management
* Not all parts of DFS can be parallelized efficiently

---

### Q2. Explain parallel BFS algorithm in brief and analyze its complexity[6]


### 🔹 **What is BFS?**

Breadth First Search (BFS) is a graph traversal algorithm that explores the graph layer by layer, i.e., it visits all nodes at the current depth (or level) before moving on to the next level.

---

### 🔹 **Why Parallel BFS?**

- Parallel BFS is used to **speed up** traversal, especially for **large and wide graphs**, by exploring multiple neighbors concurrently.
- Because of its level-wise structure, BFS is easier to parallelize compared to Depth First Search (DFS).
- Ideal for AI, social network analysis, web crawling

### 🔹 **How It Works**

* For each level of the graph:

  * We can assign **different threads or processors** to visit different nodes in that level.
  * Each thread then explores the **neighbors of its assigned node** at the **same time**, i.e., **in parallel**.

This leads to a significant **reduction in total traversal time**, especially for **large and wide graphs**.

---

### ✅ **Steps of Parallel BFS Algorithm**

1. **Initialize** a queue with the start node.
2. Create a **visited\[] array** to track visited nodes.
3. **Repeat until queue is empty**:

   * For each node in the current level:

     * In **parallel**, visit all unvisited neighbors.
     * Mark them as visited and add to the next level's queue.
4. Swap current and next level queues and continue.

---

### ✅ **Time Complexity of Parallel BFS**

* **Sequential BFS Time Complexity**:

  $$
  O(V + E)
  $$

  where:

  * $V$ = number of vertices
  * $E$ = number of edges

* **Parallel BFS Time Complexity**:

  $$
  O\left(\frac{V + E}{P}\right)
  $$

  where:

  * $P$ = number of processors or threads

---

### Q3. Explain odd-even transportation in bubble sort using parallel formulation. Give one stepwise example solution using odd-even transportation. [8]


**Bubble Sort is a simple comparison-based sorting algorithm that repeatedly swaps adjacent elements if they are in the wrong order, causing larger elements to "bubble up" to the end of the list.**


## ✅ Odd-Even Transposition Sort (Parallel Bubble Sort)

### 🔸 **Definition**:

Odd-Even Transposition Sort is a parallel variant of Bubble Sort, designed to run in multiple phases to sort a list of numbers using **parallel comparisons and swaps**.

* It uses alternating phases of **odd-indexed** and **even-indexed** comparisons to gradually push larger elements towards the end.

---

## ⚙️ **Parallel Formulation**:



**In the parallel version, instead of comparing just one pair at a time, we divide the comparisons into two phases:**

1. **Even Phase** – Compare and swap elements at even indices with their next elements (i.e., index pairs (0,1), (2,3), (4,5), …).
2. **Odd Phase** – Compare and swap elements at odd indices with their next elements (i.e., index pairs (1,2), (3,4), (5,6), …).


## 🧮 **Example** (Step-by-step):

i) Even phase: (compare A\[0]-A\[1], A\[2]-A\[3])
Compare (5,1) → \[1,5,4,2]
Compare (4,2) → \[1,5,2,4]

ii) Odd phase: (compare A\[1]-A\[2])
Compare (5,2) → \[1,2,5,4]

iii) Even phase: (compare A\[0]-A\[1], A\[2]-A\[3])
Compare (1,2) → \[1,2,5,4]
Compare (5,4) → \[1,2,4,5]

iv) Odd phase: (compare A\[1]-A\[2])
Compare (2,4) → \[1,2,4,5]

Sorted!


## 📌 **Key Points**:

* Runs in **n** phases (n = number of elements).
* **Time complexity**:

  * Sequential: $O(n^2)$
  * Parallel: $O(n)$ with $\frac{n}{2}$ processors.
* Efficient when implemented on parallel machines.

---

### Q4. Compare an algorithm for sequential and parallel Merge sort. Analyze the complexity for the same. [8]
(Write short notes on : i Parallel Merge sort)


### **Parallel Merge Sort – Explanation**

**Parallel Merge Sort** is an extension of the traditional (sequential) merge sort algorithm designed to leverage multiple processors or threads for faster sorting.

---

### **Steps of Parallel Merge Sort:**

1. **Divide:**

   * Split the input array into `p` equal parts where `p` is the number of available processors/threads.
   * Each processor gets a sub-array of size `n/p`.

2. **Sort:**

   * Each processor independently sorts its assigned sub-array using merge sort (or any other efficient sort) in parallel.

3. **Merge:**

   * After sorting, sub-arrays are merged in parallel using a tree-based merging strategy:

     * Merge pairs of sorted sub-arrays in multiple rounds until one final sorted array is obtained.
     * Merging can be done in parallel at each level.

---

### **Example:**

Given array: `[8, 3, 5, 1, 7, 6, 4, 2]`

* Suppose we have 4 processors.

**Step 1: Divide**

* P1: `[8, 3]`, P2: `[5, 1]`, P3: `[7, 6]`, P4: `[4, 2]`

**Step 2: Sort in parallel**

* P1: `[3, 8]`, P2: `[1, 5]`, P3: `[6, 7]`, P4: `[2, 4]`

**Step 3: Merge in parallel**

* Level 1:

  * P1 & P2 → `[1, 3, 5, 8]`
  * P3 & P4 → `[2, 4, 6, 7]`

* Level 2:

  * Final merge → `[1, 2, 3, 4, 5, 6, 7, 8]`

---
### **Advantages:**

* Reduces execution time by parallelizing sorting and merging.
* Efficient on multi-core CPUs or GPUs.

### **Limitations:**

* Overhead of thread management.
* Requires synchronization during merge steps.


**Comparison of Sequential and Parallel Merge Sort**

| Aspect         | Sequential Merge Sort                                        | Parallel Merge Sort                                                      |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------ |
| **Approach**   | Recursively divides array into halves and sorts sequentially | Divides array and sorts subarrays concurrently using multiple processors |
| **Execution**  | One processor performs all tasks in order                    | Multiple processors work on different parts simultaneously               |
| **Merge step** | Merging is done sequentially at each level                   | Merging can be done in parallel for different subproblems                |
| **Speedup**    | Limited by single processor speed                            | Can achieve significant speedup depending on processors available        |
| **Overhead**   | Minimal overhead                                             | Additional overhead due to thread management and synchronization         |

---

### Complexity Analysis

* **Sequential Merge Sort**:
  Time Complexity: $O(n \log n)$

  * Recursively divides array into $\log n$ levels
  * Each level merges $n$ elements

* **Parallel Merge Sort**:
  Time Complexity: $O\left(\frac{n}{p} \log \frac{n}{p} + \log p \right)$
  where $p$ = number of processors

  * Each processor sorts $\frac{n}{p}$ elements: $O\left(\frac{n}{p} \log \frac{n}{p}\right)$
  * Parallel merging takes $O(\log p)$ time

---

### Q5. Explain Classification in Distributed Computing (HPC)**


### **What is Document Classification?**

**Document Classification** is a task in Natural Language Processing (NLP) where a document is assigned to one or more predefined categories based on its content.
Example: Classifying emails as *spam* or *not spam*, categorizing news articles by topics (sports, politics, etc.).

### **Why Use Distributed Computing / HPC?**

When dealing with:

* Large volumes of documents (e.g., millions of articles, tweets, emails),
* High-dimensional data (thousands of features/words),
* Complex models (deep learning, large SVMs),
* Allows real-time classification at scale.

**High-Performance Computing (HPC)** and **Distributed Computing** are used to **speed up processing** and **handle scalability**.

---

### **How Document Classification Works in HPC (One line per point):**

1. **Data Distribution** – Large document sets are split across multiple processors or nodes.
2. **Parallel Feature Extraction** – Each node extracts features (e.g., TF-IDF) from its documents.
3. **Distributed Model Training** – Nodes train the model in parallel and synchronize updates.
4. **Parallel Inference** – Trained model is used to classify documents in parallel.
5. **Result Aggregation** – Classification results from all nodes are collected and combined.

### **Use Cases:**

* Spam detection on email servers.
* Sentiment analysis of social media data.
* News article categorization in online platforms.
* Legal document classification in law firms.

---

### Q6.  What is Kubernets? Explain its features and applications. [4]

### **What is Kubernetes?**

* **Kubernetes** is an open-source framework developed by **Google** for managing containerized applications in a **clustered environment**.
* It **automates deployment**, **scaling**, and **management** of application containers across **multiple hosts**.
* Kubernetes simplifies managing **complex applications** with many **interconnected services** by abstracting infrastructure.
* It works especially well withh containers like Docker

---

### **Features of Kubernetes:**

1. **Automated Scheduling** – Efficiently assigns containers to nodes based on resource availability.
2. **Self-Healing** – Automatically restarts failed containers, replaces and reschedules them when nodes die.
3. **Horizontal Scaling** – Scale applications up or down automatically based on CPU usage or custom metrics.
4. **Load Balancing and Service Discovery** – Exposes containers using DNS or IP and balances traffic.
5. **Automated Rollouts and Rollbacks** – Manages updates to applications and rolls back changes if needed.
6. **Storage Orchestration** – Automatically mounts the storage system of your choice (local, cloud, etc.).

---

### **Applications of Kubernetes:**

1. **Microservices Management** – Efficiently deploy and manage microservices-based architectures.
2. **Cloud-Native App Deployment** – Used for deploying scalable applications in cloud environments.
3. **DevOps Automation** – Automates CI/CD pipelines and simplifies the DevOps workflow.
4. **Edge Computing** – Deploys and manages applications on distributed edge infrastructure.
5. **Big Data and AI/ML Workloads** – Manages complex data pipelines and ML models across clusters.

---

### Q7. Write short note on GPU Applications. [4]


**Note on GPU Applications:**

* **Graphics Rendering:** Originally designed for rendering 2D/3D graphics in games and animations.
* **Scientific Computing:** Used in simulations, weather prediction, molecular modeling, and astrophysics.
* **Machine Learning & AI:** Accelerates training of deep neural networks due to massive parallelism.
* **Image & Signal Processing:** Speeds up tasks like filtering, edge detection, and FFTs.
* **Financial Modeling:** Used for risk analysis, option pricing, and algorithmic trading.
* **Cryptography & Blockchain:** Accelerates hashing functions and mining algorithms.
* **Medical Imaging:** Improves speed and precision in MRI/CT scan processing and diagnostics.
* **Data Analytics:** Enhances performance of large-scale data analysis and real-time processing.
* **Robotics:** Enables real-time control and navigation in autonomous robots.
* ***Autonomous Vehicles:** Accelerates perception and decision-making in self-driving cars.

### Q8. What are the issues in sorting on parallel computers? Explain with appropriate example? [6]


### **Parallel Computing in AI/ML**

* Parallel computing accelerates AI/ML tasks involving large datasets and complex models.
* CPUs alone are slow for such heavy computations.
* Parallel computing allows multiple tasks to run simultaneously using multiple cores/processors.
* GPUs are commonly used due to thousands of smaller cores capable of parallel processing.
* This leads to faster training and efficient computation in AI/ML workloads.

---

### **Issues in Sorting on Parallel Computers:**

1. **Load Balancing:**

   * Equal work distribution among processors is hard.
   * Some processors might be idle while others are overloaded.

2. **Data Dependency:**

   * Sorting often needs comparison between elements.
   * Ensuring correct order when data is split is difficult.

3. **Communication Overhead:**

   * Excessive communication between processors to exchange data or merge results slows performance.

4. **Memory Access Conflicts:**

   * Multiple processors accessing shared memory can cause conflicts or bottlenecks.

5. **Scalability:**

   * Algorithms may not scale well with increased number of processors.

---

### **Example: Parallel Merge Sort**

**Input:** `[8, 3, 7, 1]`
**Step 1 (Divide):**
Split data into chunks → `[8, 3]`, `[7, 1]`

**Step 2 (Sort in Parallel):**
Sort each chunk in parallel → `[3, 8]`, `[1, 7]`

**Step 3 (Merge in Parallel):**
Merge `[3, 8]` and `[1, 7]` → Final sorted array: `[1, 3, 7, 8]`

**Issue:** During merging, synchronization and communication are required between threads, which may lead to delays and inefficiencies.

---






