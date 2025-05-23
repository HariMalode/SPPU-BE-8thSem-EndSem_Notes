# Analytical Modeling of Parallel Programs

### Q1. Explain various sources of overhead in parallel systems. [7]


- In parallel computing, a task is divided into smaller subtasks that are executed simultaneously on multiple processors. Ideally, this should lead to faster execution. However, in reality, we do not always achieve perfect performance due to various overheads associated with parallel execution.

- In high-performance computing, **overhead** refers to the extra time and resources required to manage parallel tasks. The main sources of overhead in parallel systems are:

- Minimizing these overheads is essential for achieving high efficiency and scalability in parallel computing systems.

1. **Communication Overhead**

   * Time spent in data exchange between processors.
   * Increases with the number of processors and distance between them.

2. **Synchronization Overhead**

   * Time taken to coordinate tasks (e.g., barriers, locks).
   * Necessary to ensure correct execution order and data consistency.

3. **Load Imbalance**

   * When tasks are not evenly distributed across processors.
   * Some processors may be idle while others are overloaded.

4. **Task Scheduling Overhead**

   * Time taken by the system to assign and manage tasks among processors.

5. **Resource Contention**

   * Competition for shared resources like memory or I/O channels.
   * Causes delays and reduced performance.

6. **Parallelization Overhead**

   * Extra effort required to break a problem into parallel tasks.
   * Includes time for setting up and managing the parallel structure.

7. **Latency and Bandwidth Limitations**

   * Delay in data transfer and limited data throughput can affect speed.
   * More noticeable in distributed systems.

8. **Software Overhead**

   * Inefficiencies in compilers, libraries, or middleware used for parallelism.
   * Affects performance despite good hardware.

---


### Q2. Explain different performance Metrics for Parallel Systems. [7]

Here’s the complete exam-ready answer with a definition at the beginning:

---

### **Performance Metrics for Parallel Systems**

**Definition:**
Performance metrics are quantitative measures used to evaluate the effectiveness, efficiency, and scalability of parallel systems. They help in understanding how well a parallel system performs compared to a sequential one and how it behaves as the number of processors increases.

---

### **Different Performance Metrics:**

1. **Speedup (S)**

   * Indicates how much faster a parallel program runs compared to its sequential version.
   * **Formula:**

     $$
     S = \frac{T_1}{T_p}
     $$

     Where $T_1$ is the execution time on a single processor, and $T_p$ is the execution time on $p$ processors.

---

2. **Efficiency (E)**

   * Measures how efficiently the processors are being utilized.
   * **Formula:**

     $$
     E = \frac{S}{p} = \frac{T_1}{p \cdot T_p}
     $$

     Efficiency is highest (close to 1) when all processors contribute equally.
     P = Number of processors
     T1 = Time taken by a single processor
     Tp = Time taken by p processors

---

3. **Scalability**

   * Describes the ability of a parallel system to maintain efficiency as the number of processors increases.
   * A highly scalable system can handle a growing workload effectively.

---

4. **Throughput**

   * Refers to the number of tasks or jobs completed per unit time.
   * Higher throughput means better overall system performance.

---

5. **Latency**

   * Time taken to complete a single operation or task.
   * Lower latency is preferred in high-performance systems.

---

6. **Parallel Overhead**

   * Extra time required for coordination between processors (e.g., communication, synchronization).
   * Reducing overhead improves overall performance.

---

7. **Amdahl’s Law**

   * Provides the theoretical limit on speedup based on the portion of the task that is parallelizable.
   * **Formula:**

     $$
     S = \frac{1}{(1 - P) + \frac{P}{N}}
     $$

     Where $P$ is the parallelizable fraction, and $N$ is the number of processors.

---

### **Conclusion:**

These performance metrics are essential for analyzing, designing, and optimizing parallel systems to achieve high speed, efficiency, and scalability.

---
###  Q3. Explain amdahl’s and gustafson’s law. [4]


**Definition:**
- Amdahl’s Law describes the **maximum possible speedup** of a program using multiple processors, based on the **portion of the program that can be parallelized**.
- Amdahl’s Law assumes that a program has two parts: one that **can be parallelized** (runs on multiple processors) and one that **must be executed sequentially** (runs on a single processor).
- The law states that **no matter how many processors we use, the sequential part of the program will always limit the overall speedup** we can achieve.


**Formula:**

$$
S = \frac{1}{(1 - P) + \frac{P}{N}}
$$

Where:

* $S$ = Speedup
* $P$ = Fraction of the program that can be parallelized
* $N$ = Number of processors

**Explanation:**

* As the number of processors (**p**) increases, the **parallel part** of the program gets faster, but the **sequential part** remains the same.
* So, after a certain point, **adding more processors doesn't significantly improve performance**.
For example, if **10% of the program is sequential** (i.e., $f = 0.1$), then **even with infinite processors**, the maximum speedup is limited to **10×**.

**Example:**
If only 80% of a program can be parallelized ($P = 0.8$) and we use 4 processors:

$$
S = \frac{1}{(1 - 0.8) + \frac{0.8}{4}} = \frac{1}{0.2 + 0.2} = 2.5
$$

---

### **Gustafson’s Law**

**Definition:**
- Gustafson's Law states that instead of Reeping problem size fixed, we can increase size of problemes as we increase no. of processors.
- Gustafson’s Law assumes that the problem size increases with the number of processors, allowing more work to be done in parallel.
-  According to Gustafson's law, speedup is not limited by small sequential part because overall problem size increases and parallel part dominates.


**Formula:**

$$
S = N - (1 - P)(N - 1)
$$

Where:

* $S$ = Scaled speedup
* $P$ = Parallel fraction
* $N$ = Number of processors


**Example:**
If $P = 0.9$ and $N = 4$:

$$
S = 4 - (1 - 0.9)(4 - 1) = 4 - 0.1 \times 3 = 4 - 0.3 = 3.7
$$

---

### Q4. What is granularity? What are effects of granularity on performance of parallel systems? [7]
(Show effect of granularity on performance with addition of n numbers on
p processing elements. [6])

### **What is Granularity in Parallel Computing?**

**Granularity** refers to the **amount of computation** performed between two communication or synchronization events in a parallel program.
It defines the **size of tasks** into which a program is divided.

---

### **Types of Granularity:**

1. **Fine-Grained:**
  Each processor adds a small number of elements (e.g., 1 or 2).
  → **Frequent communication** between processors to combine results.
  → **High overhead**, **low performance**.

2. **Coarse-Grained:**
  Each processor adds a large block of numbers (e.g., $n/p$ elements).
  → **Less communication**, results combined only once.
  → **Low overhead**, **better performance**.
---

### **Effects of Granularity on Performance:**

1. **Communication Overhead:**

   * Fine granularity leads to more frequent communication, increasing overhead and reducing performance.
   * This frequent communication takes time & consumessystem resources like Bandwidth & CPU cycles. In contrast, coarse- grained systems communicate less often, reducing overhead.

2. **Load Balancing:**

   * Coarse granularity can lead to **load imbalance** if tasks are not evenly distributed among processors.
   * Fine granularity allows better load distribution.


3. **Processor Utilization:**

   * With coarse granularity, processors stay busy longer between communications, improving utilization.
   * With fine granularity, processors wait longer between tasks, reducing utilization.

4. **Performance Trade-off:**

   * An **optimal granularity** balances computation and communication, minimizing overhead while keeping processors efficiently used.

---

### Q5. Comment on “Scalability of Parallel Systems”. [4]

Here is a well-structured and properly worded **exam-ready answer** for:

---

### **Scalability in Parallel Systems**

**Definition:**
- Scalability refers to a parallel system’s ability to maintain efficiency as the number of processors or problem size increases.
- It measures how effectively a parallel program or system can utilize **increasing computing resources (like processors)** to improve its performance.

---

### **Key Idea:**

* A **highly scalable system** continues to show performance improvements (e.g., reduced execution time) as more processors are added.
* A **poorly scalable system** sees little or no improvement beyond a certain number of processors.

---

### **Challenges in Achieving Perfect Scalability:**

Achieving ideal scalability in real-world systems is difficult due to the following issues:

1. **Communication Overhead:**

   * Frequent communication between processors consumes time in sending and receiving data.
   * This overhead reduces the benefits of parallel execution.
   * **Solution:** Minimize inter-processor communication where possible.

2. **Load Imbalance:**

   * If some processors are assigned more work than others, they take longer to finish while others remain idle.
   * This leads to poor utilization and reduced speedup.
   * **Solution:** Use **dynamic scheduling** and divide the work into **smaller sub-tasks** for better distribution.

3. **Amdahl’s Law (Serial Bottleneck):**

   * According to Amdahl’s Law, if any portion of a program is inherently sequential, it will limit the overall speedup, regardless of the number of processors.
   * **Solution:** Use techniques like **pipelining** and **algorithm redesign** to reduce serial components.

---

### **Design Considerations for Better Scalability:**

* Choose scalable algorithms.
* Optimize memory architecture for parallel access.
* Balance workloads effectively.

---

### Q6. Explain “Scaling Down (downsizing)” a parallel system with example


### **Scaling Down (Downsizing) in Parallel Systems**

**Definition:**
Scaling down, or **downsizing**, refers to the process of **reducing the number of processors** in a parallel system while still attempting to maintain **acceptable performance**.

---

### **Purpose and Application:**

* Often applied when **hardware availability is limited** or **cost reduction** is a priority.
* The **same workload** is distributed among **fewer processors**, increasing the computational burden per processor.
* It helps evaluate how efficiently an algorithm performs under **resource-constrained conditions**.

---

### **Impact on Performance:**

* May result in **longer execution times**, but should not cause major failures.
* A well-designed algorithm/system should continue to function **correctly and efficiently**, even with fewer processors.

---

### **Example:**

Suppose an application is designed to run on **16 processors** to simulate a large scientific model in **1 hour**.
Due to budget cuts or hardware issues, the system is scaled down to **4 processors**.

* Now, each processor handles **4 times the workload**.
* The overall execution time increases to, say, **3.5 hours**.
* If the system still runs correctly and efficiently, the **algorithm is considered to scale well** under reduced resources.

---

### Q7. Explain Minimum Execution Time and Minimum Cost Optimal Execution Time. [4]


### **1. Minimum Execution Time:**

* It refers to the **shortest possible time** required to complete a parallel task using **any number of processors**.
* The goal is to **maximize speedup**, even if many processors are used (which may increase cost).
* It focuses only on **time efficiency**, not on cost or resource usage.

**Example:**
If a task takes 100 seconds on 1 processor, and only 10 seconds on 20 processors, then **10 seconds is the minimum execution time**.

---

### **2. Minimum Cost Optimal Execution Time:**

* It refers to the execution time achieved when the task is completed using **the least cost in terms of total processor-time product**.
* The focus is on **efficiency and economy**—getting good speedup **without wasting resources**.
* It ensures that we use **just enough processors** to achieve a good execution time **without increasing total cost unnecessarily**.

**Formula:**
Cost = Number of Processors × Execution Time
Minimum cost-optimal execution time occurs when **cost is minimized** while **speedup remains acceptable**.

**Example:**
If 10 processors finish a job in 12 seconds → Cost = 120
But 15 processors finish it in 10 seconds → Cost = 150
Then, **12 seconds is the minimum cost-optimal time**, not 10 seconds.

---

### **Conclusion:**

* **Minimum Execution Time** aims for **speed**, regardless of cost.
* **Minimum Cost Optimal Execution Time** aims for a balance between **speed and efficiency**, minimizing overall computational cost.

---

### Q8. Explain Asymptotic Analysis of Parallel Programs

Here’s an exam-ready explanation of **Asymptotic Analysis of Parallel Programs**:

---

### **Asymptotic Analysis of Parallel Programs**

**Definition:**
- Asymptotic analysis is the study of a program’s **performance (time or space)** as the **input size (n) becomes very large**.
- In parallel computing, it helps understand how a parallel program **scales** and behaves with increasing problem size and number of processors.

---

### **Purpose:**

* To **evaluate efficiency** and **scalability** of algorithms as system resources and input grow.
* Focuses on the **growth rate** of execution time rather than exact time values.

---

### **Key Components:**

1. **Work (Total Time)**:

   * The total number of operations done across all processors.
   * Denoted as **T₁(n)** – time taken by the best serial algorithm.

2. **Span (Critical Path Length):**

   * The time taken by the **longest dependent chain** of operations (assuming infinite processors).
   * Denoted as **T∞(n)**.

3. **Parallel Time:**

   * Time taken by the parallel algorithm using **p processors** – denoted as **Tp(n)**.

4. **Speedup:**

   * Ratio of serial time to parallel time:
     **Speedup = T₁(n) / Tp(n)**

5. **Efficiency:**

   * Efficiency = Speedup / Number of processors
   * Measures how well processors are being utilized.

---

### **Asymptotic Notations Used:**

* **Big O (O):** Upper bound of performance (worst case)
* **Omega (Ω):** Lower bound (best case)
* **Theta (Θ):** Tight bound (average case)

---

### Q9. Explain parallel Matrix —Matrix multiplication algorithm with example? [7]


### **Parallel Matrix–Matrix Multiplication Algorithm**

**Definition:**
Matrix–Matrix multiplication involves computing the product of two matrices **A (m × n)** and **B (n × p)** to produce matrix **C (m × p)**. In **parallel computing**, this task is divided among multiple processors to speed up execution.

---

### **Algorithm Steps:**

1. **Input:**
   Matrices A (m×n), B (n×p)
   Output: Matrix C (m×p)

2. **Initialization:**
   Assign matrix blocks or rows to available processors.

3. **Computation:**
   Each processor calculates a **portion** of the resulting matrix C.

   Formula for each element:

   $$
   C[i][j] = \sum_{k=1}^{n} A[i][k] \times B[k][j]
   $$

4. **Parallelization Strategy:**

   * **Row-wise:** Each processor handles one or more **rows** of matrix C.
   * **Block-wise:** Matrices are divided into submatrices and assigned to processors.

5. **Synchronization:**
   Combine partial results from all processors to form the final matrix C.

---

### **Example:**

Let A and B be 2×2 matrices:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

Let’s assume we have 4 processors (P1, P2, P3, P4), each computing one element of matrix C.

**Step 1: Compute C = A × B**

$$
C[0][0] = 1×5 + 2×7 = 19 \quad \text{(P1)}  
$$

$$
C[0][1] = 1×6 + 2×8 = 22 \quad \text{(P2)}  
$$

$$
C[1][0] = 3×5 + 4×7 = 43 \quad \text{(P3)}  
$$

$$
C[1][1] = 3×6 + 4×8 = 50 \quad \text{(P4)}  
$$

**Final Result:**

$$
C = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

---

### **Advantages:**

* Faster computation for large matrices.
* Efficient use of multi-core or distributed systems.

---

