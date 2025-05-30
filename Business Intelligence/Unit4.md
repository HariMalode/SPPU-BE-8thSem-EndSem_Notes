# Data preparation

### Q1.  Discuss the need for data pre-processing and any 2 techniques used.[6]

**Data Pre-processing in Business Intelligence**

**Definition:**
Data pre-processing is the process of cleaning, transforming, and organizing raw data into a usable format before it is analyzed. In Business Intelligence (BI), this step is critical to ensure the data is accurate, consistent, and complete for generating reliable insights.

---

### **Need for Data Pre-processing:**

1. **Improves Data Quality:**
   Real-world data is often incomplete, inconsistent, or contains errors. Pre-processing ensures the data is clean and suitable for analysis.

2. **Enables Accurate Analysis:**
   Good quality data leads to meaningful and accurate insights, helping better decision-making in business.

3. **Handles Missing or Noisy Data:**
   Pre-processing fills missing values, removes duplicates, and smoothens noisy data.

4. **Standardizes Data:**
   Helps convert data into a standard format so it can be used effectively by BI tools.

---

### **Three Common Data Pre-processing Techniques:**

1. **Data Cleaning:**

   * **Purpose:** Fix or remove incorrect, corrupted, or incomplete data.
   * **Example Techniques:** Filling missing values, removing duplicates, correcting typos.
   * **Example:** Replacing missing entries in a sales column with the average sales value.

2. **Data Transformation:**

   * **Purpose:** Convert data into appropriate formats or scales.
   * **Example Techniques:** Normalization, aggregation, encoding categorical variables.
   * **Example:** Converting "Yes"/"No" responses into binary 1/0 values.

3. **Data Reduction:**

   * **Purpose:** Reduce the volume of data without losing relevant information.
   * **Example Techniques:** Dimensionality reduction (e.g., PCA), data cube aggregation, sampling.
   * **Example:** Using Principal Component Analysis (PCA) to reduce variables in customer datasets.

---

### Q2. Explain data validation, incompleteness, noise, inconsistency of quality of input data. [5]

### **1. Data Validation**

**Definition:**
Data validation is the process of ensuring that data is correct, meaningful, and meets the required standards or formats before it is used.
- It ensures that data is accurate, consistent, and relevant for analysis.
- without validation, data can lead to incorrect or misleading results in reports and analysis.

**Purpose:**

* To check accuracy, type, format, and completeness of data.
* To prevent entry of invalid or corrupt data.

**Example:**

* Ensuring a date of birth field contains a valid date and not text like "abcd".

---

### **2. Incompleteness**

**Definition:**
Incompleteness refers to missing data or unfilled fields in a dataset.
- This can happen due to human errors, data collection issues or system failures.
- If not addressed, it can lead to inaccurate analysis and biased results.
- It's impostant to either fill missing ralues or remove incamplete entries during cleaning

**Causes:**

* Human error during data entry.
* Failure in data collection systems.

**Problems:**

* Leads to inaccurate analysis and biased outcomes.

**Example:**

* Customer record missing email or phone number.

---

### **3. Noise**

**Definition:**
Noise refers to random errors or irrelevant data that does not contribute to the analysis.
- It can be caused by errors in data collection or entry.
- Noise can lead to misleading results and reduce the reliability of data.

**Sources:**

* Sensor errors, data entry mistakes, or outdated information.

**Problems:**

* Misleads analysis and reduces data reliability.

**Example:**

* Typing "10000a" instead of "10000" in a sales figure.

---

### **4. Inconsistency**

**Definition:**
Inconsistency occurs when data contradicts itself or uses different formats for the same information.

**Causes:**

* Multiple data sources with different conventions.
* Manual entry differences.

**Problems:**

* Confuses data processing tools and analysts.

**Example:**

* Recording gender as "M/F" in one system and "Male/Female" in another.
* Dates formatted as "01/01/2023" and "01-01-2023" in different systems.

---


### Q3. What is data Transformation? Explain Data Transformation Process in Detail. [5]
(What is data transformation? Why it is needed? Explain at least 3 techniques.)
(Explain data transformation in detail with example. [5])


---

### **Data Transformation**

* **Definition**:
  • Data transformation is the process of converting raw data into a clean and usable format so it can be used in reports, dashboards, etc.
  • This step is crucial because raw data collected from various sources is often in different formats.
  • Data transformation helps by reorganizing, converting, or standardizing data to make it easier to understand and analyze.

---

### **Example**

* Imagine you have a dataset with customer birthdates written as:
  `"January 1, 2025"`, `"01/01/2025"`, `"2025-01-01"`
* These formats are different, so using data transformation, you convert all dates into one standard format like:
  `01-01-2025 (DD-MM-YYYY)`

---
### **Why Data Transformation is Needed?** *(4 Main Points)*

1. **Standardization of Data**
   • Converts data into a consistent format for easier analysis (e.g., dates, currencies).

2. **Data Cleaning and Correction**
   • Removes errors, duplicates, and handles missing or inconsistent values.

3. **Improved Compatibility**
   • Makes data suitable for use in BI tools, databases, or machine learning models.

4. **Better Analysis and Insights**
   • Ensures accurate and meaningful results in reports, dashboards, and predictions.


### **Data Transformation Process**

1. **Data Collection**
   • Data is collected from different sources like databases, CSV files, APIs, etc.

2. **Data Inspection**
   • The collected data is inspected to find inconsistencies and errors.

3. **Applying Transformation Rules**
   • It involves applying rules to clean, standardize, and convert data.
   • This includes removing duplicates, correcting errors, and converting data types.
   • For example, converting dates from different formats to a standard format.
   • This step uses techniques like data cleaning, formatting, and normalization.

4. **Loading Transformed Data**
   • After transformation, the data is stored or loaded into a data warehouse for analysis.

---

### **Techniques of Data Transformation**

1. **Normalization**
   • Scales numeric data to a common range, like 0 to 1.
   • Helps remove the effect of different units (e.g., dollars vs. rupees) and improves performance.
   • Example: Income values like 10,000, 1,00,000 & 5,00,000 can be scaled to 0.1, 1.0, and 5.0 respectively.

2. **Encoding (Categorical to Numerical)**
   • Converts text labels into numbers so they can be analyzed or used in models.
   • Text values like "Yes", "No" cannot be used directly in calculations.
   • Example: Convert "Yes" = 1 and "No" = 0

3. **Data Aggregation**
   • Groups data and calculates summary values like totals, averages, etc.
   • Helps in understanding trends and creating summarized reports.
   • Example: Group sales data by region and calculate total sales per region.

4. **Data Binning**
   • Groups continuous data into intervals or "bins" for easier analysis.
   • Example: Age groups like 0-18, 19-35, 36-60, 60+.
   • Helps in understanding patterns and trends in data.
   • This step uses techniques like binning, bucketing, or clustering.
---


### Q4. Explain data reduction in detial with example. [7]
### Explain following Data reduction technique: Sampling, Feature selection, Principal component analysis. [5]
### What is data reduction? 

* Data reduction is the process of reducing the volume of data while preserving its integrity and meaningful information.
* It helps in improving data storage efficiency, reducing processing time, and making analysis faster and easier.

**Example:**
Imagine a company has customer data with **100 columns**, but **only 10 of them are important** for sales analysis.
By reducing the dataset to just those **10 useful columns**, the analysis becomes:

* **Faster** – less data to process
* **Cleaner** – irrelevant information is removed
* **More focused** – insights are drawn from the most relevant features

### **Data Reduction Techniques Explained**

---

### 1. **Sampling**

- Sampling is a basic yet powerful technique where a subset of data is selected from a large dataset to represent the whole.
- The goal is to perform analysis on this smaller portion which still reflects the overall characteristics of the original data.
- This technique: Saves time , Reduces computational resources , Is especially useful when working with massive datasets
* **Types:**

  * **Random Sampling:** Selecting data points randomly.
  * **Stratified Sampling:** Dividing data into groups (strata) and sampling proportionally.
  * **Systematic Sampling:** Selecting every k-th record from the dataset.

* **Example:**
  From a dataset of 1 million records, taking a random sample of 10,000 records for analysis.

---

### 2. **Feature Selection**

* **Definition:**
  Feature selection involves choosing a subset of relevant features (variables) from the original dataset and discarding irrelevant or redundant ones.

* **Purpose:**
  To improve model accuracy, reduce overfitting, and decrease training time by removing unnecessary features.

* **Methods:**

  * **Filter methods:** Use statistical tests (e.g., correlation, Chi-square).
  * **Wrapper methods:** Use predictive models to evaluate feature subsets.
  * **Embedded methods:** Perform feature selection during model training (e.g., Lasso regression).

* **Example:**
  In a dataset with 50 features, selecting only 10 most important features like age, income, and education for a prediction model.

---

### 3. **Principal Component Analysis (PCA)**

* **Definition:**
  PCA is a mathematical technique that transforms the original features into a smaller number of new variables called principal components, which capture most of the data’s variance.

* **Purpose:**
  To reduce dimensionality while preserving as much information (variance) as possible.

* **How it works:**

  * Computes new axes (principal components) that are linear combinations of original features.
  * These components are uncorrelated and ordered by the amount of variance they explain.

* **Example:**
  Reducing a 20-feature dataset to 3 principal components that explain 90% of the variance for easier visualization and analysis.

---



### Q5. Explain Dimensionality Reduction and Data Compression [6]

### **Dimensionality Reduction**

* **Purpose:** Reduce number of input variables without losing key information.
* **Problems solved:** Overfitting, high storage cost, slower processing.
* **Benefits:** Simplifies data, improves visualization clarity, speeds up ML processes.
* **Types:** Supervised (guided by outcome), Unsupervised (independent of output).
* **Additional:** Removes noise and redundant features to improve accuracy.
* Why it's important: Simplifies data, improves accuracy, speeds up ML processes.
Why it's needed:
* To remove irrelevant, redundant, or highly correlated features.
* To reduce computation time and improve model performance.
* To visualize high-dimensional data in 2D or 3D.
---

### **Data Compression**

* **Purpose:** Reduce physical size of dataset to save space and speed transmission.
* **Focus:** Efficient storage and data handling (different from sampling/feature reduction).
* **Types:** Lossless (original data preserved), Lossy (some info removed).
* **Examples:** ZIP files, ORC formats in data warehousing.
* **Additional:** Reduces network bandwidth usage for faster data transfer.
* Why it's needed:
* Saves storage space.
* Reduces cost of data storage and transmission.
* Speeds up data retrieval and processing.
---


### Q6. Write a short note on data discretization. [5]

### Note on Data Discretization


### **Data Discretization**

* Data discretization is the process of **converting continuous data** (like height, age, etc.) into discrete groups or intervals.
* It is helpful in Business Intelligence (BI) and data mining to **simplify the data**, make patterns easier to detect, and **improve the performance of algorithms**.
* It simplifies data by transforming numeric values into discrete bins, making it easier to analyze and interpret.
* **Example:** Instead of using exact ages like 23, 24, 25…, we can discretize age into ranges such as:

  * 18–25 = Young
  * 26–35 = Adult
  * 36–60 = Senior

---

### **Methods of Data Discretization**

#### 1. **Binning**

* Binning is one of the most common discretization methods.
* It divides continuous variables into a fixed number of equal-sized bins or bins based on data distribution.

**Types of Binning:**

* **a) Equal-width Binning:**

  * The range is divided into equal-sized intervals.
  * *Example:* For values 0 to 100 and 5 bins → bins are 0-20, 21-40, 41-60, 61-80, 81-100.

* **b) Equal-frequency Binning:**

  * Each bin has the same number of data points regardless of the actual range.
  * *Example:* A list of 10 values divided into 2 bins, each containing 5 values.

* **c) Clustering-based Binning:**

  * Similar values are grouped using clustering algorithms like K-means.

---

#### 2. **Histogram-based Discretization**

* A histogram is used to determine how to divide data into intervals based on its distribution.
* Unlike simple binning, this method considers how frequently values occur and sets boundaries at natural gaps in the data.
* *Example:* If most values are between 0-30, the histogram might create more bins in that range and fewer in ranges where data is sparse.

---

### Q7. Explain data exploration in detail with example. [7]

### Data Exploration

**Definition:**
Data exploration is the initial step in the data analysis process where analysts examine and understand the dataset. It involves summarizing the main characteristics of the data, often using visual methods and statistical techniques to detect patterns, spot anomalies, test hypotheses, and check assumptions.

---

### Importance of Data Exploration

* Helps **understand data structure and content**.
* Identifies **missing values, outliers, and errors**.
* Reveals **relationships between variables**.
* Guides **data cleaning, transformation, and modeling decisions**.
* Improves the accuracy and efficiency of subsequent data analysis or machine learning models.

---

### Steps in Data Exploration

1. **Data Collection:**
   Gather the data from different sources like databases, files, or APIs.

2. **Summary Statistics:**
   Calculate measures like mean, median, mode, variance, standard deviation, min, and max for numeric data.

3. **Data Visualization:**
   Use graphs like histograms, scatter plots, box plots, and bar charts to see data distribution and relationships.

4. **Checking Data Quality:**
   Identify missing values, duplicate records, inconsistencies, and outliers.

5. **Correlation Analysis:**
   Analyze how variables relate to each other using correlation coefficients or heatmaps.

---

### Example of Data Exploration


Suppose you are analyzing **Superstore Sales Data** in Power BI. You might begin exploration like this:

1. **Understand Data Columns**

   * Columns like Order ID, Customer Name, Region, Sales, Profit, etc.

2. **Check Summary Statistics**

   * Average Sales = 50,000/-
   * Maximum Profit = 15,000/-

3. **Detect Missing Values**

   * You notice some rows have missing **Shipping Date**.

4. **Visualize Patterns**

   * You create a bar chart showing that the **Technology** category has the highest sales.
   * A scatter plot shows that higher sales often lead to higher profits, but not always.

5. **Spot Outliers**

   * A few orders have huge sales amounts but zero profit — this could indicate an error.

---

### Q8. What is a Contingency Table? What is Marginal Distribution? Justify with suitable example. [7]
### Contingency Table
* A contingency table is a **cross-tabulation** of two or more categorical variables.
* It shows the **frequency distribution** of one variable based on the categories of the other variable.
* It helps in understanding the relationship between two variables.
* Contingency tables are often used in **statistical analysis** and **data analysis**, reporting, and **decision-making**, identifying patterns, and **making predictions**.

* **Example:**
  * A contingency table of **Gender** (Male/Female) and **Smoking Status** (Smoker/Non-Smoker) can show the distribution of smokers and non-smokers based on gender.
  * This table can help in understanding if there is a relationship between gender and smoking status.

### **Example:**

Suppose a company wants to analyze the relationship between **Gender** (Male/Female) and **Product Preference** (Electronics/Clothing).
The data might be arranged in a **2×2 contingency table** as follows:

| Gender    | Electronics | Clothing | Total  |
| --------- | ----------- | -------- | ------ |
| Male      | 20          | 10       | 30     |
| Female    | 15          | 25       | 40     |
| **Total** | **35**      | **35**   | **70** |

---

### **Interpretation:**

* This table clearly shows how preferences differ between genders:

  * More **males** prefer **Electronics**.
  * More **females** prefer **Clothing**.

---

### **Marginal Distribution**
* The marginal distribution is the distribution of a single variable, ignoring the other variables.
* It shows the total count or percentage of each category of a variable, regardless of the other variables.
* It is calculated by summing the values in each row or column of the contingency table.
* Marginal distributions are often used to compare the distribution of a variable across different categories of another variable.
* Marginal distribution help summarize data give an overview of how categories of single vaxiable are spread across dataset
* These totals are usually found in margins (edges) of table - that's wby it's called "marginal"

### **Example:**

Using the earlier contingency table:

| Gender    | Electronics | Clothing | **Total** |
| --------- | ----------- | -------- | --------- |
| Male      | 20          | 10       | **30**    |
| Female    | 15          | 25       | **40**    |
| **Total** | **35**      | **35**   | **70**    |

* **Marginal distribution of Gender**:

  * Male: 30 / 70 = **42.86%**
  * Female: 40 / 70 = **57.14%**

* **Marginal distribution of Product Preference**:

  * Electronics: 35 / 70 = **50%**
  * Clothing: 35 / 70 = **50%**

---

### **Use:**

* Helps in understanding the **overall distribution** of each variable.
* Useful in **probability calculations** and identifying **general trends**.

----

### Q9.Explain univariate, bi variate and multivariate analysis with example and applications.

### **Univariate Analysis**

* Univariate analysis involves analyzing **one variable at a time**.
* The goal is to **summarize**, **describe**, and **find patterns** in a single dataset column.
* The main. aim is to understand basic chaarecteristics of that single variable, like mean, mode, minimum, maximum, etc
* It is used to understand the **distribution** of a single variable, detect **outliers**, and **prepare data** for further analysis.

### ✅ **Techniques Used:**

* **For Categorical Data**:

  * Frequency tables
  * Bar charts
  * Pie charts

* **For Numerical Data**:

  * Mean, Median, Mode
  * Range, Variance, Standard Deviation
  * Histograms, Box plots

---

### ✅ **Example:**

Suppose we are analyzing the **"Sales"** column of a retail dataset.

* Average Sales = ₹5,000
* Minimum = ₹100
* Maximum = ₹20,000
* Histogram shows most sales are between ₹4,000–₹6,000
* A box plot highlights one outlier at ₹20,000

This gives a quick overview of **how sales values are spread**.

---

### ✅ **Applications:**

1. **Business Intelligence**:
   Understand product sales, customer demographics, etc.

2. **Quality Control**:
   Detect anomalies in a manufacturing process.

3. **Marketing**:
   Analyze customer age, income, or purchase frequency.

4. **Healthcare**:
   Track patient vitals like temperature, blood pressure trends.

5. **Education**:
   Analyze student scores or attendance rates.

---

### Q10. What is Bivariate analysis? Why it is important. Discuss the different types with example? [6]

### **Bivariate Analysis**

* Bivariate analysis involves analysis of **two variables** to understand **relationship or association** between them.
* This analysis is useful in **comparison & correlation studies**.
* Techniques like **scatter plots**, **correlation coefficients**, etc. are commonly used.

---

### **Example:**

A business may want to see if there's a connection between **marketing spend & sales revenue**.

* If spending increases and sales also increase,
  → the two variables have a **positive relationship**.

---

### **Applications:**

Marketing, Education, Business

---

### **Need / Importance of Bivariate Analysis:**

* Bivariate analysis is important because it **reveals how two variables interact**.
* This helps **businesses & researchers** make better decisions.
* It also helps in identifying **cause-effect relationships**, **predictive patterns**, etc.

---

### **Types of Bivariate Analysis:**

#### i) Numerical vs Numerical

* In this, **both variables are numbers**, and we study their relationship using tools like **scatter plots** or **correlation**.
* **Eg:** Height vs Weight.

#### ii) Numerical vs Categorical

* One variable is **numeric** & the other is **categorical**.
* We compare numeric values across different groups using **bar charts / box plots**.
* **Eg:** Salary across job roles.

#### iii) Categorical vs Categorical

* Both variables are **categories**.
* We check their association using **contingency tables** or **chi-square tests**.
* **Eg:** Gender vs Product Preference

---

### Q11. Explanation of **Multivariate Analysis**:


### **Multivariate Analysis**

* **Definition**:
  Multivariate Analysis is the examination of **more than two variables** simultaneously to understand **relationships**, **patterns**, and **influences** among them.

* **Purpose**:
  It helps in analyzing **complex data sets** where multiple variables may impact each other or the outcome.

---

### **Example:**

In a **customer satisfaction survey**, you might analyze:

* Age
* Income
* Frequency of purchase
* Satisfaction score

All at once to see which factors most influence satisfaction.

### **Applications:**

* **Marketing** – Customer segmentation and targeting
* **Healthcare** – Diagnosing diseases from multiple symptoms
* **Finance** – Risk assessment involving multiple indicators
* **Education** – Analyzing performance based on different factors
* **Social Sciences** – Studying behavior with many influencing variables

---

### **Advantages:**

* Handles **complex relationships**
* Identifies **hidden patterns**
* Improves **decision-making**
* Supports **predictive modeling**

---

### Q12. Difference between univariate, Bivariate, Multivariate analysis. [5]

### **1. Number of Variables:**

* **Univariate**: One variable
* **Bivariate**: Two variables
* **Multivariate**: More than two variables

---

### **2. Purpose:**

* **Univariate**: To describe or summarize a single variable
* **Bivariate**: To identify relationships between two variables
* **Multivariate**: To understand patterns and relationships among multiple variables

---

### **3. Example:**

* **Univariate**: Distribution of salaries
* **Bivariate**: Salary vs years of experience
* **Multivariate**: Salary vs experience vs education level

---

### **4. Techniques Used:**

* **Univariate**: Mean, median, histograms
* **Bivariate**: Correlation, scatter plots
* **Multivariate**: Multiple regression, PCA, clustering

---

### **5. Type of Analysis:**

* **Univariate**: Descriptive analysis
* **Bivariate**: Comparative or relational analysis
* **Multivariate**: Complex, predictive, or explanatory analysis

---

### **6. Complexity:**

* **Univariate**: Least complex
* **Bivariate**: Moderate complexity
* **Multivariate**: Most complex

---

### Q13. Compute Mean, Median and Mode for following data [7]

| Class Interval | 10–15 | 15–20 | 20–25 | 25–30 | 30–35 | 35–40 | 40–45 | 45–50 |
| -------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Frequency (f)  | 2     | 28    | 125   | 270   | 303   | 197   | 65    | 10    |


Here's your text cleaned up and formatted clearly:

---

### 1) **Mean**

**Formula:**

$$
\text{Mean} = \frac{\sum f \cdot x}{\sum f}
$$

where $x$ is the midpoint of each class.

| Class Interval | Midpoint (x) | Frequency (f) | $f \times x$ |
| -------------- | ------------ | ------------- | ------------ |
| 10–15          | 12.5         | 2             | 25           |
| 15–20          | 17.5         | 28            | 490          |
| 20–25          | 22.5         | 125           | 2812.5       |
| 25–30          | 27.5         | 270           | 7425         |
| 30–35          | 32.5         | 303           | 9847.5       |
| 35–40          | 37.5         | 197           | 7387.5       |
| 40–45          | 42.5         | 65            | 2762.5       |
| 45–50          | 47.5         | 10            | 475          |

$$
\sum f = 1000, \quad \sum f \cdot x = 30225
$$

$$
\text{Mean} = \frac{30225}{1000} = 30.23
$$

---

### 2) **Median**

* Total frequency $N = 1000$
* $N/2 = 500$

Find the class where cumulative frequency ≥ 500:

| Class Interval | Frequency (f) | Cumulative Frequency (CF) |                |
| -------------- | ------------- | ------------------------- | -------------- |
| 10–15          | 2             | 2                         |                |
| 15–20          | 28            | 30                        |                |
| 20–25          | 125           | 155                       |                |
| 25–30          | 270           | 425                       |                |
| 30–35          | 303           | 728                       | ← Median Class |
| 35–40          | 197           | 925                       |                |
| 40–45          | 65            | 990                       |                |
| 45–50          | 10            | 1000                      |                |

Median class = **30–35**

**Formula:**

$$
\text{Median} = L + \left(\frac{\frac{N}{2} - CF}{f}\right) \times h
$$

Where:

* $L = 30$ (lower limit of median class)
* $CF = 425$ (cumulative frequency before median class)
* $f = 303$ (frequency of median class)
* $h = 5$ (class width)

Calculate:

$$
\text{Median} = 30 + \left(\frac{500 - 425}{303}\right) \times 5 = 30 + \left(\frac{75}{303}\right) \times 5
$$

$$
= 30 + 0.2475 \times 5 = 30 + 1.2376 = 31.24
$$

---

### 3) **Mode**

Modal class = class with highest frequency = **30–35** (frequency = 303)

**Formula:**

$$
\text{Mode} = L + \frac{(f_1 - f_0)}{(2f_1 - f_0 - f_2)} \times h
$$

Where:

* $L = 30$ (lower limit of modal class)
* $f_1 = 303$ (frequency of modal class)
* $f_0 = 270$ (frequency of previous class)
* $f_2 = 197$ (frequency of next class)
* $h = 5$ (class width)

Calculate:

$$
\text{Mode} = 30 + \frac{303 - 270}{2 \times 303 - 270 - 197} \times 5 = 30 + \frac{33}{606 - 467} \times 5
$$

$$
= 30 + \frac{33}{139} \times 5 = 30 + 0.237 \times 5 = 30 + 1.187 = 31.19
$$

---

**Final values:**

* Mean = 30.23
* Median = 31.24
* Mode = 31.19

---

### Q14. Define dirty data. What are the reasons of dirty data. [6]

**Dirty Data:**

* Dirty data refers to data that contains errors, inconsistencies, inaccuracies, or is incomplete.
* It reduces the quality and reliability of data analysis and decision-making.
* Dirty data can lead to incorrect conclusions and poor decision-making.
* It can also lead to wasted resources and time in data cleaning and correction.

**Reasons for Dirty Data:**

1. **Missing values** – Data fields left blank or not recorded.
2. **Duplicate records** – Repeated entries of the same data.
3. **Inconsistent data** – Different formats or representations for the same information.
4. **Incorrect data** – Errors in data entry or measurement.
5. **Outdated data** – Data that is no longer current or valid.
6. **Noise or irrelevant data** – Data not relevant to the analysis or with random errors.

---

### Q15. Explain the working of binning with suitable example. [6]
**Binning:**
* Binning is a data preprocessing technique used to convert continuous numerical data into discrete categories or bins.
* It helps in simplifying the data and reducing the complexity of analysis.
* Binning can be used to group similar values together and reduce the number of unique values in a dataset.
**Example:**
* Suppose we have a dataset of ages of people in a population. The ages range from 0 to 100.
* We can divide the ages into bins of 10 years each, such as 0-9, 10-19, 20-29, and so on.
* This will reduce the number of unique values in the dataset from 101 to 11.
* The binning process can be done using various methods, such as equal-width binning or equal-frequency binning.
* Equal-width binning divides the range of values into equal-sized bins, while equal-frequency binning divides the data into bins with approximately the same number of observations.


### Working of Binning:

1. **Divide data range into bins**
   The continuous range of values is divided into intervals (bins) either of equal width or based on data distribution.

2. **Assign data points to bins**
   Each data value is placed into one of the bins according to its value.

3. **Replace original values**
   The data points in each bin can be replaced by a representative value, such as the bin mean, median, or boundary values.

---

### Example:

Suppose you have ages of 10 people:
`18, 20, 22, 25, 27, 30, 35, 40, 42, 45`

* Divide into 3 equal-width bins:

  * Bin 1: 18–26
  * Bin 2: 27–35
  * Bin 3: 36–45

* Assign values:

  * Bin 1: 18, 20, 22, 25
  * Bin 2: 27, 30, 35
  * Bin 3: 40, 42, 45

* Replace by bin mean (for Bin 1): (18 + 20 + 22 + 25)/4 = 21.25
  Similarly for other bins.

This groups continuous ages into categories making data simpler to analyze.
