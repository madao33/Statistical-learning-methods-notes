# 第４章　朴素贝叶斯法

> 朴素贝叶斯(naive Bayes)法是基于贝叶斯定理与特征条件独立假设的方法.

## 4.1 朴素贝叶斯法的学习与分类

### 4.1.1 基本方法

设输入空间 $x\subseteq R^n$, 输出空间 $Y = {c_1, c_2, c_3, ..., c_k}$, 输入特征向量 $x \in X$, 输出为类标记(class label) $y \in Y$, $P(X,Y)$表示联合概率分布,训练数据集为:

$$T = {(x_1, y_1), (x_2, y_2),...,(x_N, y_N)}$$

#### 先验概率分布

$$P(Y=c_k), k=1,2,...,K$$

#### 条件概率分布

$$p(X=x|Y=c_k)=P(X^{(1)} = x^{(1)},...,X^{(n)}=x^{(n)}), k=1,2,...,k$$

条件概率分布有指数级的参数，假设 $x^{(j)}$　有 $S_j$ 个，$Y$ 可能值有　$k$ 个，那么参数个数为 $k \prod_{j=1}^n S_j $.

#### 条件独立性假设

朴素贝叶斯对条件概率分布做了条件独立性的假设

$$P(X=x|Y=c_k)=P(X^{(1)} = x^{(1)},...,X^{(n)}=x^{(n)}) =\prod_{j=1}^n P(X^{(i)}=x^{(j)}|Y=c_k)$$

* > 朴素贝叶斯实际是学习到生成数据的机制，所以属于生成模型
* > 条件独立性假设等于说在类确定的条件下都是条件独立的，这个假设使得朴素贝叶斯法变的简单，但有时会牺牲一定的分类准确率

#### 后验概率

根据贝叶斯定理

$$ P(Y=c_k|X=x) = \frac{P(Y=c_k|X=x)P(Y=c_k)}{\sum_k P(Y=c_k|X=x)P(Y=c_k)}$$

结合条件独立性假设，得到

$$P(Y=c_k|X=x) = \frac{P(Y=c_k)\prod \limits_{j=1}^n P(X^{(i)}=x^{(j)}|Y=c_k)}{\sum_k P(Y=c_k) \prod \limits_{j=1}^n P(X^{(i)}=x^{(j)}|Y=c_k)}$$

所有贝叶斯分类器表示为

$$y=f(x)=arg \max_{c_k}\frac{P(Y=c_k)\prod \limits_{j=1}^n P(X^{(i)}=x^{(j)}|Y=c_k)}{\sum_k P(Y=c_k) \prod \limits_{j=1}^n P(X^{(i)}=x^{(j)}|Y=c_k)}$$

贝叶斯分类计算上式不同$c_k$取值的最大值，而对于所有的 $c_k$ 分母值都相同，故简化为下式

$$y=arg \max_{c_k}P(Y=c_k)\prod_{j=1}^n P(X^{(i)}=x^{(j)}|Y=c_k)$$

### 4.1.2 后验概率最大化的含义

类似于期望风险最小化

选择 $0-1$ 损失函数

$$L(Y,f(x))=
\begin{cases}
1,Y \not ={f(x)} \\
0,Y=f(x)
\end{cases}$$

期望风险函数为

$$R_exp(f)=E[L(Y,f(X))]$$

根据联合分布，得到条件期望

$$R_{exp}(f)=Ex \sum_{k=1}^K[L(c_k,f(X))]P(c_k|X)$$

使期望风险最小化，得到后验概率最大化准则

$$f(x)=arg \max_{c_k}P(c_k|X=x)$$

## 4.2 朴素贝叶斯的参数估计

### 4.2.1 极大似然估计法

估计类条件概率的一种常用策略是先假定其具有某种确定的概率分布形式，再基于训练样本对概率分布的参数进行估计。

记关于类别 $c$ 的类条件概率为 $P(x|c)$, 为求得参数向量，记为 $P(x|\theta_c)$ 

令 $D_c$ 表示训练集 $D$ 中第 $c$ 类样本组成的集合

$$ P(D_c|\theta_c)=\sum_{x \in D_c}P(x|\theta_c)$$

为避免连乘操作导致下溢，通常使用对数似然(log-likelihood)

$$LL(\theta_c)=log P(D_c|\theta_c)=\sum_{x \in D_c}\log P(x|\theta_c)$$

极大似然估计为

$$\hat{\theta_c}=\arg \max_{\theta_c} LL(\theta_c)$$

> 这种参数化的方法能使类条件概率估计变的相对简单，但是估计结果的准确性严重依赖于所假设的概率分布形式是否符合潜在的真实数据分布.

### 4.2.2 学习与分类算法

#### 算法4.1(朴素贝叶斯算法(naive Bayes algorithm))

输入:训练数据 $T = {(x_1, y_1), (x_2, y_2),…,(x_N, y_N)}$，其中 $x_i=(x^{(1)}, x^{(2)},…，x^{(n)})^T$, $x_i^{(j)}$ 是第 $i$ 个样本的第 $j$ 个特征, $y_i \in \{c_1, c_2,…,c_k \}$;实例 $x$;
输出:实例 $x$ 的分类;

(1)计算先验概率及条件概率

$$P(Y=c_k)=\frac{\sum \limits_{i=1}^N I(y_i=c_k)}{N},k=1,2,……,K$$

$$p(x^{(j)}=a_{jl}|Y=c_k)=\frac{\sum \limits_{i=1}^N I(x^{(j)}=a_{jl}, y_i=c_k)}{\sum \limits_{i=1}^N I(y_i=c_k)}$$

$$(j=1,2,……,n;\quad l=1,2,……,S_j;\quad k=1,2,……,K)$$

(2)对于给定的实例 $x=(x^{(1)},x^{(2)},……,x^{(n)})^T$, 计算

$$P(Y=c_k)\prod \limits_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k), k=1,2,……,K$$

(3) 确定实例 $x$ 的类

$$y=\arg \max_{c_k}P(Y=c_k) \prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)$$

### 4.2.3 贝叶斯估计

> 用极大似然估计可能会出现估计的概率为0的情况, 这是可以采用贝叶斯估计

$$P_\lambda(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum \limits_{i=1}{N}I(x_i^{(j)}=a_{ji},y_i = c_k)+\lambda}{\sum \limits_{i=1}^N I(y_i=c_k)+S_j \lambda}$$

式中，$\lambda>0$，当

* $\lambda=0$ 时就是极大释然估计
* 常取 $\lambda=1$, 这时称为拉普拉斯平滑(Laplacian smoothing)

先验概率的贝叶斯估计是

$$P_\lambda(Y=c_k)=\frac{\sum \limits_{i=1}^N I(y_i=c_k)+\lambda}{N+K\lambda}$$

