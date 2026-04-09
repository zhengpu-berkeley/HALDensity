Here is the clean and correct completion, aligned with your theory and notation:

⸻

For a data structure with sample size n, dimension d, and assuming k-th order differentiability, the basis functions are given by:

1. Parametric part

x_1^{a_1}x_2^{a_2}\cdots x_d^{a_d}, \qquad a_j \in \{0,1,\dots,k\}.

So we have
(k+1)^d
basis functions.

⸻

2. Sectional part

For each nonempty section s \subset [d], define its complement s^c.
Then for each:
• multi-index
a = (a*j)*{j \in s^c}, \qquad a_j \in \{0,1,\dots,k\},
• and each projected data point
u_s = X_j(s), \qquad j = 1,2,\dots,n,

the basis functions are
\left(\prod*{l \in s^c} x_l^{a_l}\right)
\left(\prod*{l \in s} (x*l - u_l)^k*+\right).

So for each section s, we have
(k+1)^{|s^c|} \cdot n
basis functions.

Summing over all nonempty s \subset [d], the total number of sectional basis functions is
n \sum\_{\emptyset \neq s \subset [d]} (k+1)^{|s^c|}
=
n\big((k+2)^d - (k+1)^d\big).

⸻

3. Nonparametric part

This corresponds to the full section s = [d], i.e., no polynomial part.

For each observation j = 1,2,\dots,n, the basis functions are
(x*1 - u*{j1})^k*+ (x_2 - u*{j2})^k*+ \cdots (x_d - u*{jd})^k\_+.

So we have
n
basis functions.

⸻

Final combined count

Total basis size:
\boxed{
(k+1)^d + n\big((k+2)^d - (k+1)^d\big).
}

⸻
