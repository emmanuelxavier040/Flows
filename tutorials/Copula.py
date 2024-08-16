import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import scipy.stats as stats

# def plot_cdf_pdf_plotly():
#     # Generate 10000 evenly distributed values from -4 to 4
#     x = np.linspace(-4.0, 4.0, 10000)
#
#     # Compute their Probability Densities and Cumulative Distributions
#     pdf = stats.norm.pdf(x)
#     cdf = stats.norm.cdf(x)
#
#     fig = make_subplots(rows=1, cols=2, subplot_titles=("PDF", "CDF"))
#
#     fig.add_trace(
#         go.Scatter(x=x, y=pdf),
#         row=1, col=1
#     )
#     fig.update_xaxes(title_text="x", row=1, col=1)
#     fig.update_yaxes(title_text="f(x)", row=1, col=1)
#
#     fig.add_trace(
#         go.Scatter(x=x, y=cdf),
#         row=1, col=2
#     )
#     fig.update_xaxes(title_text="x", row=1, col=2)
#     fig.update_yaxes(title_text="F(x)", row=1, col=2)
#
#     # Update yaxis properties
#
#     fig.update_layout(height=400, width=900, showlegend=False)
#     fig.show()
#
# plot_cdf_pdf_plotly()
#
#
# X = stats.norm.rvs(size=10000)
# X_pit = stats.norm.cdf(X)
#
# fig = make_subplots(rows=1, cols=2, subplot_titles=("Samples", "Transformed Samples"))
#
# fig.add_trace(
#     go.Histogram(x=X),
#     row=1, col=1
# )
#
# fig.add_trace(
#     go.Histogram(x=X_pit),
#     row=1, col=2
# )
#
# fig.update_layout(height=400, width=900, showlegend=False)
# fig.show()


from copulas.datasets import sample_bivariate_age_income
from copulas.visualization import scatter_2d
from copulas.visualization import dist_1d
from copulas.multivariate import GaussianMultivariate

df = sample_bivariate_age_income()
df.head()

scatter_2d(df).show()
dist_1d(df['age'], title='Age').show()
dist_1d(df['income'], title='Income').show()
copula = GaussianMultivariate()
copula.fit(df)
age_cdf = copula.univariates[0].cdf(df['age'])
dist_1d(age_cdf, title='Age').show()
income_cdf = copula.univariates[1].cdf(df['income'])
dist_1d(income_cdf, title='income').show()