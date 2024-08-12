import General
import General_with_InverseGamma
import General_with_hyper_priors
import GroupLassoRegression
import LassoRegression
import RidgeRegressionCNF
import RidgeRegression_Inverse_Gamma_CNF


def Experiment_1():
    print("NF Ridge, without lambda, Analytical Posterior Vs flow Posterior (General)")
    General.main()


def Experiment_2():
    print("NF Ridge, without lambda, Analytical vs Flow Posterior Predictive - Normal dist (hyper prior)")
    General_with_hyper_priors.main()


def Experiment_3():
    print("NF Ridge, without lambda, Integrated out variance, Analytical vs Flow  posterior -T-dist (Inverse Gamma)")
    General_with_InverseGamma.main()


def Experiment_4():
    print("NF Lasso, with fixed lambda, Normal vs T-distribution likelihood (LassoRegression)")
    LassoRegression.main()


def Experiment_5():
    print("NF GLasso, with fixed lambda, Normal vs T-distribution likelihood (GroupLassoRegression)")
    GroupLassoRegression.main()


def Experiment_6():
    print("CNF Ridge, simulated annealing,  posterior, marginal likelihood at T=1, MAP, standardized coefficient ("
          "RidgeCNF)")
    RidgeRegressionCNF.main()


def Experiment_7():
    print("CNF, Ridge, inverse gamma parameters vs Posterior Ridge CNF Inverse Gamma")
    RidgeRegression_Inverse_Gamma_CNF.main()


