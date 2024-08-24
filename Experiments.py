import RidgeRegression
import RidgeRegression_with_InverseGamma
import RidgeRegression_with_hyper_priors
import GroupLassoRegression
import GroupLassoRegressionCNF
import LassoRegression
import LassoRegressionCNF
import RidgeRegressionCNF
import RidgeRegressionCNF_Inverse_Gamma


def Experiment_1():
    print("NF Ridge, without lambda, Analytical Posterior Vs flow Posterior (General)")
    RidgeRegression.main()


def Experiment_2():
    print("NF Ridge, without lambda, Analytical vs Flow Posterior Predictive - Normal dist (hyper prior)")
    RidgeRegression_with_hyper_priors.main()


def Experiment_3():
    print("NF Ridge, without lambda, Integrated out variance, Analytical vs Flow  posterior -T-dist (Inverse Gamma)")
    RidgeRegression_with_InverseGamma.main()


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
    RidgeRegressionCNF_Inverse_Gamma.main()


def Experiment_8():
    print("CNF, Lasso, simulated annealing, posterior, marginal likelihood at T=1, MAP, standardized coefficient ("
          "LassoCNF)")
    LassoRegressionCNF.main()


def Experiment_9():
    print("CNF, GroupLasso, simulated annealing, posterior, marginal likelihood at T=1, MAP, standardized coefficient ("
          "GroupLassoCNF)")
    GroupLassoRegressionCNF.main()


def Experiment_10():
    print("Run the Group Lasso CNF for Betas. For a fixed lambda, get the MAP beta. Similarly, run the Group Lasso CNF"
          " without Betas with taus. For the same lambda, get the MAP tau. With this tau, sample betas. "
          "See if the betas from initial part matches to the one from second part.")
