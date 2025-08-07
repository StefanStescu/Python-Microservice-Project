#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sleep Quality and Lifestyle Factors Analysis

Author: Stefan Stescu
Date: January 2, 2025

This script analyzes factors that influence sleep quality, including stress levels,
physical activity, and pre-existing sleep disorders using statistical methods.
"""

# ------------------------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------------------------
# Data manipulation
import pandas as pd
import numpy as np

# Statistical analysis
import scipy.stats as stats
from scipy.stats import (
    mode,
    skew,
    kurtosis,
    normaltest,
    kstest,
    pearsonr,
    chi2_contingency,
    chisquare,
    spearmanr,
    levene,
)
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Other libraries
from datetime import datetime
from statsmodels.stats.weightstats import DescrStatsW
from typing import NamedTuple


# ------------------------------------------------------------------------------
# Data Import and Preprocessing
# ------------------------------------------------------------------------------
def load_and_preprocess_data():
    """Load and preprocess the sleep dataset."""
    # Import the data from the CSV file
    sleep_data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

    # Replace spaces in column names with dots
    sleep_data.columns = sleep_data.columns.str.replace(" ", ".", regex=False)

    # Create a filtered dataset with specific conditions
    sleep_quality_check = sleep_data[
        ((sleep_data["Sleep.Duration"] >= 5.0) & (sleep_data["Sleep.Duration"] <= 7.5))
        & ((sleep_data["Stress.Level"] >= 2) & (sleep_data["Stress.Level"] <= 9))
        & (
            (sleep_data["Quality.of.Sleep"] >= 3)
            & (sleep_data["Quality.of.Sleep"] <= 9)
        )
    ]

    # Drop unnecessary columns
    drop_columns = ["Daily.Steps", "Blood.Pressure"]
    sleep_quality_check = sleep_quality_check.drop(columns=drop_columns)

    # Fill NA values for Sleep.Disorder with 'None'
    sleep_quality_check = sleep_quality_check.fillna({"Sleep.Disorder": "None"})

    # Save the filtered dataset to a new CSV file
    sleep_quality_check.to_csv("sleep_quality_check.csv", index=False)

    # Transform Sleep.Disorder to categorical
    sleep_quality_check["Sleep.Disorder"] = pd.Categorical(
        sleep_quality_check["Sleep.Disorder"],
        categories=["Insomnia", "None", "Sleep Apnea"],
    )

    # Create stress level categories
    sleep_quality_check["sleep_disorder_stress"] = pd.cut(
        sleep_quality_check["Stress.Level"],
        bins=[0, 3, 5, 9],
        labels=["Low", "Moderate", "High"],
        right=True,
    )

    return sleep_quality_check


# ------------------------------------------------------------------------------
# Exploratory Data Analysis
# ------------------------------------------------------------------------------
def perform_eda(sleep_quality_check):
    """Perform exploratory data analysis on the dataset."""

    # Display dataset information
    print("Dataset shape:", sleep_quality_check.shape)
    print("Dataset structure:")
    print(sleep_quality_check.info())
    print("Column names:", sleep_quality_check.columns)

    # Check unique values
    print("Unique Sleep Disorders:", sleep_quality_check["Sleep.Disorder"].unique())
    print(
        "Unique Stress Levels:", sleep_quality_check["sleep_disorder_stress"].unique()
    )

    # Create numerical subset for analysis
    data_sep_num = sleep_quality_check.loc[
        :,
        [
            "Age",
            "Sleep.Duration",
            "Stress.Level",
            "Quality.of.Sleep",
            "Physical.Activity.Level",
            "Heart.Rate",
        ],
    ]

    # Display statistical summary
    pd.set_option("display.max_columns", None)
    summary = data_sep_num.describe()
    print("Statistical Summary:")
    print(summary)

    # Calculate skewness and kurtosis for each numerical variable
    for column in data_sep_num.columns:
        print(f"The skewness of {column} is: {skew(sleep_quality_check[column])}")
        print(f"The kurtosis of {column} is: {kurtosis(sleep_quality_check[column])}")

    # Analyze categorical variables
    print("sleep_disorder_stress statistics:")
    print(sleep_quality_check["sleep_disorder_stress"].describe())
    print("Sleep.Disorder statistics:")
    print(sleep_quality_check["Sleep.Disorder"].describe())

    return data_sep_num


# ------------------------------------------------------------------------------
# Data Visualization
# ------------------------------------------------------------------------------
def create_visualizations(sleep_quality_check):
    """Create various visualizations for data analysis."""

    # Create histograms for numerical variables
    numerical_vars = [
        "Age",
        "Sleep.Duration",
        "Stress.Level",
        "Quality.of.Sleep",
        "Physical.Activity.Level",
        "Heart.Rate",
    ]

    for var in numerical_vars:
        plt.figure(figsize=(10, 8))
        plt.hist(x=sleep_quality_check[var], edgecolor="white", color="green")
        plt.title(f"Histogram of {var}")
        plt.xlabel(var)
        plt.ylabel("Frequency")
        plt.tight_layout()
        # plt.savefig(f'histogram_{var}.png')  # Uncomment to save figures

    # Create histograms for categorical variables
    plt.figure(figsize=(10, 8))
    plt.hist(x=sleep_quality_check["Sleep.Disorder"], edgecolor="white", color="red")
    plt.title("Sleep Disorders")
    plt.xlabel("Sleep Disorder Type")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.figure(figsize=(10, 8))
    plt.hist(
        x=sleep_quality_check["sleep_disorder_stress"], edgecolor="white", color="blue"
    )
    plt.title("Stress Levels")
    plt.xlabel("Stress Level Category")
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Create boxplots to identify outliers
    for var in numerical_vars:
        plt.figure(figsize=(6, 7))
        plt.boxplot(sleep_quality_check[var])
        plt.title(f"Boxplot of {var}")
        plt.tight_layout()
        # plt.savefig(f'boxplot_{var}.png')  # Uncomment to save figures


# ------------------------------------------------------------------------------
# Cross-tabulation and Chi-Square Analysis
# ------------------------------------------------------------------------------
def perform_crosstab_analysis(sleep_quality_check):
    """Perform cross-tabulation and chi-square analysis on categorical variables."""

    # Create cross-tabulation
    tabelarea_datelor = pd.crosstab(
        sleep_quality_check["Sleep.Disorder"],
        sleep_quality_check["sleep_disorder_stress"],
    )
    print("Cross-tabulation:")
    print(tabelarea_datelor)

    # Calculate relative frequencies
    print("\nPartial relative frequencies:")
    freq_rel_partial = pd.crosstab(
        sleep_quality_check["Sleep.Disorder"],
        sleep_quality_check["sleep_disorder_stress"],
        normalize=True,
        margins=False,
    )
    print(freq_rel_partial)

    # Calculate row-conditional frequencies
    print("\nSleep.Disorder conditional frequencies (by row):")
    freq_cond_row = pd.crosstab(
        sleep_quality_check["Sleep.Disorder"],
        sleep_quality_check["sleep_disorder_stress"],
        normalize="index",
        margins=False,
    )
    print(freq_cond_row)

    # Calculate column-conditional frequencies
    print("\nSleep.Disorder conditional frequencies (by column):")
    freq_cond_col = pd.crosstab(
        sleep_quality_check["Sleep.Disorder"],
        sleep_quality_check["sleep_disorder_stress"],
        normalize="columns",
        margins=False,
    )
    print(freq_cond_col)

    # Calculate marginal frequencies
    print("\nMarginal frequencies:")
    freq_marginal = pd.crosstab(
        sleep_quality_check["Sleep.Disorder"],
        sleep_quality_check["sleep_disorder_stress"],
        normalize=True,
        margins=True,
    )
    print(freq_marginal)

    # Perform chi-square test on the cross-tabulation
    chisq, df, p_value, _ = chi2_contingency(tabelarea_datelor)
    print("\nChi-square test results:")
    print(f"Chisq: {chisq}")
    print(f"df: {df}")
    print(f"P-value: {p_value}")

    return tabelarea_datelor


# ------------------------------------------------------------------------------
# Frequency Analysis for Categorical Variables
# ------------------------------------------------------------------------------
def analyze_categorical_frequencies(sleep_quality_check):
    """Analyze frequency distributions of categorical variables."""

    # Frequency analysis for sleep_disorder_stress
    print("\nFrequency analysis for sleep_disorder_stress:")
    tabel_frecvente_SDS = sleep_quality_check["sleep_disorder_stress"].value_counts()
    X_sq, p_value = chisquare(tabel_frecvente_SDS)
    print(f"X-sq: {X_sq}")
    print(f"df: {len(tabel_frecvente_SDS) - 1}")
    print(f"P-value: {p_value}")

    # Compare observed frequencies with theoretical distribution
    print("\nComparing sleep_disorder_stress with theoretical distribution:")
    distr_teoretica = [0.1, 0.3, 0.6]
    tabel_frecvente_SDS2 = sleep_quality_check["sleep_disorder_stress"].value_counts()
    distr_frecv_asteptate = [
        prob * len(sleep_quality_check) for prob in distr_teoretica
    ]
    X_sq, p_value = chisquare(tabel_frecvente_SDS2, distr_frecv_asteptate)
    print(f"X-sq: {X_sq}")
    print(f"df: {len(tabel_frecvente_SDS2) - 1}")
    print(f"P-value: {p_value}")

    # Frequency analysis for Sleep.Disorder
    print("\nFrequency analysis for Sleep.Disorder:")
    tabel_frecvente_SleepD = sleep_quality_check["Sleep.Disorder"].value_counts()
    X_sq, p_value = chisquare(tabel_frecvente_SleepD)
    print(f"X-sq: {X_sq}")
    print(f"df: {len(tabel_frecvente_SleepD) - 1}")
    print(f"P-value: {p_value}")

    # Compare observed frequencies with theoretical distribution
    print("\nComparing Sleep.Disorder with theoretical distribution:")
    distr_teoretica_SD = [0.1, 0.2, 0.7]
    tabel_frecvente_SleepD2 = sleep_quality_check["Sleep.Disorder"].value_counts()
    distr_frecv_asteptate2 = [
        prob * len(sleep_quality_check) for prob in distr_teoretica_SD
    ]
    X_sq, p_value = chisquare(tabel_frecvente_SleepD2, distr_frecv_asteptate2)
    print(f"X-sq: {X_sq}")
    print(f"df: {len(tabel_frecvente_SleepD2) - 1}")
    print(f"P-value: {p_value}")


# ------------------------------------------------------------------------------
# Correlation Analysis
# ------------------------------------------------------------------------------
def perform_correlation_analysis(sleep_quality_check, data_sep_num):
    """Perform correlation analysis on numerical variables."""

    # Covariance analysis
    analiza_cov = np.cov(data_sep_num, rowvar=False)
    df_analiza_cov = pd.DataFrame(
        analiza_cov, columns=data_sep_num.columns, index=data_sep_num.columns
    )
    print("\nCovariance matrix:")
    print(df_analiza_cov)

    # Pearson correlation
    matrice_corelatieP = np.corrcoef(data_sep_num, rowvar=False)
    df_matrice_corelatieP = pd.DataFrame(
        matrice_corelatieP, columns=data_sep_num.columns, index=data_sep_num.columns
    )
    print("\nPearson correlation matrix:")
    print(df_matrice_corelatieP)

    # Spearman correlation
    result = spearmanr(data_sep_num, axis=0)
    matrice_corelatieS = result.correlation
    df_matrice_corelatieS = pd.DataFrame(
        matrice_corelatieS, columns=data_sep_num.columns, index=data_sep_num.columns
    )
    print("\nSpearman correlation matrix:")
    print(df_matrice_corelatieS)

    # Perform correlation tests between Stress.Level and other variables
    variables = [
        "Age",
        "Sleep.Duration",
        "Quality.of.Sleep",
        "Physical.Activity.Level",
        "Heart.Rate",
    ]

    print("\nCorrelation tests with Stress.Level:")
    for var in variables:
        coef, p_val = pearsonr(
            sleep_quality_check["Stress.Level"], sleep_quality_check[var]
        )
        print(f"Correlation between Stress.Level and {var}:")
        print(f"  Correlation coefficient r: {coef}")
        print(f"  P-value: {p_val}")


# ------------------------------------------------------------------------------
# Regression Analysis
# ------------------------------------------------------------------------------
def perform_regression_analysis(sleep_quality_check):
    """Perform regression analysis to model stress levels."""

    # Simple linear regression - Quality of Sleep predicting Stress Level
    print("\nSimple Linear Regression Model:")
    x = sleep_quality_check["Quality.of.Sleep"]
    y = sleep_quality_check["Stress.Level"]
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const)
    regresie_lin_simpla = model.fit()
    print(regresie_lin_simpla.summary())

    # Multiple linear regression - Quality of Sleep and Age predicting Stress Level
    print("\nMultiple Linear Regression Model:")
    x_multiple = sleep_quality_check[["Quality.of.Sleep", "Age"]]
    x_multiple_with_const = sm.add_constant(x_multiple)
    model = sm.OLS(y, x_multiple_with_const)
    regresie_lin_multipla = model.fit()
    print(regresie_lin_multipla.summary())

    # Non-linear regression - Quadratic model for Quality of Sleep
    print("\nNon-linear Regression Model:")
    Y = sleep_quality_check["Stress.Level"]
    sleep_quality_check["Quality.of.Sleep^2"] = (
        sleep_quality_check["Quality.of.Sleep"] ** 2
    )
    X_rn = sm.add_constant(
        sleep_quality_check[["Quality.of.Sleep", "Quality.of.Sleep^2"]]
    )
    model = sm.OLS(Y, X_rn)
    regresie_neliniara = model.fit()
    print(regresie_neliniara.summary())

    # Compare regression models
    print("\nComparing Regression Models (ANOVA):")
    print(sm.stats.anova_lm(regresie_lin_simpla, regresie_lin_multipla))

    return regresie_lin_simpla, regresie_lin_multipla, regresie_neliniara


# ------------------------------------------------------------------------------
# Hypothesis Testing
# ------------------------------------------------------------------------------
def perform_hypothesis_testing(sleep_quality_check):
    """Perform various hypothesis tests on the dataset."""

    # Confidence interval for Stress.Level mean
    print("\nConfidence interval for Stress.Level mean:")
    conf_int = DescrStatsW(sleep_quality_check["Stress.Level"]).tconfint_mean()
    print(f"Confidence interval: {conf_int}")

    # Custom function for confidence interval calculation
    def interval_incredere_medie(data, incredere, val_testata=0):
        a = np.array(data)
        n = len(a)
        media = np.mean(a)
        sem = stats.sem(a)
        h = sem * stats.t.ppf((1 + incredere) / 2.0, n - 1)
        p_val = stats.ttest_1samp(a, val_testata)
        return media - h, media + h, p_val

    interval_incredere = interval_incredere_medie(
        sleep_quality_check["Stress.Level"], 0.95
    )
    print(f"Confidence interval (custom function): {interval_incredere[:2]}")

    # T-test for Stress.Level against a fixed value (5)
    result = stats.ttest_1samp(sleep_quality_check["Stress.Level"], 5)
    t_calc = result.statistic
    p_value = result.pvalue
    p_value_uni = p_value / 2 if t_calc > 0 else 1 - (p_value / 2)

    mean = np.mean(sleep_quality_check["Stress.Level"])
    print("\nOne-sample t-test (Stress.Level against 5):")
    print(f"T statistic: {t_calc}")
    print(f"P-value: {p_value}")
    print(f"Mean: {mean}")

    # Test difference between two groups (Insomnia vs Sleep Apnea)
    data_filtered_SQC = sleep_quality_check[
        sleep_quality_check["Sleep.Disorder"].isin(["Insomnia", "Sleep Apnea"])
    ]

    insomnia_PC = data_filtered_SQC[data_filtered_SQC["Sleep.Disorder"] == "Insomnia"][
        "Stress.Level"
    ]

    sleep_apnea_PC = data_filtered_SQC[
        data_filtered_SQC["Sleep.Disorder"] == "Sleep Apnea"
    ]["Stress.Level"]

    # Test for equal variances
    t_calc, p_value = stats.bartlett(insomnia_PC, sleep_apnea_PC)
    print("\nBartlett's test for equal variances:")
    print(f"T statistic: {t_calc}")
    print(f"P-value: {p_value}")

    # Welch's t-test for two independent samples
    t_calc2, p_value2 = stats.ttest_ind(insomnia_PC, sleep_apnea_PC)
    mean_IPC = np.mean(insomnia_PC)
    mean_SAPC = np.mean(sleep_apnea_PC)
    print("\nWelch's t-test for Insomnia vs Sleep Apnea:")
    print(f"T statistic: {t_calc2}")
    print(f"P-value: {p_value2}")
    print(f"Mean for Insomnia: {mean_IPC}")
    print(f"Mean for Sleep Apnea: {mean_SAPC}")

    # ANOVA test for stress levels across stress categories
    sleep_quality_check.rename(columns={"Stress.Level": "Stress_Level"}, inplace=True)
    model_means = ols(
        "Stress_Level ~ sleep_disorder_stress", data=sleep_quality_check
    ).fit()
    anova_rezultat_test = sm.stats.anova_lm(model_means, typ=2)
    print("\nANOVA test results:")
    print(anova_rezultat_test)
    print("Model parameters:")
    print(model_means.params)

    # Change column name back
    sleep_quality_check.rename(columns={"Stress_Level": "Stress.Level"}, inplace=True)


# ------------------------------------------------------------------------------
# Diagnostic Tests for Regression Models
# ------------------------------------------------------------------------------
def perform_diagnostic_tests(
    regresie_lin_simpla, regresie_lin_multipla, regresie_neliniara
):
    """Perform diagnostic tests on regression models."""

    from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey
    from statsmodels.stats.stattools import durbin_watson
    import statsmodels.stats.api as sma

    # Simple Linear Regression Tests
    print("\nDiagnostic Tests for Simple Linear Regression:")

    # White test for heteroskedasticity
    white_test_rls = het_white(
        regresie_lin_simpla.resid, regresie_lin_simpla.model.exog
    )
    labels = [
        "Test Statistic",
        "Test Statistic P-value",
        "F-Statistic",
        "F-Test P-value",
    ]
    print("White test for heteroskedasticity:")
    print(dict(zip(labels, white_test_rls)))

    # Breusch-Pagan test for heteroskedasticity
    test_BP = sma.het_breuschpagan(
        regresie_lin_simpla.resid, regresie_lin_simpla.model.exog
    )
    print("Breusch-Pagan test for heteroskedasticity:")
    print(dict(zip(labels, test_BP)))

    # Breusch-Godfrey test for autocorrelation
    bg_test = acorr_breusch_godfrey(regresie_lin_simpla)
    print("Breusch-Godfrey test for autocorrelation:")
    print(dict(zip(labels, bg_test)))

    # Multiple Linear Regression Tests
    print("\nDiagnostic Tests for Multiple Linear Regression:")

    # Durbin-Watson test for autocorrelation
    dw_stat_mlr = durbin_watson(regresie_lin_multipla.resid)
    print(f"Durbin-Watson Statistic: {dw_stat_mlr}")

    # Jarque-Bera test for normality
    jb_stat_mlr, jb_pval_mlr = stats.jarque_bera(regresie_lin_multipla.resid)
    print(f"Jarque-Bera test: statistic={jb_stat_mlr}, p-value={jb_pval_mlr}")

    # Breusch-Pagan test for heteroskedasticity
    test_BP_RLM = sma.het_breuschpagan(
        regresie_lin_multipla.resid, regresie_lin_multipla.model.exog
    )
    print("Breusch-Pagan test for heteroskedasticity:")
    print(dict(zip(labels, test_BP_RLM)))

    # White test for heteroskedasticity
    white_test_rlm = het_white(
        regresie_lin_multipla.resid, regresie_lin_multipla.model.exog
    )
    print("White test for heteroskedasticity:")
    print(dict(zip(labels, white_test_rlm)))

    # Non-linear Regression Tests
    print("\nDiagnostic Tests for Non-linear Regression:")

    # Durbin-Watson test for autocorrelation
    dw_stat_nlr = durbin_watson(regresie_neliniara.resid)
    print(f"Durbin-Watson Statistic: {dw_stat_nlr}")

    # Jarque-Bera test for normality
    jb_stat_nlr, jb_pval_nlr = stats.jarque_bera(regresie_neliniara.resid)
    print(f"Jarque-Bera test: statistic={jb_stat_nlr}, p-value={jb_pval_nlr}")

    # White test for heteroskedasticity
    white_test_rn = het_white(regresie_neliniara.resid, regresie_neliniara.model.exog)
    print("White test for heteroskedasticity:")
    print(dict(zip(labels, white_test_rn)))


# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------
def main():
    """Main function to execute the analysis."""

    # Load and preprocess data
    sleep_quality_check = load_and_preprocess_data()

    # Perform exploratory data analysis
    data_sep_num = perform_eda(sleep_quality_check)

    # Create visualizations
    create_visualizations(sleep_quality_check)

    # Perform cross-tabulation analysis
    tabelarea_datelor = perform_crosstab_analysis(sleep_quality_check)

    # Analyze categorical frequencies
    analyze_categorical_frequencies(sleep_quality_check)

    # Perform correlation analysis
    perform_correlation_analysis(sleep_quality_check, data_sep_num)

    # Perform regression analysis
    regresie_lin_simpla, regresie_lin_multipla, regresie_neliniara = (
        perform_regression_analysis(sleep_quality_check)
    )

    # Perform hypothesis testing
    perform_hypothesis_testing(sleep_quality_check)

    # Perform diagnostic tests for regression models
    perform_diagnostic_tests(
        regresie_lin_simpla, regresie_lin_multipla, regresie_neliniara
    )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
