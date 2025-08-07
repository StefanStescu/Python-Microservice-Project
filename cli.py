import click
from data_loader import DataLoader
from models import SleepRecord
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
import statsmodels.stats.api as sms
from scipy.stats import jarque_bera
import json
import csv
import asyncio
import os

# CLI for Sleep Data Management
# This module provides a command-line interface to analyze, summarize, plot, and run diagnostics on sleep data
# It uses the DataLoader class to load and preprocess the dataset
# It also uses the SleepRecord Pydantic model to validate records
@click.group()
def cli():
    """CLI for Sleep Data Management"""
    pass

# ------------------------
# ANALYZE
# ------------------------
@cli.command()
@click.option("--file", "-f", type=click.Path(exists=True), required=True, help="Path to CSV file")
def analyze(file):
    click.echo(f"📂 Loading file: {file}")
    loader = DataLoader(file)
    df_clean = loader.preprocess()
    click.echo(f"✅ Preprocessed data shape: {df_clean.shape}")

    valid_records = loader.validate_records()
    click.echo(f"✅ Valid records: {len(valid_records)}")

    click.echo("\n📊 Valid Records Preview:")
    for rec in valid_records[:5]:
        click.echo(rec.model_dump_json(indent=5))

# ------------------------
# SUMMARY
# ------------------------
@cli.command()
@click.option("--file", "-f", type=click.Path(exists=True), required=True)
def summary(file):
    click.echo(f"📂 Loading file: {file}")
    loader = DataLoader(file)
    df_clean = loader.preprocess()

    click.echo("\n📈 Statistical summary:")
    click.echo(df_clean.describe(include="all").to_string())

    click.echo("\n📊 Skewness and Kurtosis:")
    numeric_columns = [
        "Age", "Sleep_Duration", "Stress_Level",
        "Quality_of_Sleep", "Physical_Activity_Level", "Heart_Rate"
    ]

    for column in numeric_columns:
        s = skew(df_clean[column])
        k = kurtosis(df_clean[column])
        click.echo(f"{column:<30} - Skewness: {s:>6.2f}, Kurtosis: {k:>6.2f}")

    click.echo("\n📊 Categorical Variables:")
    click.echo(df_clean["Sleep_Disorder"].value_counts().to_string())
    click.echo(df_clean["Stress_Category"].value_counts().to_string())

# ------------------------
# PLOT
# ------------------------
@cli.command()
@click.option("--file", "-f", type=click.Path(exists=True), required=True)
@click.option("--show", is_flag=True)
@click.option("--save", is_flag=True)
def plot(file, show, save):
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    click.echo(f"📂 Plotting data from: {file}")
    loader = DataLoader(file)
    df_clean = loader.preprocess()

    numeric_columns = [
        "Age", "Sleep_Duration", "Stress_Level",
        "Quality_of_Sleep", "Physical_Activity_Level", "Heart_Rate"
    ]

    # Histograms
    for column in numeric_columns:
        filename = os.path.join(output_dir, f"{column}_histogram.png")
        if save and os.path.exists(filename):
            click.echo(f"🟡 Histogram already exists: {filename} — Skipped.")
            continue

        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_clean, x=column, kde=True, color="skyblue", edgecolor="black")
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        if save:
            plt.savefig(filename)
            click.echo(f"✅ Saved histogram: {filename}")
        if show:
            plt.show()
        else:
            plt.close()

    # Boxplots
    for column in numeric_columns:
        filename = os.path.join(output_dir, f"{column}_boxplot.png")
        if save and os.path.exists(filename):
            click.echo(f"🟡 Boxplot already exists: {filename} — Skipped.")
            continue

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df_clean[column], color="lightgreen")
        plt.title(f"Boxplot of {column}")

        if save:
            plt.savefig(filename)
            click.echo(f"✅ Saved boxplot: {filename}")
        if show:
            plt.show()
        else:
            plt.close()

    # Countplots
    for cat_col in ["Sleep_Disorder", "Stress_Category"]:
        filename = os.path.join(output_dir, f"{cat_col}_countplot.png")
        if save and os.path.exists(filename):
            click.echo(f"🟡 Countplot already exists: {filename} — Skipped.")
            continue

        plt.figure(figsize=(10, 6))
        sns.countplot(x=df_clean[cat_col], palette="pastel")
        plt.title(f"Count of {cat_col}")

        if save:
            plt.savefig(filename)
            click.echo(f"✅ Saved countplot: {filename}")
        if show:
            plt.show()
        else:
            plt.close()

    click.echo("📊 Plot generation complete.")


# ------------------------
# REGRESSION LOGIC
# ------------------------
async def perform_regression_logic(file):
    loader = DataLoader(file)
    df_clean = loader.preprocess()
    await asyncio.sleep(0.1)

    x = df_clean["Quality_of_Sleep"]
    y = df_clean["Stress_Level"]
    x_with_const = sm.add_constant(x)
    model_simple = sm.OLS(y, x_with_const).fit()

    x_multi = df_clean[["Quality_of_Sleep", "Age"]]
    x_multi_with_const = sm.add_constant(x_multi)
    model_multiple = sm.OLS(y, x_multi_with_const).fit()

    df_clean["Quality_of_Sleep_squared"] = df_clean["Quality_of_Sleep"] ** 2
    x_quad = df_clean[["Quality_of_Sleep", "Quality_of_Sleep_squared"]]
    x_quad_with_const = sm.add_constant(x_quad)
    model_quad = sm.OLS(y, x_quad_with_const).fit()

    anova_results = sm.stats.anova_lm(model_simple, model_multiple)

    return {
        "simple_summary": model_simple.summary().as_text(),
        "multiple_summary": model_multiple.summary().as_text(),
        "quadratic_summary": model_quad.summary().as_text(),
        "anova": anova_results.to_string()
    }


# CLI entry point for regression
@cli.command()
@click.option("--file", "-f", type=click.Path(exists=True), required=True)
def regression(file):
    asyncio.run(perform_regression_logic(file))

# ------------------------
# DIAGNOSTICS
# ------------------------

def run_diagnostics_logic(file):
    loader = DataLoader(file)
    df = loader.preprocess()

    y = df["Stress_Level"]
    x = df["Quality_of_Sleep"]
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()

    white = het_white(model.resid, model.model.exog)
    bp = sms.het_breuschpagan(model.resid, model.model.exog)
    dw = durbin_watson(model.resid)
    jb_stat, jb_pval = jarque_bera(model.resid)

    return {
        "White Test": dict(zip(["Test Statistic", "P-Value", "F-Statistic", "F-Test P-Value"], white)),
        "Breusch-Pagan": dict(zip(["Test Statistic", "P-Value", "F-Statistic", "F-Test P-Value"], bp)),
        "Durbin-Watson": dw,
        "Jarque-Bera": {
            "Test Statistic": jb_stat,
            "P-Value": jb_pval
        }
    }

@cli.command()
@click.option("--file", "-f", type=click.Path(exists=True), required=True)
def diagnostics(file):
    result = run_diagnostics_logic(file)
    click.echo(json.dumps(result, indent=4))


# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    cli()
