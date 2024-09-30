from typing import Dict, List
from warnings import simplefilter

import numpy as np
import pandas as pd
import yfinance as yf

from tools import roc, sma

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.copy_on_write = True

# def
MAX_STOCKS = 10
QUANTILE_LOW_MOMENTUM = 0.15
QUANTILE_HIGH_BETA = 0.9


def get_monthly_index():
    sp_500 = yf.download("^GSPC")
    sp_500["sma"] = sma(sp_500.Close, 150)
    sp_500["Date"] = sp_500.index
    sp_500["month"] = sp_500["Date"].dt.strftime("%y-%m")

    sp_500 = (
        sp_500.groupby("month")
        .agg(
            Date=("Date", "last"),
            Close=("Close", "last"),
            sma=("sma", "last"),
        )
        .reset_index()
        .set_index("Date")
        .sort_index()
    )
    return sp_500


def get_stocks(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """_summary_

    Args:
        symbols (List[str]): _description_

    Returns:
        Dict[str, pd.DataFrame]: _description_
    """

    dfs = {}
    stock_data = yf.download(
        symbols,
        rounding=2,
        progress=False,
        group_by="ticker",
    )

    # perform some pre preparation
    for symbol in stock_data.columns.get_level_values(0).unique():
        # drop unclear items
        df = stock_data[symbol]
        df = df[~(df.High == df.Low)]
        df = df.dropna()
        # df.index = pd.to_datetime(df.index).tz_convert(None)

        if len(df):
            dfs[symbol.lower()] = df

    return dfs


def resample_stocks_to_month(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = df.index
    df["month"] = df["Date"].dt.strftime("%y-%m")

    df = df.groupby("month").agg(
        Date=("Date", "last"),
        Open=("Open", "first"),
        Close=("Close", "last"),
        beta=("beta", "last"),
        changes=("changes", "sum"),
    )
    return df.reset_index().set_index("Date").sort_index()


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data["pct"] = data.Close.pct_change()
    data["changes"] = np.sign(data["pct"])

    return data


def max_beta(df: pd.DataFrame) -> pd.DataFrame:
    df["month"] = df.index.strftime("%y-%m")

    # Sort DataFrame by month and percentage change in descending order
    df_sorted = df.sort_values(by=["month", "pct"], ascending=[True, False])

    # Calculate the average of the top five percentage changes for each month
    df_sorted["rank"] = df_sorted.groupby("month")["pct"].rank(
        method="first", ascending=False
    )
    monthly_top_five_avg = (
        df_sorted[df_sorted["rank"] <= 5].groupby("month")["pct"].mean().rename("beta")
    )

    # Merge the calculated beta values back into the original DataFrame
    df = df.merge(monthly_top_five_avg, on="month", how="left").set_index(df.index)

    return df


def momentum(df: pd.DataFrame) -> pd.DataFrame:
    for interval in [3, 6, 9, 12]:
        df["roc"] = roc(df.Close, interval)
        df[f"changes_{interval}"] = df.changes.rolling(interval).sum().shift(-1)
        df[f"changes_{interval}"] = np.where(
            (df.roc > 0) & (df[f"changes_{interval}"] > 0),
            df[f"changes_{interval}"] + (df.roc / 1000),
            df[f"changes_{interval}"],
        )
    return df


def sp_500_list():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return list(table.Symbol.str.replace(".", "-"))


def prepare_stocks(index: pd.DataFrame) -> pd.DataFrame:
    stocks = get_stocks(sp_500_list())

    for symbol, df in stocks.items():
        df = add_indicators(df)
        df = max_beta(df)

        df = resample_stocks_to_month(df)
        df = momentum(df)

        stocks[symbol] = df

    change_periods = [6, 9, 12]
    changes = {
        period: index.copy().reset_index().set_index("month")
        for period in change_periods
    }
    beta = index.copy().reset_index().set_index("month")

    for symbol, df in stocks.items():
        beta = beta.merge(
            df[["month", "beta"]].set_index("month").rename(columns={"beta": symbol}),
            left_index=True,
            right_index=True,
            how="left",
        )

        for period in change_periods:
            changes[period] = changes[period].merge(
                df[["month", f"changes_{period}"]]
                .set_index("month")
                .rename(columns={f"changes_{period}": symbol}),
                left_index=True,
                right_index=True,
                how="left",
            )

    neg_symbols = {}
    for period in [6, 9]:
        period_changes = changes[period].drop(
            columns=["sma", "Close", "Date"], errors="ignore"
        )
        neg_symbols.update(
            {
                month: neg_symbols.get(month, [])
                + period_changes.columns[
                    period_changes.loc[month]
                    < period_changes.loc[month].quantile(QUANTILE_LOW_MOMENTUM)
                ].tolist()
                for month in period_changes.index
            }
        )

    beta_changes = beta.drop(columns=["sma", "Close", "Date"], errors="ignore")
    for month in beta_changes.index:
        neg_symbols[month] = (
            neg_symbols.get(month, [])
            + beta_changes.columns[
                beta_changes.loc[month]
                > beta_changes.loc[month].quantile(QUANTILE_HIGH_BETA)
            ].tolist()
        )

    for month, symbols in neg_symbols.items():
        # if len(symbols):
        #    changes[12].loc[month, symbols] = np.nan
        for symbol in symbols:
            try:
                # changes[12].loc[month][symbol] = np.nan
                changes[12].loc[month, symbol] = np.nan
            except Exception as e:
                print(e)
                print(month, symbol)

    return changes[12].reset_index().set_index("Date")


def get_top_stocks(df: pd.DataFrame) -> list:
    df = df.drop(["month", "Close", "sma"]).reset_index()
    df.columns = ["symbol", "beta"]
    df = df[df.beta > 0]

    return df.sort_values("beta").tail(MAX_STOCKS).symbol.values


if __name__ == "__main__":
    sp_500 = get_monthly_index()
    sp_500_stocks = prepare_stocks(index=sp_500)

    if sp_500.iloc[-1].Close > sp_500.iloc[-1].sma:
        current_month = get_top_stocks(sp_500_stocks.iloc[-2].dropna().to_frame())

        next_month = get_top_stocks(sp_500_stocks.iloc[-1].dropna().to_frame())

        unchanged_stocks = set(current_month).intersection(next_month)
        removed_stocks = set(current_month) - set(next_month)
        added_stocks = set(next_month) - set(current_month)

        changes_txt = "# Planned transactions for next month\n"
        changes_txt = (
            changes_txt
            + "\n## New\n"
            + "\n".join([f"+ {stocks}" for stocks in added_stocks])
        )
        changes_txt = (
            changes_txt
            + "\n## Unchanged\n"
            + "\n".join([f"* {stocks}" for stocks in unchanged_stocks])
        )
        changes_txt = (
            changes_txt
            + "\n## Leave\n"
            + "\n".join([f"- {stocks}" for stocks in removed_stocks])
        )

        with open("CHANGES.md", "w") as text_file:
            text_file.write(changes_txt)
