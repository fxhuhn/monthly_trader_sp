import os
from datetime import timedelta
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
QUANTILE_HIGH_BETA = 0.85


class SP_500_stocks:
    df = None

    def __init__(self, filename="S&P_500_Historical_04-08-2024.csv") -> None:
        if os.path.isfile(filename):
            self.df = pd.read_csv(filename, index_col="date")
            self.df = self.df[self.df.index >= "2000-01-01"]
            self.df["tickers"] = self.df["tickers"].apply(
                lambda x: sorted(x.split(","))
            )

    def get_symbols(self, year: int, month: int, day: int = 1) -> list[str]:
        if self.df is None:
            return None

        snap_shot = f"{year}-{month:02}-{day:02}"
        df2 = self.df[self.df.index <= snap_shot]
        return df2.tail(1).tickers.values[0]

    def all_symbols(self) -> list[str]:
        return sorted(set(sum(self.df.tickers.values, [])))


sp_500_stocks = SP_500_stocks()


def get_monthly_index():
    sp_500 = yf.download("^GSPC", start="2010-01-01")
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
    dfs = {}
    stock_data = yf.download(
        symbols, rounding=2, progress=False, group_by="ticker", start="2010-01-01"
    )

    for symbol in stock_data.columns.get_level_values(0).unique():
        # drop unclear items
        df = stock_data[symbol]
        df = df[~(df.High == df.Low)]
        df = df.dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)

        if len(df):
            dfs[symbol.lower()] = df

    try:
        dfs.pop("googl")
    except KeyError:
        pass

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
    data["changes"] = np.sign(data["pct"].round(2))
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
        df["roc"] = roc(df.Close, interval).shift(-1)
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
    stocks = get_stocks(sp_500_stocks.all_symbols())

    for symbol, df in stocks.items():
        df = add_indicators(df)
        df = max_beta(df)

        df = resample_stocks_to_month(df)
        df = momentum(df)

        stocks[symbol] = df

    for index_date in index.index:
        monthly_stocks = sp_500_stocks.get_symbols(index_date.year, index_date.month)
        monthly_stocks = [monthly_stock.lower() for monthly_stock in monthly_stocks]

        for symbol in list(set(stocks.keys()) - set(monthly_stocks)):
            df = stocks[symbol]
            df.loc[df.month == f"{index_date:%y-%m}", "beta"] = np.nan
            for i in [3, 6, 9, 12]:
                df.loc[df.month == f"{index_date:%y-%m}", f"changes_{i}"] = np.nan
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

    """
    symbol_txt = ""
    for key, value in neg_symbols.items():
        symbol_txt = symbol_txt + f'{key},{",".join(value)}\n'
    with open("./neg_symbols.csv", "w") as symbol_file:
        symbol_file.write(symbol_txt)
    """
    return changes[12].reset_index().set_index("Date")


def get_top_stocks(df: pd.DataFrame) -> list:
    df = df.drop(["month", "Close", "sma"]).reset_index()
    df.columns = ["symbol", "beta"]
    df = df[df.beta > 0]
    # print(df.sort_values("beta").tail(MAX_STOCKS))
    return df.sort_values("beta").tail(MAX_STOCKS).symbol.values


if __name__ == "__main__":
    sp_500 = get_monthly_index()
    sp_500_stocks = prepare_stocks(index=sp_500)
    sp_500_stocks = sp_500_stocks["2016-12-01":]

    portfolio = []
    for month in range(len(sp_500_stocks)):
        if (sp_500_stocks.iloc[month].Close > sp_500_stocks.iloc[month].sma) and (
            sp_500_stocks.iloc[month].month != sp_500_stocks.month.max()
        ):
            top_stocks = get_top_stocks(sp_500_stocks.iloc[month].dropna().to_frame())

            for ticker in top_stocks:
                portfolio.append(
                    {
                        "month": f"{(sp_500_stocks.iloc[month].name+timedelta(days=10)).year}-{(sp_500_stocks.iloc[month].name+timedelta(days=10)).month:0>2}",
                        "symbol": ticker,
                    }
                )
    portfolio = pd.DataFrame(portfolio)

    stocks = get_stocks(list(portfolio.symbol.unique()))
    for pos, position in portfolio.iterrows():
        df = stocks[position.symbol]
        df["Date"] = df.index
        df["month"] = df["Date"].dt.strftime("%Y-%m")

        df_month = df.groupby("month").agg(
            End=("Date", "last"),
            Start=("Date", "first"),
            Open=("Open", "first"),
            Close=("Close", "last"),
        )
        try:
            portfolio.loc[pos, "start"] = df_month.loc[position.month].Start
            portfolio.loc[pos, "end"] = df_month.loc[position.month].End
            portfolio.loc[pos, "buy"] = df_month.loc[position.month].Open
            portfolio.loc[pos, "sell"] = df_month.loc[position.month].Close
            portfolio.loc[pos, "profit"] = (
                (
                    (
                        df_month.loc[position.month].Close
                        / df_month.loc[position.month].Open
                    )
                    - 1
                )
                * 100
            ).round(1)
        except:
            pass

    trade_journal = portfolio.set_index("month").astype(str).to_markdown(floatfmt=".2f")

    try:
        portfolio["invest"] = (10_000 / portfolio["buy"]).astype(int) * portfolio["buy"]
    except:
        print(portfolio["buy"])

    portfolio["profit"] = (10_000 / portfolio["buy"]).astype(int) * portfolio["sell"]

    monthly = (
        portfolio.groupby("month")
        .agg(
            Positions=("symbol", "count"),
            Invest=("invest", "sum"),
            Profit=("profit", "sum"),
        )
        .dropna()
    )
    monthly["earning"] = (
        (monthly.Profit - monthly.Invest) / monthly.Invest * 100
    ).round(1)

    readme_txt = f"# S&P 500 Trader\nStock Trading and Screening only end of month. With an average monthly return of {monthly.earning.mean():.2f}%. Every month!\n\n"
    readme_txt = (
        readme_txt
        + f'## Average Monthly Return\n{monthly.groupby(monthly.index.str[-2:]).agg(profit=("earning", "mean")).to_markdown(floatfmt=".2f")}\n\n'
    )

    readme_txt = readme_txt + f"## Tradehistory\n{trade_journal}\n\n"

    with open("README.md", "w") as text_file:
        text_file.write(readme_txt)
