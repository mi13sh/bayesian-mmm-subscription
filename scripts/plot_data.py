"""Quick visual check of the simulated data. Saves artifacts/data_overview.png."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"

df = pd.read_csv("data/weekly.csv", parse_dates=["week"], index_col="week")

panels = [
    ("pclv_new_subs_eur", "pCLV of new subscriptions (EUR / week)"),
    ("new_subs", "New paid subscriptions / week"),
    ("spend_search", "Paid search spend (EUR)"),
    ("spend_social", "Paid social spend (EUR)"),
    ("spend_display", "Display spend (EUR)"),
    ("spend_affiliate", "Affiliate spend (EUR)"),
    ("spend_tv", "TV spend (EUR)"),
    ("email_accounts_reached", "Email accounts reached"),
    ("price_monthly_eur", "Monthly list price (EUR)"),
]
launches = {c.replace("flag_", "").replace("_", " "): df.index[df[c].astype(bool)][0]
            for c in df if c.startswith("flag_")}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(5, 2, figsize=(11, 12), sharex=True)
for ax, (col, title) in zip(axes.flat, panels):
    step = col == "price_monthly_eur"
    ax.plot(df.index, df[col], color=BLUE, lw=1.6, drawstyle="steps-post" if step else "default")
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    if col == "pclv_new_subs_eur":
        for name, date in launches.items():
            ax.axvline(date, color=MUTED, lw=0.8, ls="--")
            ax.text(date, ax.get_ylim()[1], f" {name}", color=MUTED, fontsize=8, va="top")
        for date in df.index[df["promo_week"] == 1]:
            ax.axvspan(date, date + pd.Timedelta(days=7), color=GRID, lw=0)
axes.flat[-1].axis("off")
for ax in axes.flat[:-1]:
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", labelbottom=True)
fig.tight_layout()

Path("artifacts").mkdir(exist_ok=True)
fig.savefig("artifacts/data_overview.png", dpi=150)
print("Saved artifacts/data_overview.png")
