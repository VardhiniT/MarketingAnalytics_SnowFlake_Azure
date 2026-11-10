#!/usr/bin/env python3
"""Generate a synthetic extension of the marketing dataset.

Learns from the six original CSV files (column names, categories, value ranges, date span) and writes:

  <out>/history/   campaigns.csv, campaign_spend.csv, orders.csv   (cover the original date span; load once)
  <out>/batches/YYYY_MM/   customers, engagement_data, customer_journey, customer_reviews,
                           orders, campaign_spend, campaigns       (one folder per month; load in sequence)
  <out>/_manifest.json, <out>/_injected_issues.csv

The original files are only read, never changed. Everything written here is synthetic.
Dirty data (duplicates, blanks, odd casing, bad formats) is injected only into the generated rows,
and every injected issue is counted in _injected_issues.csv so tests can be reconciled.

Usage:
  python generate_extension.py --src ~/Documents/marketing_data --out ~/Documents/marketing_data_extension
"""
import argparse
import json
import uuid
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

FIRST = ["Aarav", "Maya", "Liam", "Sofia", "Noah", "Isha", "Ethan", "Zara", "Lucas", "Anika", "Mason", "Priya",
         "Oliver", "Nina", "Daniel", "Leila", "Samuel", "Kavya", "Henry", "Elena", "Jack", "Tara", "Owen", "Mira",
         "Caleb", "Rhea", "Adrian", "Sana", "Felix", "Naomi", "Victor", "Alia", "Marco", "Ivy", "Rohan", "Chloe"]
LAST = ["Sharma", "Patel", "Nguyen", "Garcia", "Smith", "Khan", "Brown", "Rossi", "Kim", "Silva", "Wilson", "Reddy",
        "Muller", "Chen", "Lopez", "Singh", "Taylor", "Ali", "Moore", "Costa", "Clark", "Das", "Hughes", "Iyer",
        "Evans", "Gupta", "Bennett", "Rao", "Foster", "Mehta", "Ward", "Nair", "Cooper", "Bose", "Reid", "Shah"]
CHANNELS = ["Email", "Social Media", "Search", "Display", "Video", "Influencer"]

REVIEW_TEXT = {
    5: (["Absolutely love this", "Fantastic product", "Exceeded my expectations", "Best purchase this year",
         "Highly recommend it"],
        ["Works perfectly and arrived quickly.", "The quality is outstanding.", "Great value for the price.",
         "Customer support was helpful too.", "Easy to use right out of the box."]),
    4: (["Really good", "Very happy with it", "Solid choice"],
        ["A couple of small things could be better.", "Does what it promises.", "Good value overall.",
         "Delivery was quick and packaging was fine."]),
    3: (["It's okay", "Average experience", "Does the job"],
        ["Nothing special, but fine.", "Delivery was slower than expected.",
         "Quality is acceptable for the price.", "Some features are confusing."]),
    2: (["Disappointed", "Not what I expected", "Below average"],
        ["Stopped working after a few weeks.", "The description was misleading.",
         "Support was slow to respond.", "Feels cheaper than it looks."]),
    1: (["Terrible", "Awful experience", "Would not recommend"],
        ["Broke on the first day.", "Complete waste of money.", "Arrived damaged and nobody helped.",
         "Customer service never replied."]),
}
SEASON = {1: 0.90, 2: 0.90, 3: 1.00, 4: 1.00, 5: 1.05, 6: 1.00, 7: 0.95, 8: 0.95, 9: 1.00, 10: 1.10, 11: 1.25,
          12: 1.35}

issues = []  # (period, table, issue, rows)


def find_col(df, name):
    m = {c.lower(): c for c in df.columns}
    if name.lower() not in m:
        raise SystemExit(f"Column '{name}' not found. Columns are: {list(df.columns)}")
    return m[name.lower()]


def read(src, name):
    p = Path(src) / f"{name}.csv"
    if not p.exists():
        raise SystemExit(f"Missing {p}")
    return pd.read_csv(p, dtype=str, keep_default_na=False)


class IdGen:
    """Continues an ID column: integers carry on from the maximum, anything else becomes a UUID."""

    def __init__(self, series):
        vals = series[series != ""]
        nums = pd.to_numeric(vals, errors="coerce")
        self.numeric = len(vals) > 0 and bool(nums.notna().all())
        self.next = int(nums.max()) + 1 if self.numeric else 0

    def new(self, n, rng):
        if self.numeric:
            out = list(range(self.next, self.next + n))
            self.next += n
            return [str(i) for i in out]
        return [str(uuid.UUID(int=(int(rng.integers(0, 2 ** 62)) << 64) | int(rng.integers(0, 2 ** 62)),
                              version=4)) for _ in range(n)]


class DateFmt:
    def __init__(self, sample):
        self.with_time = any(len(x) > 10 for x in sample if x)

    def fmt(self, ts):
        return ts.strftime("%Y-%m-%d %H:%M:%S") if self.with_time else ts.strftime("%Y-%m-%d")


def rand_day(rng, period, late=False):
    start = period.start_time
    if late:
        start = (period - 1).start_time
        days = (period - 1).days_in_month
    else:
        days = period.days_in_month
    return start + pd.Timedelta(days=int(rng.integers(0, days)))


def choice_w(rng, items, weights, n):
    w = np.asarray(weights, dtype=float)
    return rng.choice(len(items), size=n, p=w / w.sum())


def num_style(series):
    """True when every non-blank value is a whole number."""
    v = pd.to_numeric(series[series != ""], errors="coerce").dropna()
    return len(v) > 0 and bool((v == v.round()).all())


def review_text(rng, rating, product):
    openers, details = REVIEW_TEXT[rating]
    opener = openers[int(rng.integers(0, len(openers)))]
    detail = details[int(rng.integers(0, len(details)))]
    if rng.random() < 0.10:                         # mixed signals: detail leans two steps the other way
        other = int(np.clip(rating + int(rng.choice([-2, 2])), 1, 5))
        d2 = REVIEW_TEXT[other][1]
        detail = d2[int(rng.integers(0, len(d2)))]
    text = f"{opener}. {detail}"
    if rating >= 3 and rng.random() < 0.25:
        text += f" The {product} arrived as described."
    return text


# ---------------------------------------------------------------- dirt (generated rows only)
def flag(period, table, issue, n):
    if n:
        issues.append((period, table, issue, int(n)))


def pick(rng, n, rate):
    return np.where(rng.random(n) < rate)[0]


def dirty(df, table, period, rng, rate, fmt_dates=None):
    """Return df (all strings) with controlled problems injected. Counts go to `issues`."""
    df = df.copy().astype(str)
    n = len(df)
    if n == 0 or rate <= 0:
        return df
    r = rate

    def setv(idx, col, values, issue):
        if len(idx):
            df.iloc[idx, df.columns.get_loc(col)] = values
            flag(period, table, issue, len(idx))

    if table == "customers":
        i = pick(rng, n, 0.02 * r)
        setv(i, "Email", [x.replace("@", "") for x in df.iloc[i]["Email"]], "email_missing_at")
        i = pick(rng, n, 0.02 * r)
        setv(i, "CustomerName", ["  " + x + " " for x in df.iloc[i]["CustomerName"]], "name_padded")
        i = pick(rng, n, 0.02 * r)
        setv(i, "Gender", [x.lower() for x in df.iloc[i]["Gender"]], "gender_lowercase")
        i = pick(rng, n, 0.015 * r)
        setv(i, "Age", "", "age_blank")
        i = pick(rng, n, 0.005 * r)
        setv(i, "Age", "150", "age_impossible")
    elif table == "engagement_data":
        i = pick(rng, n, 0.03 * r)
        setv(i, "ContentType", [x.lower().replace(" ", "") for x in df.iloc[i]["ContentType"]],
             "contenttype_variant_case")
        i = pick(rng, n, 0.01 * r)
        setv(i, "ViewsClicksCombined", "N/A", "views_clicks_malformed")
        i = pick(rng, n, 0.003 * r)
        setv(i, "Likes", "-5", "likes_negative")
        i = pick(rng, n, 0.01 * r)
        if len(i):
            alt = [pd.to_datetime(x).strftime("%d-%m-%Y") for x in df.iloc[i]["EngagementDate"]]
            setv(i, "EngagementDate", alt, "date_other_format")
    elif table == "customer_journey":
        i = pick(rng, n, 0.03 * r)
        setv(i, "Duration", "", "duration_blank")
        i = pick(rng, n, 0.02 * r)
        setv(i, "Stage", [x.lower() for x in df.iloc[i]["Stage"]], "stage_lowercase")
        i = pick(rng, n, 0.01 * r)
        setv(i, "Action", [x + " " for x in df.iloc[i]["Action"]], "action_trailing_space")
    elif table == "customer_reviews":
        i = pick(rng, n, 0.05 * r)
        setv(i, "ReviewText", [x.replace(". ", ".  ", 1) for x in df.iloc[i]["ReviewText"]], "review_double_space")
        i = pick(rng, n, 0.01 * r)
        setv(i, "ReviewText", "", "review_text_blank")
        i = pick(rng, n, 0.005 * r)
        setv(i, "Rating", rng.choice(["0", "6"], size=len(i)) if len(i) else [], "rating_out_of_range")
        i = pick(rng, n, 0.005 * r)
        setv(i, "Rating", "", "rating_blank")
    elif table == "orders":
        i = pick(rng, n, 0.005 * r)
        setv(i, "Quantity", rng.choice(["0", "-1"], size=len(i)) if len(i) else [], "quantity_not_positive")
        i = pick(rng, n, 0.005 * r)
        setv(i, "UnitPrice", "", "unitprice_blank")
        i = pick(rng, n, 0.02 * r)
        setv(i, "OrderStatus", [x.upper() for x in df.iloc[i]["OrderStatus"]], "status_uppercase")
        i = pick(rng, n, 0.003 * r)
        if len(i):
            fut = [(pd.to_datetime(x) + pd.DateOffset(years=2)).strftime("%Y-%m-%d") for x in df.iloc[i]["OrderDate"]]
            setv(i, "OrderDate", fut, "order_date_future")
    elif table == "campaign_spend":
        i = pick(rng, n, 0.003 * r)
        setv(i, "Spend", "-10", "spend_negative")
        i = pick(rng, n, 0.01 * r)
        setv(i, "Impressions", "", "impressions_blank")

    # exact duplicate rows (same keys)
    d = pick(rng, n, 0.01 * r)
    if len(d):
        df = pd.concat([df, df.iloc[d]], ignore_index=True)
        flag(period, table, "duplicate_row", len(d))
    return df


# ---------------------------------------------------------------- generator
class World:
    def __init__(self, src, seed):
        self.seed = seed
        self.o = {n: read(src, n) for n in
                  ["customers", "geography", "products", "customer_reviews", "customer_journey", "engagement_data"]}
        o = self.o
        self.cols = {n: list(df.columns) for n, df in o.items()}
        c = {n: {k: find_col(o[n], k) for k in cols} for n, cols in {
            "customers": ["CustomerID", "CustomerName", "Email", "Gender", "Age", "GeographyID"],
            "geography": ["GeographyID"],
            "products": ["ProductID", "ProductName", "Price"],
            "customer_reviews": ["ReviewID", "CustomerID", "ProductID", "ReviewDate", "Rating", "ReviewText"],
            "customer_journey": ["JourneyID", "CustomerID", "ProductID", "VisitDate", "Stage", "Action", "Duration"],
            "engagement_data": ["EngagementID", "ContentID", "CampaignID", "ProductID", "ContentType",
                                "ViewsClicksCombined", "Likes", "EngagementDate"]}.items()}
        self.c = c
        rng = np.random.default_rng([seed, 0])

        self.cust_ids = list(o["customers"][c["customers"]["CustomerID"]])
        self.cust_w = list(rng.lognormal(0, 0.8, len(self.cust_ids)))
        self.prod_ids = list(o["products"][c["products"]["ProductID"]])
        self.prod_names = dict(zip(self.prod_ids, o["products"][c["products"]["ProductName"]]))
        self.price = {p: float(v) for p, v in zip(self.prod_ids, o["products"][c["products"]["Price"]])}
        self.prod_w = list(rng.lognormal(0, 0.6, len(self.prod_ids)))
        self.geo_ids = list(o["geography"][c["geography"]["GeographyID"]])
        self.gender_vals = o["customers"][c["customers"]["Gender"]].tolist()
        ages = pd.to_numeric(o["customers"][c["customers"]["Age"]], errors="coerce").dropna()
        self.age_mu, self.age_sd = (ages.mean(), ages.std()) if len(ages) else (38, 12)

        self.ids = {
            "customer": IdGen(o["customers"][c["customers"]["CustomerID"]]),
            "review": IdGen(o["customer_reviews"][c["customer_reviews"]["ReviewID"]]),
            "journey": IdGen(o["customer_journey"][c["customer_journey"]["JourneyID"]]),
            "engagement": IdGen(o["engagement_data"][c["engagement_data"]["EngagementID"]]),
            "content": IdGen(o["engagement_data"][c["engagement_data"]["ContentID"]]),
            "campaign": IdGen(o["engagement_data"][c["engagement_data"]["CampaignID"]]),
        }
        self.ids["order"] = IdGen(pd.Series(["0"]))
        self.content_pool = list(dict.fromkeys(o["engagement_data"][c["engagement_data"]["ContentID"]]))

        eng, rev, jr = o["engagement_data"], o["customer_reviews"], o["customer_journey"]
        self.fmt = {
            "eng": DateFmt(eng[c["engagement_data"]["EngagementDate"]].head(50)),
            "rev": DateFmt(rev[c["customer_reviews"]["ReviewDate"]].head(50)),
            "jr": DateFmt(jr[c["customer_journey"]["VisitDate"]].head(50)),
        }
        dates = pd.concat([pd.to_datetime(eng[c["engagement_data"]["EngagementDate"]], errors="coerce"),
                           pd.to_datetime(rev[c["customer_reviews"]["ReviewDate"]], errors="coerce"),
                           pd.to_datetime(jr[c["customer_journey"]["VisitDate"]], errors="coerce")]).dropna()
        if dates.empty:
            raise SystemExit("Could not read any dates from engagement, review or journey files.")
        self.d_min, self.d_max = dates.min(), dates.max()
        span_months = max(1, (self.d_max.to_period("M") - self.d_min.to_period("M")).n + 1)
        self.span_months = span_months
        self.per_month = {"eng": len(eng) / span_months, "rev": len(rev) / span_months,
                          "jr": len(jr) / span_months}

        # learn distributions
        self.ctype = eng[c["engagement_data"]["ContentType"]].value_counts()
        vc = eng[c["engagement_data"]["ViewsClicksCombined"]].str.split("-", n=1, expand=True)
        if vc.shape[1] == 2:
            v = pd.to_numeric(vc[0], errors="coerce")
            k = pd.to_numeric(vc[1], errors="coerce")
            ok = v.notna() & k.notna() & (v > 0)
            v, k = v[ok], k[ok]
        else:
            v, k = pd.Series([1000.0]), pd.Series([50.0])
        self.log_v_mu, self.log_v_sd = float(np.log(v.clip(lower=1)).mean()), float(np.log(v.clip(lower=1)).std() or 0.5)
        self.ctr = float((k / v).clip(0, 1).mean()) if len(v) else 0.05
        likes = pd.to_numeric(eng[c["engagement_data"]["Likes"]], errors="coerce")
        self.like_rate = float((likes / k.reindex(likes.index).replace(0, np.nan)).median()) if len(k) else 0.3
        if not np.isfinite(self.like_rate) or self.like_rate <= 0:
            self.like_rate = 0.3
        self.pairs = jr[[c["customer_journey"]["Stage"], c["customer_journey"]["Action"]]].values.tolist()
        dur = pd.to_numeric(jr[c["customer_journey"]["Duration"]], errors="coerce").dropna()
        self.durs = dur.values if len(dur) else np.array([30.0])
        self.dur_int = num_style(jr[c["customer_journey"]["Duration"]])
        self.ratings = pd.to_numeric(rev[c["customer_reviews"]["Rating"]], errors="coerce").dropna()
        self.rate_vals = self.ratings.round().clip(1, 5).astype(int).value_counts().reindex(range(1, 6), fill_value=0)
        if self.rate_vals.sum() == 0:
            self.rate_vals[:] = [1, 1, 2, 3, 3]
        self.prod_rating_bias = {p: float(np.random.default_rng([seed, 7, i]).normal(0, 0.4))
                                 for i, p in enumerate(self.prod_ids)}
        self.campaigns = []   # dicts: id,name,channel,start,end,budget

        # campaigns implied by the original engagement data
        eng_dates = pd.to_datetime(eng[c["engagement_data"]["EngagementDate"]], errors="coerce")
        grp = pd.DataFrame({"cid": eng[c["engagement_data"]["CampaignID"]], "d": eng_dates}).dropna()
        for cid, g in grp.groupby("cid"):
            r2 = np.random.default_rng([seed, 11, zlib.crc32(str(cid).encode()) % 10 ** 6])
            start = g["d"].min() - pd.Timedelta(days=int(r2.integers(0, 10)))
            end = g["d"].max() + pd.Timedelta(days=int(r2.integers(0, 10)))
            self.campaigns.append(dict(id=str(cid), name=f"Campaign {cid}", channel=str(r2.choice(CHANNELS)),
                                       start=start, end=end,
                                       budget=round(float(r2.uniform(3000, 25000)), 2)))

    # ---- helpers
    def pick_cust(self, rng, n):
        return [self.cust_ids[i] for i in choice_w(rng, self.cust_ids, self.cust_w, n)]

    def pick_prod(self, rng, n):
        return [self.prod_ids[i] for i in choice_w(rng, self.prod_ids, self.prod_w, n)]

    def active_campaigns(self, period):
        s, e = period.start_time, period.end_time
        return [c for c in self.campaigns if c["start"] <= e and c["end"] >= s]

    def new_campaigns(self, rng, period):
        out = []
        n = int(rng.integers(2, 5))
        ids = self.ids["campaign"].new(n, rng)
        for cid in ids:
            start = rand_day(rng, period)
            dur = int(rng.integers(45, 121))
            camp = dict(id=cid, name=f"Campaign {cid}", channel=str(rng.choice(CHANNELS)), start=start,
                        end=start + pd.Timedelta(days=dur), budget=round(float(rng.uniform(4000, 30000)), 2))
            self.campaigns.append(camp)
            out.append(camp)
        return out

    def spend_rows(self, rng, period, camps):
        rows = []
        first, last = period.start_time, period.end_time.normalize()
        for c in camps:
            s, e = max(c["start"].normalize(), first), min(c["end"].normalize(), last)
            if e < s:
                continue
            total_days = max(1, (c["end"].normalize() - c["start"].normalize()).days + 1)
            base = c["budget"] / total_days
            for day in pd.date_range(s, e):
                spend = round(base * float(rng.lognormal(0, 0.2)), 2)
                imp = int(spend * rng.uniform(80, 200))
                clk = int(imp * rng.uniform(0.01, 0.05))
                rows.append([c["id"], day.strftime("%Y-%m-%d"), spend, imp, clk])
        return pd.DataFrame(rows, columns=["CampaignID", "SpendDate", "Spend", "Impressions", "Clicks"])

    def orders(self, rng, period, factor, customers=None, late=0.02):
        cust = customers if customers is not None else self.cust_ids
        n = max(1, int(round(len(cust) * 0.15 * factor)))
        oid = self.ids["order"].new(n, rng)
        idx = choice_w(rng, cust, [self.cust_w[self.cust_ids.index(x)] if x in self.cust_ids else 1 for x in cust], n)
        pr = self.pick_prod(rng, n)
        rows = []
        for k in range(n):
            p = pr[k]
            if rng.random() < 0.002 * len(self.prod_ids):
                self.price[p] = round(self.price[p] * float(rng.uniform(0.92, 1.10)), 2)
            qty = int(rng.choice([1, 2, 3, 4], p=[0.62, 0.25, 0.09, 0.04]))
            disc = int(rng.choice([0, 5, 10, 15, 20], p=[0.55, 0.15, 0.15, 0.10, 0.05]))
            status = str(rng.choice(["Completed", "Returned", "Cancelled"], p=[0.90, 0.06, 0.04]))
            when = rand_day(rng, period, late=bool(rng.random() < late))
            rows.append([oid[k], cust[idx[k]], p, when.strftime("%Y-%m-%d"), qty, self.price[p], disc, status])
        return pd.DataFrame(rows, columns=["OrderID", "CustomerID", "ProductID", "OrderDate", "Quantity",
                                           "UnitPrice", "DiscountPct", "OrderStatus"])

    def month(self, rng, period, rate):
        """One monthly batch: dict of table name -> DataFrame (strings, dirty rows included)."""
        factor = SEASON[period.month] * (1.02 ** (period - self.d_max.to_period("M")).n) * getattr(self, "scale", 1.0)
        p = str(period)
        out = {}

        # new customers
        n_new = int(rng.integers(10, 31))
        cid = self.ids["customer"].new(n_new, rng)
        rows = []
        for x in cid:
            f, l = str(rng.choice(FIRST)), str(rng.choice(LAST))
            age = int(np.clip(rng.normal(self.age_mu, self.age_sd), 18, 80))
            rows.append({"CustomerID": x, "CustomerName": f"{f} {l}", "Email": f"{f}.{l}{x}@example.com".lower(),
                         "Gender": str(rng.choice(self.gender_vals)), "Age": age,
                         "GeographyID": str(rng.choice(self.geo_ids))})
        cu = pd.DataFrame(rows)
        cu.columns = [self.c["customers"].get(c_, c_) for c_ in cu.columns]
        self.cust_ids += cid
        self.cust_w += list(rng.lognormal(0, 0.8, n_new))
        cu = cu[self.cols["customers"]]
        out["customers"] = dirty(cu, "customers", p, rng, rate)

        # campaigns and spend
        newc = self.new_campaigns(rng, period)
        camps = pd.DataFrame([[c["id"], c["name"], c["channel"], c["start"].strftime("%Y-%m-%d"),
                               c["end"].strftime("%Y-%m-%d"), c["budget"]] for c in newc],
                             columns=["CampaignID", "CampaignName", "Channel", "StartDate", "EndDate", "Budget"])
        out["campaigns"] = camps.astype(str)
        active = self.active_campaigns(period)
        out["campaign_spend"] = dirty(self.spend_rows(rng, period, active), "campaign_spend", p, rng, rate)

        # engagement
        n = max(1, int(round(self.per_month["eng"] * factor)))
        ct_items = list(self.ctype.index)
        cts = choice_w(rng, ct_items, self.ctype.values, n)
        camp_ids = [c["id"] for c in active]
        camp_w = [c["budget"] for c in active]
        cps = choice_w(rng, camp_ids, camp_w, n)
        views = np.maximum(1, rng.lognormal(self.log_v_mu, self.log_v_sd, n)).astype(int)
        clicks = np.minimum(views, (views * np.clip(rng.normal(self.ctr, self.ctr * 0.3, n), 0.001, 0.9)).astype(int))
        likes = (clicks * np.clip(rng.normal(self.like_rate, 0.1, n), 0.0, 1.0)).astype(int)
        contents = []
        for _ in range(n):
            if rng.random() < 0.3 or not self.content_pool:
                new_id = self.ids["content"].new(1, rng)[0]
                self.content_pool.append(new_id)
                contents.append(new_id)
            else:
                contents.append(self.content_pool[int(rng.integers(0, len(self.content_pool)))])
        eid = self.ids["engagement"].new(n, rng)
        pr = self.pick_prod(rng, n)
        late = rng.random(n) < 0.02
        e = pd.DataFrame({
            self.c["engagement_data"]["EngagementID"]: eid,
            self.c["engagement_data"]["ContentID"]: contents,
            self.c["engagement_data"]["CampaignID"]: [camp_ids[i] for i in cps],
            self.c["engagement_data"]["ProductID"]: pr,
            self.c["engagement_data"]["ContentType"]: [ct_items[i] for i in cts],
            self.c["engagement_data"]["ViewsClicksCombined"]: [f"{a}-{b}" for a, b in zip(views, clicks)],
            self.c["engagement_data"]["Likes"]: likes,
            self.c["engagement_data"]["EngagementDate"]: [self.fmt["eng"].fmt(rand_day(rng, period, bool(l)))
                                                         for l in late]})
        flag(p, "engagement_data", "late_arriving_prior_month", int(late.sum()))
        e = e[self.cols["engagement_data"]]
        e = e.rename(columns={self.c["engagement_data"][k]: k for k in self.c["engagement_data"]})
        e = dirty(e, "engagement_data", p, rng, rate)
        out["engagement_data"] = e.rename(columns={k: self.c["engagement_data"][k] for k in self.c["engagement_data"]})

        # journeys: sessions of 1-4 steps
        target = max(1, int(round(self.per_month["jr"] * factor)))
        rows = []
        while len(rows) < target:
            cu_, pr_ = self.pick_cust(rng, 1)[0], self.pick_prod(rng, 1)[0]
            day = rand_day(rng, period, late=bool(rng.random() < 0.02))
            for _ in range(int(rng.integers(1, 5))):
                st, ac = self.pairs[int(rng.integers(0, len(self.pairs)))]
                d = float(rng.choice(self.durs)) * float(rng.lognormal(0, 0.15))
                rows.append([cu_, pr_, self.fmt["jr"].fmt(day), st, ac,
                             str(int(round(d))) if self.dur_int else f"{d:.1f}"])
        rows = rows[:target]
        jid = self.ids["journey"].new(len(rows), rng)
        jc = self.c["customer_journey"]
        j = pd.DataFrame({jc["JourneyID"]: jid, jc["CustomerID"]: [r[0] for r in rows],
                          jc["ProductID"]: [r[1] for r in rows], jc["VisitDate"]: [r[2] for r in rows],
                          jc["Stage"]: [r[3] for r in rows], jc["Action"]: [r[4] for r in rows],
                          jc["Duration"]: [r[5] for r in rows]})[self.cols["customer_journey"]]
        j = j.rename(columns={jc[k]: k for k in jc})
        j = dirty(j, "customer_journey", p, rng, rate)
        out["customer_journey"] = j.rename(columns={k: jc[k] for k in jc})

        # reviews
        n = max(1, int(round(self.per_month["rev"] * factor)))
        rc = self.c["customer_reviews"]
        pr = self.pick_prod(rng, n)
        ratings = []
        for pid in pr:
            base = rng.choice([1, 2, 3, 4, 5], p=self.rate_vals.values / self.rate_vals.values.sum())
            ratings.append(int(np.clip(round(base + self.prod_rating_bias[pid]), 1, 5)))
        late = rng.random(n) < 0.02
        rv = pd.DataFrame({
            rc["ReviewID"]: self.ids["review"].new(n, rng),
            rc["CustomerID"]: self.pick_cust(rng, n),
            rc["ProductID"]: pr,
            rc["ReviewDate"]: [self.fmt["rev"].fmt(rand_day(rng, period, bool(l))) for l in late],
            rc["Rating"]: ratings,
            rc["ReviewText"]: [review_text(rng, r, self.prod_names[pid]) for r, pid in zip(ratings, pr)]})
        flag(p, "customer_reviews", "late_arriving_prior_month", int(late.sum()))
        rv = rv[self.cols["customer_reviews"]].rename(columns={rc[k]: k for k in rc})
        rv = dirty(rv, "customer_reviews", p, rng, rate)
        out["customer_reviews"] = rv.rename(columns={k: rc[k] for k in rc})

        # orders
        o_ = self.orders(rng, period, factor)
        out["orders"] = dirty(o_, "orders", p, rng, rate)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="marketing_data", help="folder with the six original CSV files")
    ap.add_argument("--out", default="data/extension", help="folder to write the extension into (not the source folder)")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply monthly volumes (2 = double)")
    ap.add_argument("--dirty-rate", type=float, default=1.0, help="1.0 = default, 0 = no dirty data, 2 = double")
    a = ap.parse_args()
    src, out = Path(a.src).expanduser(), Path(a.out).expanduser()
    if src.resolve() == out.resolve():
        raise SystemExit("--out must be a different folder from --src (originals are never modified)")

    w = World(src, a.seed)
    w.scale = a.scale
    hist = out / "history"
    hist.mkdir(parents=True, exist_ok=True)

    # history: campaigns, spend and orders over the original date span (load once)
    h_rng = np.random.default_rng([a.seed, 1])
    camps = pd.DataFrame([[c["id"], c["name"], c["channel"], c["start"].strftime("%Y-%m-%d"),
                           c["end"].strftime("%Y-%m-%d"), c["budget"]] for c in w.campaigns],
                         columns=["CampaignID", "CampaignName", "Channel", "StartDate", "EndDate", "Budget"])
    camps.astype(str).to_csv(hist / "campaigns.csv", index=False)
    spends, orders = [], []
    for per in pd.period_range(w.d_min.to_period("M"), w.d_max.to_period("M"), freq="M"):
        rng = np.random.default_rng([a.seed, 2, per.year * 12 + per.month])
        spends.append(w.spend_rows(rng, per, w.active_campaigns(per)))
        orders.append(w.orders(rng, per, SEASON[per.month], customers=list(w.o["customers"][w.c["customers"]["CustomerID"]])))
    sp = dirty(pd.concat(spends, ignore_index=True), "campaign_spend", "history", h_rng, a.dirty_rate)
    od = dirty(pd.concat(orders, ignore_index=True), "orders", "history", h_rng, a.dirty_rate)
    sp.to_csv(hist / "campaign_spend.csv", index=False)
    od.to_csv(hist / "orders.csv", index=False)

    # monthly batches
    first = (w.d_max.to_period("M") + 1)
    periods = [first + i for i in range(a.months)]
    counts = {}
    for i, per in enumerate(periods):
        rng = np.random.default_rng([a.seed, 3, i])
        tables = w.month(rng, per, a.dirty_rate)
        folder = out / "batches" / f"{per.year}_{per.month:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(folder / f"{name}.csv", index=False)
            counts[f"{per}/{name}"] = len(df)

    pd.DataFrame(issues, columns=["period", "table", "issue", "rows"]).to_csv(out / "_injected_issues.csv", index=False)
    (out / "_manifest.json").write_text(json.dumps({
        "note": "SYNTHETIC extension generated from the original CSVs; originals are untouched.",
        "seed": a.seed, "months": a.months, "dirty_rate": a.dirty_rate,
        "original_date_span": [str(w.d_min.date()), str(w.d_max.date())],
        "first_batch": str(periods[0]), "last_batch": str(periods[-1]),
        "history_rows": {"campaigns": len(camps), "campaign_spend": len(sp), "orders": len(od)},
        "batch_rows": counts}, indent=2))
    print(f"Original data spans {w.d_min.date()} to {w.d_max.date()}")
    print(f"History: campaigns {len(camps)}, campaign_spend {len(sp)}, orders {len(od)}")
    print(f"Batches: {periods[0]} to {periods[-1]} ({len(periods)} folders) in {out / 'batches'}")
    print(f"Injected issues are listed in {out / '_injected_issues.csv'}")


if __name__ == "__main__":
    main()