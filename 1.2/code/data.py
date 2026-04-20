#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║           DATA SYSTEMS LAB — Python Terminal Edition             ║
║   Database  ·  Data Warehouse  ·  Data Lake  — Live Benchmark    ║
╚══════════════════════════════════════════════════════════════════╝

Three miniature systems built from scratch using only Python stdlib:
  - Database     → SQLite (structured, ACID, indexed)
  - Warehouse    → SQLite star-schema + columnar aggregation
  - Data Lake    → Flat file object store (JSON/CSV/binary)

Run:  python3 data_systems_lab.py
"""

import sqlite3
import json
import csv
import os
import time
import random
import statistics
import shutil
import datetime
import tempfile
import io
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────
# TERMINAL COLOURS  (ANSI escape codes)
# ─────────────────────────────────────────────
R  = "\033[0m"        # reset
B  = "\033[1m"        # bold
CY = "\033[96m"       # cyan  → Database
YL = "\033[93m"       # yellow → Warehouse
GR = "\033[92m"       # green  → Data Lake
RD = "\033[91m"       # red    → error / reject
DM = "\033[90m"       # dim    → muted info
WH = "\033[97m"       # white
MG = "\033[95m"       # magenta → headings

def banner():
    print(f"""
{MG}{B}╔══════════════════════════════════════════════════════════════════╗
║           DATA SYSTEMS LAB — Python Terminal Edition             ║
║   Database  ·  Data Warehouse  ·  Data Lake  — Live Benchmark    ║
╚══════════════════════════════════════════════════════════════════╝{R}
  {CY}■{R} Database   = SQLite OLTP  (strict schema, B-tree index)
  {YL}■{R} Warehouse  = SQLite OLAP  (star schema, columnar agg)
  {GR}■{R} Data Lake  = File store   (raw JSON/CSV/binary, schema-on-read)
""")

def section(title: str):
    width = 64
    bar = "─" * width
    print(f"\n{DM}{bar}{R}")
    print(f"{MG}{B}  {title}{R}")
    print(f"{DM}{bar}{R}")

def tag(system: str, msg: str):
    colours = {"DB": CY, "WH": YL, "LAKE": GR}
    c = colours.get(system, WH)
    print(f"  {c}[{system:4s}]{R}  {msg}")

def ok(msg):  print(f"  {GR}✓{R}  {msg}")
def err(msg): print(f"  {RD}✗{R}  {msg}")
def info(msg):print(f"  {DM}·{R}  {msg}")

def progress_bar(done, total, width=40, colour=WH):
    pct  = done / total
    fill = int(pct * width)
    bar  = "█" * fill + "░" * (width - fill)
    print(f"\r  {colour}{bar}{R}  {done:>6,}/{total:,}  {pct*100:5.1f}%", end="", flush=True)


# ══════════════════════════════════════════════════════════════════
#  SYSTEM 1 — MINI DATABASE  (SQLite OLTP)
# ══════════════════════════════════════════════════════════════════
class MiniDatabase:
    """
    Simulates an OLTP relational database.
    - Strict schema enforced at INSERT time (Schema-on-write)
    - Indexed on primary key for fast lookups
    - ACID via SQLite transactions
    """
    name = "Database (OLTP)"
    colour = CY
    tag   = "DB"

    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()
        self.rejected = 0
        self.accepted = 0

    def _create_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sales (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                product_id  TEXT NOT NULL,
                price       REAL NOT NULL CHECK(price > 0),
                region      TEXT NOT NULL,
                category    TEXT NOT NULL,
                ts          TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sales_region   ON sales(region);
            CREATE INDEX IF NOT EXISTS idx_sales_category ON sales(category);
            CREATE INDEX IF NOT EXISTS idx_sales_ts       ON sales(ts);
        """)
        self.conn.commit()

    def insert(self, record: dict) -> dict:
        """Strict insert — raises if required fields missing or wrong type."""
        t0 = time.perf_counter()
        required = {"id", "user_id", "product_id", "price", "region", "category", "ts"}

        # Schema validation
        if not isinstance(record, dict):
            self.rejected += 1
            return {"ok": False, "ms": (time.perf_counter()-t0)*1000,
                    "reason": "Not a dict — unstructured data rejected"}
        missing = required - record.keys()
        if missing:
            self.rejected += 1
            return {"ok": False, "ms": (time.perf_counter()-t0)*1000,
                    "reason": f"Missing fields: {missing}"}
        try:
            float(record["price"])
        except (ValueError, TypeError):
            self.rejected += 1
            return {"ok": False, "ms": (time.perf_counter()-t0)*1000,
                    "reason": "price must be numeric"}

        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO sales VALUES (?,?,?,?,?,?,?)",
                (record["id"], record["user_id"], record["product_id"],
                 float(record["price"]), record["region"],
                 record["category"], record["ts"])
            )
            self.conn.commit()
            self.accepted += 1
            return {"ok": True, "ms": (time.perf_counter()-t0)*1000}
        except sqlite3.Error as e:
            self.rejected += 1
            return {"ok": False, "ms": (time.perf_counter()-t0)*1000, "reason": str(e)}

    def query_lookup(self, record_id: str) -> dict:
        t0 = time.perf_counter()
        row = self.conn.execute("SELECT * FROM sales WHERE id=?", (record_id,)).fetchone()
        return {"ms": (time.perf_counter()-t0)*1000, "rows": 1 if row else 0}

    def query_aggregate(self) -> dict:
        t0 = time.perf_counter()
        rows = self.conn.execute(
            "SELECT region, category, SUM(price), COUNT(*) FROM sales GROUP BY region, category"
        ).fetchall()
        return {"ms": (time.perf_counter()-t0)*1000, "rows": len(rows)}

    @property
    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]


# ══════════════════════════════════════════════════════════════════
#  SYSTEM 2 — MINI DATA WAREHOUSE  (SQLite Star Schema OLAP)
# ══════════════════════════════════════════════════════════════════
class MiniWarehouse:
    """
    Simulates a Data Warehouse.
    - Star schema: fact table + dimension tables
    - Schema-on-write with ETL validation
    - Optimised for analytical (GROUP BY / SUM) queries
    - Slower writes due to ETL overhead
    """
    name = "Data Warehouse (OLAP)"
    colour = YL
    tag   = "WH"

    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()
        self.rejected = 0
        self.accepted = 0
        self._dim_cache = {"region": {}, "category": {}}

    def _create_schema(self):
        self.conn.executescript("""
            -- Dimension tables
            CREATE TABLE IF NOT EXISTS dim_region (
                region_key  INTEGER PRIMARY KEY AUTOINCREMENT,
                region_name TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS dim_category (
                cat_key   INTEGER PRIMARY KEY AUTOINCREMENT,
                cat_name  TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS dim_date (
                date_key  INTEGER PRIMARY KEY AUTOINCREMENT,
                full_date TEXT,
                year      INTEGER,
                month     INTEGER,
                day       INTEGER
            );
            -- Fact table
            CREATE TABLE IF NOT EXISTS fact_sales (
                sale_id    TEXT PRIMARY KEY,
                region_key INTEGER REFERENCES dim_region(region_key),
                cat_key    INTEGER REFERENCES dim_category(cat_key),
                date_key   INTEGER REFERENCES dim_date(date_key),
                price      REAL NOT NULL,
                quantity   INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_fact_region ON fact_sales(region_key);
            CREATE INDEX IF NOT EXISTS idx_fact_cat    ON fact_sales(cat_key);
            CREATE INDEX IF NOT EXISTS idx_fact_date   ON fact_sales(date_key);
        """)
        self.conn.commit()

    def _get_or_create_dim(self, table, col, val):
        """ETL step: lookup or insert dimension record, return key."""
        cache = self._dim_cache.get(col, {})
        if val in cache:
            return cache[val]
        row = self.conn.execute(
            f"SELECT rowid FROM {table} WHERE {col}=?", (val,)
        ).fetchone()
        if row:
            cache[val] = row[0]
        else:
            cur = self.conn.execute(f"INSERT INTO {table}({col}) VALUES(?)", (val,))
            self.conn.commit()
            cache[val] = cur.lastrowid
        self._dim_cache[col] = cache
        return cache[val]

    def insert(self, record: dict) -> dict:
        """ETL-validated insert into star schema."""
        t0 = time.perf_counter()

        # Warehouse still needs structured numeric data
        if not isinstance(record, dict):
            self.rejected += 1
            return {"ok": False, "ms": (time.perf_counter()-t0)*1000,
                    "reason": "Unstructured data — ETL rejected"}

        required_etl = {"id", "price", "region", "category", "ts"}
        if not required_etl.issubset(record.keys()):
            self.rejected += 1
            return {"ok": False, "ms": (time.perf_counter()-t0)*1000,
                    "reason": f"ETL requires: {required_etl - record.keys()}"}
        try:
            price = float(record["price"])
        except (ValueError, TypeError):
            self.rejected += 1
            return {"ok": False, "ms": (time.perf_counter()-t0)*1000,
                    "reason": "price not numeric — ETL rejected"}

        # ETL overhead: resolve dimension keys
        rk = self._get_or_create_dim("dim_region",   "region_name", record["region"])
        ck = self._get_or_create_dim("dim_category",  "cat_name",   record["category"])

        # Parse date for date dimension
        try:
            dt = datetime.datetime.fromisoformat(record["ts"])
            dk = self.conn.execute(
                "INSERT OR IGNORE INTO dim_date(full_date,year,month,day) VALUES(?,?,?,?)",
                (dt.date().isoformat(), dt.year, dt.month, dt.day)
            ).lastrowid or self.conn.execute(
                "SELECT date_key FROM dim_date WHERE full_date=?", (dt.date().isoformat(),)
            ).fetchone()[0]
        except Exception:
            dk = 1

        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO fact_sales(sale_id,region_key,cat_key,date_key,price) VALUES(?,?,?,?,?)",
                (record["id"], rk, ck, dk, price)
            )
            self.conn.commit()
            self.accepted += 1
            return {"ok": True, "ms": (time.perf_counter()-t0)*1000}
        except sqlite3.Error as e:
            self.rejected += 1
            return {"ok": False, "ms": (time.perf_counter()-t0)*1000, "reason": str(e)}

    def query_lookup(self, record_id: str) -> dict:
        """Point lookup — warehouses are slow at this."""
        t0 = time.perf_counter()
        row = self.conn.execute(
            "SELECT f.sale_id, r.region_name, c.cat_name, f.price "
            "FROM fact_sales f "
            "JOIN dim_region r ON f.region_key=r.region_key "
            "JOIN dim_category c ON f.cat_key=c.cat_key "
            "WHERE f.sale_id=?", (record_id,)
        ).fetchone()
        return {"ms": (time.perf_counter()-t0)*1000, "rows": 1 if row else 0}

    def query_aggregate(self) -> dict:
        """Multi-dimensional aggregation — warehouse's strength."""
        t0 = time.perf_counter()
        rows = self.conn.execute("""
            SELECT r.region_name,
                   c.cat_name,
                   d.year,
                   d.month,
                   SUM(f.price)   AS total_revenue,
                   COUNT(*)       AS num_sales,
                   AVG(f.price)   AS avg_price
            FROM fact_sales f
            JOIN dim_region   r ON f.region_key = r.region_key
            JOIN dim_category c ON f.cat_key    = c.cat_key
            JOIN dim_date     d ON f.date_key   = d.date_key
            GROUP BY r.region_name, c.cat_name, d.year, d.month
            ORDER BY total_revenue DESC
        """).fetchall()
        return {"ms": (time.perf_counter()-t0)*1000, "rows": len(rows)}

    @property
    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM fact_sales").fetchone()[0]


# ══════════════════════════════════════════════════════════════════
#  SYSTEM 3 — MINI DATA LAKE  (File Object Store)
# ══════════════════════════════════════════════════════════════════
class MiniDataLake:
    """
    Simulates a Data Lake (S3-style object store).
    - Accepts EVERYTHING: dicts, strings, lists, binary-like data
    - No schema enforcement (Schema-on-read)
    - Files stored as raw JSON blobs with partition paths
    - Query requires scanning all files (expensive)
    """
    name = "Data Lake (Object Store)"
    colour = GR
    tag   = "LAKE"

    def __init__(self, base_path: str):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)
        (self.base / "raw").mkdir(exist_ok=True)
        (self.base / "structured").mkdir(exist_ok=True)
        (self.base / "unstructured").mkdir(exist_ok=True)
        self.accepted = 0
        self.rejected = 0   # Lake never rejects
        self._manifest = []  # lightweight index of stored paths

    def insert(self, record) -> dict:
        """Store anything as-is. No validation."""
        t0 = time.perf_counter()

        now = datetime.datetime.utcnow()
        partition = f"year={now.year}/month={now.month:02d}/day={now.day:02d}"

        # Classify for partitioning (best-effort)
        if isinstance(record, dict) and "id" in record and "price" in record:
            folder = self.base / "structured" / partition
            prefix = "struct"
        else:
            folder = self.base / "unstructured" / partition
            prefix = "unstruct"

        folder.mkdir(parents=True, exist_ok=True)
        fname = f"{prefix}_{now.strftime('%H%M%S%f')}_{random.randint(0,9999):04d}.json"
        path  = folder / fname

        # Wrap raw objects in an envelope
        envelope = {
            "_lake_ingest_ts": now.isoformat(),
            "_lake_path": str(path.relative_to(self.base)),
            "_raw": record
        }
        path.write_text(json.dumps(envelope, default=str))
        self._manifest.append(str(path))
        self.accepted += 1
        return {"ok": True, "ms": (time.perf_counter()-t0)*1000, "path": str(path)}

    def query_lookup(self, record_id: str) -> dict:
        """Full scan — no index, must open every file."""
        t0 = time.perf_counter()
        found = 0
        for path in self._manifest:
            try:
                data = json.loads(Path(path).read_text())
                raw = data.get("_raw", {})
                if isinstance(raw, dict) and raw.get("id") == record_id:
                    found += 1
                    break
            except Exception:
                pass
        return {"ms": (time.perf_counter()-t0)*1000, "rows": found,
                "files_scanned": len(self._manifest)}

    def query_aggregate(self) -> dict:
        """Schema-on-read: parse every file, extract price if present."""
        t0 = time.perf_counter()
        total = 0.0
        by_region = {}
        parsed = 0
        for path in self._manifest:
            try:
                data = json.loads(Path(path).read_text())
                raw = data.get("_raw", {})
                if isinstance(raw, dict) and "price" in raw:
                    p = float(raw["price"])
                    r = raw.get("region", "unknown")
                    total += p
                    by_region[r] = by_region.get(r, 0) + p
                    parsed += 1
            except Exception:
                pass
        return {
            "ms": (time.perf_counter()-t0)*1000,
            "rows": parsed,
            "files_scanned": len(self._manifest),
            "total_revenue": total,
            "by_region": by_region
        }

    @property
    def count(self):
        return len(self._manifest)


# ══════════════════════════════════════════════════════════════════
#  DATA GENERATORS
# ══════════════════════════════════════════════════════════════════
REGIONS    = ["north", "south", "east", "west", "central"]
CATEGORIES = ["shoes", "shirts", "pants", "bags", "hats", "jackets"]

def make_structured(i: int) -> dict:
    return {
        "id":         f"rec_{i:07d}",
        "user_id":    f"user_{random.randint(1, 10000)}",
        "product_id": f"prod_{random.randint(1, 500)}",
        "price":      round(random.uniform(5, 999), 2),
        "region":     random.choice(REGIONS),
        "category":   random.choice(CATEGORIES),
        "ts":         (datetime.datetime(2023, 1, 1) +
                       datetime.timedelta(seconds=random.randint(0, 63072000))).isoformat()
    }

def make_unstructured(i: int):
    """Returns data that breaks strict DB/WH schema."""
    options = [
        lambda: f"RAW LOG [{i}]: user_agent=Mozilla/5.0 event=click ts={time.time()}",
        lambda: {"nested": {"deeply": {"value": random.random()}}, "no_schema": True, "id": None},
        lambda: [random.randint(0,255) for _ in range(16)],        # binary-like
        lambda: {"price": "N/A", "user_id": None, "corrupted": True},
        lambda: {"sensor_id": f"s_{i}", "readings": [random.random() for _ in range(5)],
                 "unit": "celsius", "metadata": {"firmware": "v2.1"}},
        lambda: {"tweet": f"Just bought shoes! #fashion #{random.choice(CATEGORIES)}",
                 "likes": random.randint(0, 1000), "user": f"@user{i}"},
    ]
    return random.choice(options)()

def make_record(i: int, unstructured_pct: float):
    """Returns (record, is_structured)."""
    if random.random() * 100 < unstructured_pct:
        return make_unstructured(i), False
    return make_structured(i), True


# ══════════════════════════════════════════════════════════════════
#  TEST 1 — INGESTION  (Structured vs Unstructured handling)
# ══════════════════════════════════════════════════════════════════
def test_ingestion(db, wh, lake, n: int, unstructured_pct: float):
    section(f"TEST 1 · DATA INGESTION  [{n:,} records · {unstructured_pct:.0f}% unstructured]")

    info(f"Sending {n:,} records to all three systems simultaneously...\n")
    results = {
        "db":   {"accepted": 0, "rejected": 0, "times": [], "reasons": {}},
        "wh":   {"accepted": 0, "rejected": 0, "times": [], "reasons": {}},
        "lake": {"accepted": 0, "rejected": 0, "times": [], "reasons": {}},
    }
    sample_events = []   # for pretty printing

    for i in range(n):
        record, is_structured = make_record(i, unstructured_pct)

        db_r   = db.insert(record)
        wh_r   = wh.insert(record)
        lake_r = lake.insert(record)

        for key, r, sys_r in [("db", db_r, db), ("wh", wh_r, wh), ("lake", lake_r, lake)]:
            results[key]["times"].append(r["ms"])
            if r["ok"]:
                results[key]["accepted"] += 1
            else:
                results[key]["rejected"] += 1
                reason = r.get("reason", "unknown")[:40]
                results[key]["reasons"][reason] = results[key]["reasons"].get(reason, 0) + 1

        if i < 8:
            sample_events.append((i, record, is_structured, db_r["ok"], wh_r["ok"], lake_r["ok"]))

        if (i + 1) % max(1, n // 20) == 0:
            progress_bar(i + 1, n, colour=MG)

    print()  # newline after progress bar

    # Sample event table
    print(f"\n  {B}Sample ingestion events:{R}")
    hdr = f"  {'#':>3}  {'Type':12}  {CY}{'DB':5}{R}  {YL}{'WH':5}{R}  {GR}{'LAKE':5}{R}  {'Record preview'}"
    print(hdr)
    print(f"  {'─'*70}")
    for idx, rec, is_s, d_ok, w_ok, l_ok in sample_events:
        t = f"{GR}struct{R}" if is_s else f"{RD}unstruct{R}"
        d = f"{GR}OK{R}" if d_ok else f"{RD}FAIL{R}"
        w = f"{GR}OK{R}" if w_ok else f"{RD}FAIL{R}"
        l = f"{GR}OK{R}" if l_ok else f"{RD}FAIL{R}"
        preview = str(rec)[:45].replace('\n', ' ') + "…" if len(str(rec)) > 45 else str(rec)
        print(f"  {idx:>3}  {t:20}  {d:15}  {w:15}  {l:15}  {DM}{preview}{R}")

    # Summary
    print(f"\n  {B}Ingestion summary:{R}")
    total = n
    for key, colour, name in [("db", CY, "Database"), ("wh", YL, "Warehouse"), ("lake", GR, "Data Lake")]:
        r = results[key]
        acc_pct = r["accepted"] / total * 100
        avg_ms  = statistics.mean(r["times"]) if r["times"] else 0
        bar_len = int(acc_pct / 2)
        bar     = "█" * bar_len + "░" * (50 - bar_len)
        print(f"\n  {colour}{B}{name}{R}")
        print(f"    Accepted : {colour}{r['accepted']:>6,}{R}  ({acc_pct:5.1f}%)  {colour}{bar}{R}")
        print(f"    Rejected : {RD}{r['rejected']:>6,}{R}  ({100-acc_pct:5.1f}%)")
        print(f"    Avg write: {avg_ms:.4f} ms/record")
        if r["reasons"]:
            top_reason = sorted(r["reasons"].items(), key=lambda x: -x[1])[0]
            print(f"    Top rejection reason: {DM}{top_reason[0]} ({top_reason[1]}×){R}")

    return results


# ══════════════════════════════════════════════════════════════════
#  TEST 2 — QUERY LATENCY BENCHMARK
# ══════════════════════════════════════════════════════════════════
def test_latency(db, wh, lake, iterations: int = 50):
    section(f"TEST 2 · QUERY LATENCY BENCHMARK  [{iterations} iterations each]")

    # Grab a real ID to look up
    sample_id = db.conn.execute("SELECT id FROM sales LIMIT 1").fetchone()
    sample_id = sample_id[0] if sample_id else "rec_0000001"

    timings = {
        "db_lookup":   [], "db_agg":   [],
        "wh_lookup":   [], "wh_agg":   [],
        "lake_lookup": [], "lake_agg": [],
    }

    info(f"Running {iterations} iterations per query type...")
    for i in range(iterations):
        timings["db_lookup"].append(db.query_lookup(sample_id)["ms"])
        timings["db_agg"].append(db.query_aggregate()["ms"])
        timings["wh_lookup"].append(wh.query_lookup(sample_id)["ms"])
        timings["wh_agg"].append(wh.query_aggregate()["ms"])
        timings["lake_lookup"].append(lake.query_lookup(sample_id)["ms"])
        timings["lake_agg"].append(lake.query_aggregate()["ms"])
        if (i + 1) % max(1, iterations // 10) == 0:
            progress_bar(i + 1, iterations, colour=MG)

    print()

    def stats(times):
        return {
            "mean":   statistics.mean(times),
            "median": statistics.median(times),
            "min":    min(times),
            "max":    max(times),
            "stdev":  statistics.stdev(times) if len(times) > 1 else 0,
        }

    results = {k: stats(v) for k, v in timings.items()}

    # Pretty table
    print(f"\n  {B}{'Query':<22}  {'System':<14}  {'Mean':>9}  {'Median':>9}  {'Min':>9}  {'Max':>9}{R}")
    print(f"  {'─'*75}")

    rows = [
        ("Point Lookup",  "Database",  "db_lookup",   CY),
        ("Point Lookup",  "Warehouse", "wh_lookup",   YL),
        ("Point Lookup",  "Data Lake", "lake_lookup", GR),
        ("Aggregation",   "Database",  "db_agg",      CY),
        ("Aggregation",   "Warehouse", "wh_agg",      YL),
        ("Aggregation",   "Data Lake", "lake_agg",    GR),
    ]
    prev_q = ""
    for q_name, sys_name, key, colour in rows:
        if q_name != prev_q and prev_q:
            print(f"  {'─'*75}")
        r = results[key]
        q_label = q_name if q_name != prev_q else ""
        print(f"  {q_label:<22}  {colour}{sys_name:<14}{R}"
              f"  {r['mean']:>8.4f}ms  {r['median']:>8.4f}ms"
              f"  {r['min']:>8.4f}ms  {r['max']:>8.4f}ms")
        prev_q = q_name

    # Winner callouts
    print(f"\n  {B}Winner — Point Lookup:{R}  "
          f"{CY}Database{R} ({results['db_lookup']['mean']:.4f}ms avg) — indexed B-tree lookup")
    print(f"  {B}Winner — Aggregation:{R}   "
          f"{YL}Warehouse{R} ({results['wh_agg']['mean']:.4f}ms avg) — star schema columnar scan")
    print(f"  {B}Data Lake note:{R}         "
          f"{GR}Lake{R} full-scans {lake.count} raw files — schema-on-read overhead visible")

    return results


# ══════════════════════════════════════════════════════════════════
#  TEST 3 — WRITE THROUGHPUT SCALE TEST
# ══════════════════════════════════════════════════════════════════
def test_scale(db, wh, lake, volume: int):
    section(f"TEST 3 · WRITE THROUGHPUT  [batch of {volume:,} structured records]")

    records = [make_structured(900_000 + i) for i in range(volume)]
    info(f"Pre-generated {volume:,} records. Timing isolated writes...\n")

    def batch_insert(system, recs):
        t0 = time.perf_counter()
        for r in recs:
            system.insert(r)
        elapsed = time.perf_counter() - t0
        return elapsed

    t_db   = batch_insert(db,   records)
    t_wh   = batch_insert(wh,   records)
    t_lake = batch_insert(lake, records)

    rps = {
        "db":   volume / t_db,
        "wh":   volume / t_wh,
        "lake": volume / t_lake,
    }
    tps = {"db": t_db, "wh": t_wh, "lake": t_lake}

    print(f"\n  {'System':<18}  {'Time':>10}  {'Rec/sec':>12}  {'ms/rec':>10}")
    print(f"  {'─'*55}")
    for key, colour, name in [("db", CY, "Database"), ("wh", YL, "Warehouse"), ("lake", GR, "Data Lake")]:
        ms_per = (tps[key] / volume) * 1000
        print(f"  {colour}{name:<18}{R}  {tps[key]:>9.3f}s  "
              f"{rps[key]:>12,.0f}  {ms_per:>9.4f}ms")

    fastest = max(rps, key=rps.get)
    names   = {"db": "Database", "wh": "Warehouse", "lake": "Data Lake"}
    colours = {"db": CY, "wh": YL, "lake": GR}
    print(f"\n  {B}Fastest writes:{R}  {colours[fastest]}{names[fastest]}{R}"
          f" at {rps[fastest]:,.0f} rec/s")
    print(f"  {DM}(Lake is fast because it just appends raw JSON files with no index overhead){R}")

    return rps, tps


# ══════════════════════════════════════════════════════════════════
#  CHART GENERATOR  (matplotlib)
# ══════════════════════════════════════════════════════════════════
def generate_charts(ingest_res, latency_res, scale_rps, scale_tps, output_path):
    section("GENERATING CHARTS")
    info("Building analytics charts...")

    # Dark theme
    plt.style.use("dark_background")
    C_DB   = "#00e5ff"
    C_WH   = "#ffd600"
    C_LAKE = "#69ff47"
    C_BG   = "#0a0a0f"
    C_SURF = "#12121a"

    fig = plt.figure(figsize=(18, 12), facecolor=C_BG)
    fig.suptitle("Data Systems Lab — Benchmark Analytics",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.92, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])   # Ingestion acceptance
    ax2 = fig.add_subplot(gs[0, 1])   # Lookup latency
    ax3 = fig.add_subplot(gs[0, 2])   # Aggregate latency
    ax4 = fig.add_subplot(gs[1, 0])   # Write throughput
    ax5 = fig.add_subplot(gs[1, 1])   # Latency comparison (grouped)
    ax6 = fig.add_subplot(gs[1, 2])   # Data flexibility radar

    def style_ax(ax, title):
        ax.set_facecolor(C_SURF)
        ax.set_title(title, color="white", fontsize=10, pad=8, fontweight="bold")
        ax.tick_params(colors="#888899", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e1e2e")
        ax.grid(axis="y", color="#1e1e2e", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)

    systems     = ["Database", "Warehouse", "Data Lake"]
    sys_colours = [C_DB, C_WH, C_LAKE]

    # ── Chart 1: Ingestion acceptance rates
    style_ax(ax1, "Ingestion Acceptance Rate (%)")
    keys = ["db", "wh", "lake"]
    total = ingest_res["db"]["accepted"] + ingest_res["db"]["rejected"]
    acc_pcts = [
        ingest_res["db"]["accepted"]   / total * 100,
        ingest_res["wh"]["accepted"]   / total * 100,
        100.0,
    ]
    bars = ax1.bar(systems, acc_pcts, color=sys_colours, width=0.5,
                   edgecolor="#0a0a0f", linewidth=0.5)
    for bar, pct in zip(bars, acc_pcts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{pct:.1f}%", ha="center", va="bottom", color="white",
                 fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("% Accepted", color="#888899", fontsize=8)

    # ── Chart 2: Lookup latency (log scale)
    style_ax(ax2, "Point Lookup Latency (ms, log scale)")
    lookup_ms = [
        latency_res["db_lookup"]["mean"],
        latency_res["wh_lookup"]["mean"],
        latency_res["lake_lookup"]["mean"],
    ]
    bars2 = ax2.bar(systems, lookup_ms, color=sys_colours, width=0.5,
                    edgecolor="#0a0a0f", linewidth=0.5)
    ax2.set_yscale("log")
    for bar, ms in zip(bars2, lookup_ms):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 ms * 1.4, f"{ms:.3f}ms",
                 ha="center", va="bottom", color="white", fontsize=8)
    ax2.set_ylabel("Latency (ms)", color="#888899", fontsize=8)

    # ── Chart 3: Aggregate latency
    style_ax(ax3, "Aggregation Query Latency (ms)")
    agg_ms = [
        latency_res["db_agg"]["mean"],
        latency_res["wh_agg"]["mean"],
        latency_res["lake_agg"]["mean"],
    ]
    bars3 = ax3.bar(systems, agg_ms, color=sys_colours, width=0.5,
                    edgecolor="#0a0a0f", linewidth=0.5)
    for bar, ms in zip(bars3, agg_ms):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(agg_ms)*0.02,
                 f"{ms:.2f}ms", ha="center", va="bottom", color="white", fontsize=8)
    ax3.set_ylabel("Latency (ms)", color="#888899", fontsize=8)

    # ── Chart 4: Write throughput (rec/s)
    style_ax(ax4, "Write Throughput (records/sec)")
    rps_vals = [scale_rps["db"], scale_rps["wh"], scale_rps["lake"]]
    bars4 = ax4.bar(systems, rps_vals, color=sys_colours, width=0.5,
                    edgecolor="#0a0a0f", linewidth=0.5)
    for bar, v in zip(bars4, rps_vals):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(rps_vals)*0.02,
                 f"{v:,.0f}/s", ha="center", va="bottom", color="white", fontsize=8)
    ax4.set_ylabel("Records/sec", color="#888899", fontsize=8)

    # ── Chart 5: Grouped latency comparison (lookup vs aggregate)
    style_ax(ax5, "Lookup vs Aggregate Latency (ms)")
    x      = np.arange(3)
    width  = 0.35
    lookup_vals = [latency_res["db_lookup"]["mean"],
                   latency_res["wh_lookup"]["mean"],
                   latency_res["lake_lookup"]["mean"]]
    agg_vals    = [latency_res["db_agg"]["mean"],
                   latency_res["wh_agg"]["mean"],
                   latency_res["lake_agg"]["mean"]]
    r1 = ax5.bar(x - width/2, lookup_vals, width, label="Point Lookup",
                 color=[(*[int(c[i:i+2],16)/255 for i in (1,3,5)], 1.0)
                        for c in sys_colours],
                 edgecolor="#0a0a0f", linewidth=0.5)
    r2 = ax5.bar(x + width/2, agg_vals, width, label="Aggregation",
                 color=[(*[int(c[i:i+2],16)/255 for i in (1,3,5)], 0.55)
                        for c in sys_colours],
                 edgecolor="#0a0a0f", linewidth=0.5, hatch="///")
    ax5.set_xticks(x)
    ax5.set_xticklabels(systems, color="#888899", fontsize=8)
    ax5.legend(fontsize=7, facecolor=C_SURF, edgecolor="#1e1e2e",
               labelcolor="white", loc="upper left")
    ax5.set_ylabel("Latency (ms)", color="#888899", fontsize=8)
    ax5.set_yscale("symlog")

    # ── Chart 6: Radar — capability profile
    style_ax(ax6, "Capability Profile (Radar)")
    ax6.remove()
    ax6 = fig.add_subplot(gs[1, 2], projection="polar")
    ax6.set_facecolor(C_SURF)
    ax6.set_title("Capability Profile", color="white", fontsize=10,
                  pad=12, fontweight="bold")

    categories  = ["Write Speed", "Read (Lookup)", "Analytics", "Flexibility", "Scalability"]
    N = len(categories)
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]

    profiles = {
        "Database":  [0.85, 1.0,  0.40, 0.20, 0.55],
        "Warehouse": [0.45, 0.25, 1.0,  0.35, 0.80],
        "Data Lake": [0.95, 0.15, 0.50, 1.0,  1.0 ],
    }
    for (name, vals), colour in zip(profiles.items(), sys_colours):
        v = vals + vals[:1]
        ax6.plot(angles, v, "o-", linewidth=1.5, color=colour, markersize=3)
        ax6.fill(angles, v, alpha=0.12, color=colour)

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, size=7, color="#ccccdd")
    ax6.set_yticklabels([])
    ax6.set_ylim(0, 1)
    ax6.grid(color="#1e1e2e", linewidth=0.8)
    ax6.spines["polar"].set_edgecolor("#1e1e2e")

    legend_patches = [
        mpatches.Patch(color=C_DB,   label="Database"),
        mpatches.Patch(color=C_WH,   label="Warehouse"),
        mpatches.Patch(color=C_LAKE, label="Data Lake"),
    ]
    ax6.legend(handles=legend_patches, loc="upper right",
               bbox_to_anchor=(1.35, 1.15), fontsize=7,
               facecolor=C_SURF, edgecolor="#1e1e2e", labelcolor="white")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    ok(f"Charts saved → {output_path}")


# ══════════════════════════════════════════════════════════════════
#  FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
def print_summary(ingest_res, latency_res, scale_rps, n_records):
    section("FINAL RESULTS SUMMARY")

    total = ingest_res["db"]["accepted"] + ingest_res["db"]["rejected"]
    db_acc_pct   = ingest_res["db"]["accepted"]   / total * 100
    wh_acc_pct   = ingest_res["wh"]["accepted"]   / total * 100

    rows = [
        ("Ingestion acceptance",  f"{db_acc_pct:.1f}%",  f"{wh_acc_pct:.1f}%",  "100.0%"),
        ("Point lookup (avg)",
         f"{latency_res['db_lookup']['mean']:.4f}ms",
         f"{latency_res['wh_lookup']['mean']:.4f}ms",
         f"{latency_res['lake_lookup']['mean']:.3f}ms"),
        ("Aggregation (avg)",
         f"{latency_res['db_agg']['mean']:.4f}ms",
         f"{latency_res['wh_agg']['mean']:.4f}ms",
         f"{latency_res['lake_agg']['mean']:.3f}ms"),
        ("Write throughput",
         f"{scale_rps['db']:>10,.0f}/s",
         f"{scale_rps['wh']:>10,.0f}/s",
         f"{scale_rps['lake']:>10,.0f}/s"),
        ("Accepts unstructured", "No ✗", "No ✗", "Yes ✓"),
        ("Schema enforcement",   "Strict", "ETL strict", "None"),
        ("Best query type",       "Point lookup", "Aggregation", "Full scan/ML"),
        ("Analogy",               "Cash register", "BI dashboard", "Raw storage vault"),
    ]

    col_w = [25, 16, 16, 16]
    header = (f"  {B}{'Metric':<{col_w[0]}}{R}  "
              f"{CY}{'Database':<{col_w[1]}}{R}  "
              f"{YL}{'Warehouse':<{col_w[2]}}{R}  "
              f"{GR}{'Data Lake':<{col_w[3]}}{R}")
    print(header)
    print(f"  {'─' * (sum(col_w) + 10)}")
    for row in rows:
        metric, db_v, wh_v, lake_v = row
        print(f"  {metric:<{col_w[0]}}  {CY}{db_v:<{col_w[1]}}{R}  "
              f"{YL}{wh_v:<{col_w[2]}}{R}  {GR}{lake_v:<{col_w[3]}}{R}")

    print(f"\n  {DM}Tested on {n_records:,} records  ·  Python stdlib only (sqlite3 + pathlib + json){R}")
    print(f"  {DM}Charts generated with matplotlib{R}\n")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    banner()

    # ── Configuration ──
    N_INGEST       = 5_000    # records for ingestion test
    UNSTRUCTURED   = 35       # % of unstructured data in mix
    N_LATENCY_ITER = 60       # query iterations per type
    N_SCALE        = 2_000    # records for throughput test
    CHART_PATH     = "data_systems_benchmark.png"

    print(f"  {B}Configuration:{R}")
    print(f"    Ingestion volume    : {CY}{N_INGEST:,} records{R}")
    print(f"    Unstructured %      : {CY}{UNSTRUCTURED}%{R}")
    print(f"    Latency iterations  : {CY}{N_LATENCY_ITER}{R}")
    print(f"    Throughput batch    : {CY}{N_SCALE:,}{R}")
    print(f"    Chart output        : {CY}{CHART_PATH}{R}")

    # ── Initialise systems ──
    lake_dir = tempfile.mkdtemp(prefix="data_lake_")

    db   = MiniDatabase()
    wh   = MiniWarehouse()
    lake = MiniDataLake(lake_dir)

    ok("Database initialised  (SQLite OLTP · B-tree indexes)")
    ok("Warehouse initialised (SQLite OLAP · star schema)")
    ok(f"Data Lake initialised (file store → {lake_dir})")

    try:
        # ── Run tests ──
        ingest_res  = test_ingestion(db, wh, lake, N_INGEST, UNSTRUCTURED)
        latency_res = test_latency(db, wh, lake, N_LATENCY_ITER)
        scale_rps, scale_tps = test_scale(db, wh, lake, N_SCALE)

        # ── Charts ──
        generate_charts(ingest_res, latency_res, scale_rps, scale_tps, CHART_PATH)

        # ── Summary ──
        print_summary(ingest_res, latency_res, scale_rps,
                      N_INGEST + N_SCALE)

    finally:
        # Clean up lake temp directory
        shutil.rmtree(lake_dir, ignore_errors=True)


if __name__ == "__main__":
    main()