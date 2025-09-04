# Create an updated Streamlit app that checks if each order has enough inventory to run
from textwrap import dedent
from pathlib import Path
import pandas as pd

root = Path("/mnt/data/bom_shortage_app")
root.mkdir(parents=True, exist_ok=True)

code = dedent("""
    # bom_shortage_canrun_streamlit.py
    # Streamlit app: Check if orders have enough inventory to run (independent & sequential modes)
    import os, io, warnings
    from typing import Optional, List, Tuple, Dict
    import numpy as np
    import pandas as pd
    import streamlit as st

    warnings.filterwarnings("ignore")

    st.set_page_config(page_title="BOM Can-Run Checker", layout="wide")
    st.title("✅ Can this order run with current inventory?")
    st.caption("Upload **Inventory**, **BOM**, and **Jobs/Orders**. The app will tell you if each job can run now, and what blocks it if not.")

    META_SHEET_TOKENS = {"readme","meta","dictionary","schema"}

    # ---------- Helpers ----------
    def _auto_pick(colnames, keywords) -> Optional[str]:
        for c in colnames:
            lc = str(c).lower()
            if any(k in lc for k in keywords):
                return c
        return colnames[0] if colnames else None

    def _num(s: pd.Series) -> pd.Series:
        return (s.astype(str)
                 .str.replace(r"[^\\"\\d\\.\\-eE]", "", regex=True)
                 .replace({"": np.nan})
                 .astype(float))

    def _read_uploaded(upload, label: str, required: bool) -> pd.DataFrame:
        if upload is None:
            if required:
                st.info(f"Upload **{label}** to proceed.")
                st.stop()
            return pd.DataFrame()

        try:
            name = getattr(upload, "name", None) or "<unnamed>"
            ext = os.path.splitext(name)[1].lower()
            data = upload.getvalue()
            if not data:
                if required:
                    st.error(f"**{label}** appears to be empty.")
                    st.stop()
                return pd.DataFrame()

            buf = io.BytesIO(data)
            if ext == ".csv":
                df = pd.read_csv(buf)
            elif ext in {".xlsx",".xls"}:
                xls = pd.ExcelFile(buf)
                pick = next((s for s in xls.sheet_names if not any(t in s.lower() for t in META_SHEET_TOKENS)),
                            xls.sheet_names[0])
                buf.seek(0)
                df = pd.read_excel(buf, sheet_name=pick)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

            if df is None or (df.empty and required):
                st.error(f"**{label}** has no rows.")
                st.stop()
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            if required:
                st.error(f"Couldn't read **{label}**: {e}")
                st.stop()
            return pd.DataFrame()

    # ---------- Sidebar uploads ----------
    with st.sidebar:
        st.header("1) Upload files")
        inv_up  = st.file_uploader("Inventory (Stock Report) (.csv, .xlsx)", type=["csv","xlsx","xls"])
        bom_up  = st.file_uploader("BOM (.csv, .xlsx)", type=["csv","xlsx","xls"])
        jobs_up = st.file_uploader("Jobs / Orders (.csv, .xlsx)", type=["csv","xlsx","xls"])

    df_inv  = _read_uploaded(inv_up,  "Inventory", required=True)
    df_bom  = _read_uploaded(bom_up,  "BOM", required=True)
    df_jobs = _read_uploaded(jobs_up, "Jobs/Orders", required=True)

    # ---------- Preview ----------
    st.markdown("### Preview uploads")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Inventory**")
        st.dataframe(df_inv.head(10), use_container_width=True)
    with c2:
        st.write("**BOM**")
        st.dataframe(df_bom.head(10), use_container_width=True)
    with c3:
        st.write("**Jobs/Orders**")
        st.dataframe(df_jobs.head(10), use_container_width=True)

    # ---------- 2) Map columns ----------
    st.markdown("### 2) Map columns")

    inv_cols = list(df_inv.columns)
    inv_item = st.selectbox("Inventory → Component/SKU", inv_cols,
        index=inv_cols.index(_auto_pick(inv_cols, ["item","component","sku","part"])) if _auto_pick(inv_cols, ["item","component","sku","part"]) in inv_cols else 0)
    inv_onhand = st.selectbox("Inventory → On Hand Qty", inv_cols,
        index=inv_cols.index(_auto_pick(inv_cols, ["qty on hand","on hand","stock","qty"])) if _auto_pick(inv_cols, ["qty on hand","on hand","stock","qty"]) in inv_cols else 0)
    inv_committed = st.selectbox("Inventory → Committed Qty (sales/etc.)", inv_cols,
        index=inv_cols.index(_auto_pick(inv_cols, ["qty commited","qty committed","committed"])) if _auto_pick(inv_cols, ["qty commited","qty committed","committed"]) in inv_cols else 0)
    inv_wo_hard = st.selectbox("Inventory → WO Hard Committed (optional)", ["<none>"]+inv_cols,
        index=(inv_cols.index(_auto_pick(inv_cols, ["wo hard","work order hard"]))+1) if _auto_pick(inv_cols, ["wo hard","work order hard"]) in inv_cols else 0)

    bom_cols = list(df_bom.columns)
    bom_fg   = st.selectbox("BOM → FG / Parent SKU", bom_cols,
        index=bom_cols.index(_auto_pick(bom_cols, ["fg","parent","finish","product","sku"])) if _auto_pick(bom_cols, ["fg","parent","finish","product","sku"]) in bom_cols else 0)
    bom_comp = st.selectbox("BOM → Component SKU", bom_cols,
        index=bom_cols.index(_auto_pick(bom_cols, ["component","part","material","sku"])) if _auto_pick(bom_cols, ["component","part","material","sku"]) in bom_cols else 0)
    bom_qty  = st.selectbox("BOM → Qty per FG", bom_cols,
        index=bom_cols.index(_auto_pick(bom_cols, ["qty per","qty","usage","quantity"])) if _auto_pick(bom_cols, ["qty per","qty","usage","quantity"]) in bom_cols else 0)

    jobs_cols = list(df_jobs.columns)
    jobs_id = st.selectbox("Jobs → Job/Run ID", jobs_cols,
        index=jobs_cols.index(_auto_pick(jobs_cols, ["job","run","wo","work order","batch","id"])) if _auto_pick(jobs_cols, ["job","run","wo","work order","batch","id"]) in jobs_cols else 0)
    jobs_fg = st.selectbox("Jobs → FG SKU", jobs_cols,
        index=jobs_cols.index(_auto_pick(jobs_cols, ["fg","parent","finish","product","sku"])) if _auto_pick(jobs_cols, ["fg","parent","finish","product","sku"]) in jobs_cols else 0)
    jobs_qty = st.selectbox("Jobs → FG Qty to build", jobs_cols,
        index=jobs_cols.index(_auto_pick(jobs_cols, ["qty","quantity","build","planned"])) if _auto_pick(jobs_cols, ["qty","quantity","build","planned"]) in jobs_cols else 0)

    # ---------- 3) Normalize ----------
    inv = pd.DataFrame({
        "component": df_inv[inv_item].astype(str).str.strip(),
        "on_hand": _num(df_inv[inv_onhand]),
        "committed": _num(df_inv[inv_committed]),
    })
    if inv_wo_hard != "<none>" and inv_wo_hard in df_inv.columns:
        inv["wo_hard"] = _num(df_inv[inv_wo_hard])
    else:
        inv["wo_hard"] = 0.0
    inv["available"] = inv["on_hand"].fillna(0) - inv["committed"].fillna(0) - inv["wo_hard"].fillna(0)

    bom = pd.DataFrame({
        "fg": df_bom[bom_fg].astype(str).str.strip(),
        "component": df_bom[bom_comp].astype(str).str.strip(),
        "qty_per": _num(df_bom[bom_qty])
    }).groupby(["fg","component"], as_index=False)["qty_per"].sum()

    orders = pd.DataFrame({
        "job_id": df_jobs[jobs_id].astype(str),
        "fg": df_jobs[jobs_fg].astype(str).str.strip(),
        "fg_qty": _num(df_jobs[jobs_qty])
    })

    if bom.empty or orders.empty:
        st.error("Your BOM or Jobs/Orders appears to have no rows after mapping. Please verify column choices.")
        st.stop()

    # Build per-job component requirements
    req = (orders.merge(bom, on="fg", how="left"))
    if req["qty_per"].isna().all():
        st.warning("No BOM rows matched your Jobs/Orders FG values. Check your FG column mappings.")
        st.stop()
    req["comp_req_qty"] = req["fg_qty"].fillna(0) * req["qty_per"].fillna(0)
    req_job_comp = (req.groupby(["job_id","fg","component"], as_index=False)["comp_req_qty"].sum())

    # ---------- 4) Check modes ----------
    st.markdown("### 3) Check mode")
    mode = st.radio("How do you want to check feasibility?", ["Independent (ignore other jobs)", "Sequential allocation (reserve inventory in job order)"])

    def independent_check(req_jc: pd.DataFrame, inv_avail: pd.DataFrame) -> pd.DataFrame:
        # Available per component (static)
        avail = inv_avail.set_index("component")["available"].fillna(0)
        records = []
        for job_id, grp in req_jc.groupby("job_id"):
            blockers = []
            can_run = True
            for _, r in grp.iterrows():
                need = r["comp_req_qty"]
                have = avail.get(r["component"], 0.0)
                if need > have + 1e-9:
                    can_run = False
                    blockers.append(f"{r['component']} (need {need:.2f}, have {have:.2f})")
            records.append({"Job ID": job_id, "FG": grp["fg"].iloc[0], "Qty": grp["comp_req_qty"].sum(),
                            "Can Run Now?": "Yes" if can_run else "No",
                            "Blocking Components": ", ".join(blockers)})
        return pd.DataFrame(records)

    def max_buildable_for_fg(fg: str, qty_per_df: pd.DataFrame, inv_avail: pd.DataFrame) -> float:
        sub = qty_per_df[qty_per_df["fg"] == fg]
        if sub.empty:
            return 0.0
        merged = sub.merge(inv_avail[["component","available"]], on="component", how="left").fillna({"available":0})
        # floor(available / qty_per) per component, then min
        with np.errstate(divide="ignore", invalid="ignore"):
            possible = np.floor(merged["available"] / merged["qty_per"])
        if possible.empty:
            return 0.0
        return float(possible.min())

    def sequential_check(req_jc: pd.DataFrame, inv_avail: pd.DataFrame, order: List[str]) -> pd.DataFrame:
        inv_map = inv_avail.set_index("component")["available"].fillna(0).to_dict()
        qty_per_df = req_jc.groupby(["fg","component"], as_index=False)["comp_req_qty"].sum()
        # Convert comp_req_qty per job to required per component for that job
        out = []
        for job_id in order:
            grp = req_jc[req_jc["job_id"] == job_id]
            blockers = []
            can_run = True
            # Check
            for _, r in grp.iterrows():
                need = float(r["comp_req_qty"])
                have = float(inv_map.get(r["component"], 0.0))
                if need > have + 1e-9:
                    can_run = False
                    blockers.append(f"{r['component']} (need {need:.2f}, have {have:.2f})")
            if can_run:
                # Reserve inventory
                for _, r in grp.iterrows():
                    inv_map[r["component"]] = float(inv_map.get(r["component"], 0.0)) - float(r["comp_req_qty"])
            fg = grp["fg"].iloc[0] if not grp.empty else ""
            # Max buildable with CURRENT inventory snapshot
            # Derive qty_per from grp by dividing comp_req_qty by job qty
            # safer approach: use BOM directly
            mb = max_buildable_for_fg(fg, qty_per_df.rename(columns={"comp_req_qty":"qty_per"}), 
                                      pd.DataFrame({"component": list(inv_map.keys()), "available": list(inv_map.values())}))
            out.append({"Job ID": job_id, "FG": fg, "Can Run (Seq)?": "Yes" if can_run else "No",
                        "Blocking Components": ", ".join(blockers), "Est. Max Buildable Now": mb})
        return pd.DataFrame(out)

    job_order = list(orders["job_id"])
    if mode.startswith("Independent"):
        res = independent_check(req_job_comp, inv)
        st.markdown("### Results — Independent")
        st.dataframe(res, use_container_width=True)
        st.download_button("Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                           "can_run_independent.csv", "text/csv")
    else:
        st.info("Sequential allocation uses the current upload order of jobs. If you need custom priority, sort your CSV before upload.")
        res = sequential_check(req_job_comp, inv, job_order)
        st.markdown("### Results — Sequential allocation")
        st.dataframe(res, use_container_width=True)
        st.download_button("Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                           "can_run_sequential.csv", "text/csv")

    # Also expose a quick per-FG 'max buildable' table from current inventory
    st.markdown("### Max buildable units per FG (from current inventory)")
    qty_per = bom.rename(columns={"qty_per":"per"})
    rows = []
    for fg in sorted(qty_per["fg"].unique()):
        merged = qty_per[qty_per["fg"]==fg].merge(inv[["component","available"]], on="component", how="left").fillna({"available":0})
        with np.errstate(divide="ignore", invalid="ignore"):
            possible = np.floor(merged["available"] / merged["per"])
        max_build = 0.0 if possible.empty else float(possible.min())
        rows.append({"FG": fg, "Max Buildable Now": max_build})
    fg_table = pd.DataFrame(rows).sort_values("FG")
    st.dataframe(fg_table, use_container_width=True)
    st.download_button("Download FG max buildable (CSV)", fg_table.to_csv(index=False).encode("utf-8"),
                       "fg_max_buildable.csv", "text/csv")
""")

(app2 := root / "bom_shortage_canrun_streamlit.py").write_text(code, encoding="utf-8")

# Provide a lightweight CLI script too
cli_code = dedent("""
    # can_run_check.py
    # CLI utility: given inventory.csv, bom.csv, jobs.csv -> prints feasibility per job and a CSV
    import sys
    import pandas as pd
    import numpy as np
    from pathlib import Path

    def _num(s: pd.Series) -> pd.Series:
        return (s.astype(str)
                 .str.replace(r"[^\\d\\.\\-eE]", "", regex=True)
                 .replace({"": np.nan})
                 .astype(float))

    def main(inv_path, bom_path, jobs_path, out_csv="can_run_independent.csv"):
        inv = pd.read_csv(inv_path)
        bom = pd.read_csv(bom_path)
        jobs = pd.read_csv(jobs_path)

        # Heuristic column picks
        def pick(cols, alts):
            l = [c for c in cols if any(k in c.lower() for k in alts)]
            return l[0] if l else cols[0]

        inv_item = pick(inv.columns, ["item","component","sku","part"])
        inv_onhand = pick(inv.columns, ["qty on hand","on hand","stock","qty"])
        inv_committed = pick(inv.columns, ["qty commited","qty committed","committed"])
        inv["available"] = _num(inv[inv_onhand]) - _num(inv[inv_committed])

        bom_fg = pick(bom.columns, ["fg","parent","finish","product","sku"])
        bom_comp = pick(bom.columns, ["component","part","material","sku"])
        bom_qty = pick(bom.columns, ["qty per","qty","usage","quantity"])
        bom = bom.rename(columns={bom_fg:"fg", bom_comp:"component", bom_qty:"qty_per"})
        bom["qty_per"] = _num(bom["qty_per"])
        bom = bom.groupby(["fg","component"], as_index=False)["qty_per"].sum()

        jobs_id = pick(jobs.columns, ["job","run","wo","work order","batch","id"])
        jobs_fg = pick(jobs.columns, ["fg","parent","finish","product","sku"])
        jobs_qty = pick(jobs.columns, ["qty","quantity","build","planned"])
        orders = jobs.rename(columns={jobs_id:"job_id", jobs_fg:"fg", jobs_qty:"fg_qty"})
        orders["fg_qty"] = _num(orders["fg_qty"])

        req = orders.merge(bom, on="fg", how="left")
        req["comp_req_qty"] = req["fg_qty"].fillna(0) * req["qty_per"].fillna(0)
        req_job_comp = (req.groupby(["job_id","fg","component"], as_index=False)["comp_req_qty"].sum())

        avail = inv.set_index("component")["available"].fillna(0)

        records = []
        for job_id, grp in req_job_comp.groupby("job_id"):
            blockers = []
            can_run = True
            for _, r in grp.iterrows():
                need = r["comp_req_qty"]
                have = avail.get(r["component"], 0.0)
                if need > have + 1e-9:
                    can_run = False
                    blockers.append(f"{r['component']} (need {need:.2f}, have {have:.2f})")
            records.append({"Job ID": job_id, "FG": grp["fg"].iloc[0], "Qty": grp["comp_req_qty"].sum(),
                            "Can Run Now?": "Yes" if can_run else "No",
                            "Blocking Components": ", ".join(blockers)})
        out = pd.DataFrame(records)
        out.to_csv(out_csv, index=False)
        print(out)
        print(f"Saved -> {out_csv}")

    if __name__ == "__main__":
        if len(sys.argv) < 4:
            print("Usage: python can_run_check.py inventory.csv bom.csv jobs.csv [out.csv]")
            raise SystemExit(2)
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "can_run_independent.csv")
""")
(cli_path := root / "can_run_check.py").write_text(cli_code, encoding="utf-8")

# Reuse sample CSVs already created in the previous step if present; else create minimal ones here
inv_path = root / "sample_inventory.csv"
bom_path = root / "sample_bom.csv"
jobs_path = root / "sample_jobs.csv"
if not inv_path.exists():
    pd.DataFrame({
        "Item": ["CAP-10uF", "RES-1k", "IC-555", "BRKT-01"],
        "Qty On Hand": [150, 500, 80, 40],
        "Qty Committed": [20, 50, 10, 5],
    }).to_csv(inv_path, index=False)
if not bom_path.exists():
    pd.DataFrame({
        "FG": ["WIDGET-A","WIDGET-A","WIDGET-A","WIDGET-B","WIDGET-B"],
        "Component": ["CAP-10uF","RES-1k","IC-555","RES-1k","BRKT-01"],
        "Qty per": [2, 4, 1, 2, 1]
    }).to_csv(bom_path, index=False)
if not jobs_path.exists():
    pd.DataFrame({
        "Job": ["JOB-1001","JOB-1002","JOB-1003"],
        "FG": ["WIDGET-A","WIDGET-A","WIDGET-B"],
        "Qty": [30, 25, 40]
    }).to_csv(jobs_path, index=False)

sorted([str(p) for p in root.iterdir()])



