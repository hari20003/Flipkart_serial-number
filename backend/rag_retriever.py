import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _norm(s: str) -> str:
    return (s or "").strip()


class ExcelRAG:
    """
    Uses ONE excel file:
      - clean_sheet: good examples
      - error_sheets: bad examples (combined)
    Ignores L1/L2 and everything else.
    """

    def __init__(self, excel_path: str, clean_sheet: str, error_sheets: list[str]):
        self.excel_path = excel_path

        # ---- Clean sheet ----
        cdf = pd.read_excel(excel_path, sheet_name=clean_sheet).fillna("")
        for col in ["Information", "Description"]:
            if col not in cdf.columns:
                raise ValueError(f"Clean sheet '{clean_sheet}' missing column: {col}")

        self.clean_df = cdf
        self.clean_corpus = (cdf["Information"].astype(str) + " " + cdf["Description"].astype(str)).tolist()

        # ---- Error sheets (combine) ----
        edfs = []
        for sh in error_sheets:
            df = pd.read_excel(excel_path, sheet_name=sh).fillna("")

            # --- find error column even if name is different ---
            cols = {c.strip().lower(): c for c in df.columns}  # normalize headers

            required = ["information", "description"]
            for r in required:
                if r not in cols:
                    raise ValueError(f"Error sheet '{sh}' missing column: {r}")

            # Accept multiple possible names for Error column
            error_candidates = ["error", "errors", "error_type", "issue", "remark", "remarks", "reason"]
            error_col = None
            for cand in error_candidates:
                if cand in cols:
                    error_col = cols[cand]
                    break

            # If not found, create blank error column instead of crashing
            if error_col is None:
                df["Error"] = ""
                error_col = "Error"

            # keep only needed columns (ignore L1/L2 etc)
            df = df[[cols["information"], cols["description"], error_col]].copy()
            df.columns = ["Information", "Description", "Error"]
            df["__sheet__"] = sh
            edfs.append(df)

            # keep only needed cols (ignore L1/L2 etc)
            df = df[["Information", "Description", "Error"]].copy()
            df["__sheet__"] = sh
            edfs.append(df)

        self.err_df = pd.concat(edfs, ignore_index=True) if edfs else None
        self.err_corpus = (
            (self.err_df["Information"].astype(str) + " " + self.err_df["Description"].astype(str)).tolist()
            if self.err_df is not None else []
        )

        # ---- Vectorizer ----
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
        fit_corpus = list(self.clean_corpus) + list(self.err_corpus)
        self.vectorizer.fit(fit_corpus)

        self.clean_matrix = self.vectorizer.transform(self.clean_corpus)
        self.err_matrix = self.vectorizer.transform(self.err_corpus) if self.err_corpus else None

    def top_clean(self, query: str, k: int = 6):
        query = _norm(query)
        if not query:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.clean_matrix).flatten()
        idxs = sims.argsort()[::-1][:k]
        out = []
        for i in idxs:
            row = self.clean_df.iloc[int(i)]
            out.append({
                "feature": _norm(str(row.get("Information", ""))),
                "desc": _norm(str(row.get("Description", ""))),
            })
        return out

    def top_errors(self, query: str, k: int = 4):
        if self.err_df is None or self.err_matrix is None:
            return []
        query = _norm(query)
        if not query:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.err_matrix).flatten()
        idxs = sims.argsort()[::-1][:k]
        out = []
        for i in idxs:
            row = self.err_df.iloc[int(i)]
            out.append({
                "bad_feature": _norm(str(row.get("Information", ""))),
                "bad_desc": _norm(str(row.get("Description", ""))),
                "error": _norm(str(row.get("Error", ""))),
                "sheet": _norm(str(row.get("__sheet__", ""))),
            })
        return out


def format_clean_examples(examples: list[dict]) -> str:
    if not examples:
        return "None"
    lines = []
    for ex in examples:
        lines.append(f"- Feature: {ex['feature']}\n  Description: {ex['desc']}")
    return "\n".join(lines)


def format_error_examples(examples: list[dict]) -> str:
    if not examples:
        return "None"
    lines = []
    for ex in examples:
        lines.append(
            f"- BAD ({ex['sheet']}):\n"
            f"  Feature: {ex['bad_feature']}\n"
            f"  Description: {ex['bad_desc']}\n"
            f"  Error: {ex['error']}"
        )
    return "\n".join(lines)