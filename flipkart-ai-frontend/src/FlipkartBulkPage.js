import React, { useEffect, useMemo, useState } from "react";
import "./App.css";
import "./index.css";

import * as XLSX from "xlsx";
import { saveAs } from "file-saver";

const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";
//const API_BASE = "http://127.0.0.1:8000"; // ✅ change your backend port here (NO trailing slash)
const apiUrl = (path) =>
  `${API_BASE.replace(/\/+$/, "")}/${String(path).replace(/^\/+/, "")}`;

function clampInt(v, min, max) {
  const n = Number(v);
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, Math.floor(n)));
}




const newItem = () => ({
  id: (crypto?.randomUUID?.() || String(Date.now() + Math.random())).replaceAll(".", ""),
  fsn: "",
  humanDescription: "",
  aiDescription: "",
  extracted: null,
  evaluation: null,
  rawEvalJson: null,
  error: "",
  running: false,
});

export default function FlipkartBulkPage() {
  const [sop, setSop] = useState("");
  const [rubric, setRubric] = useState("");
  const [threshold, setThreshold] = useState(7);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const [count, setCount] = useState(1);
  const [step2, setStep2] = useState(false);

  const [globalErr, setGlobalErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [, setError] = useState("");
  const [items, setItems] = useState([newItem()]);

  const rows = useMemo(() => {
    const n = clampInt(count, 1, 50);
    return Array.from({ length: n }, (_, i) => i);
  }, [count]);

  // Ensure items length always >= rows length
  useEffect(() => {
    const n = rows.length;
    setItems((prev) => {
      const copy = [...prev];
      if (copy.length < n) {
        while (copy.length < n) copy.push(newItem());
      } else if (copy.length > n) {
        copy.length = n;
      }
      return copy;
    });

    setStep2(false);
    setGlobalErr("");
  }, [rows.length]);

  const anyLoading = useMemo(
    () => busy || items.some((x) => x?.running),
    [busy, items]
  );

  const updateItem = (idx, patch) => {
    setItems((prev) => {
      const copy = [...prev];
      // ensure slot exists
      if (!copy[idx]) copy[idx] = newItem();
      copy[idx] = { ...copy[idx], ...patch };
      return copy;
    });
  };

  const getItem = (idx) => items[idx] || newItem();

  const continueToStep2 = () => {
    setGlobalErr("");
    for (let i = 0; i < rows.length; i++) {
      const fsn = (getItem(i).fsn || "").trim();
      if (!fsn) {
        setGlobalErr(`FSN/URL missing in row ${i + 1}`);
        return;
      }
    }
    setStep2(true);
  };

 const runOne = async (idx) => {
  const it = getItem(idx);
  const fsn = (it?.fsn || "").trim();

  // reset row state
  updateItem(idx, { error: "", evaluation: null, rawEvalJson: null });

  // validations
  if (!fsn) {
    updateItem(idx, { error: "FSN/URL is required." });
    return;
  }
  if (!(it?.humanDescription || "").trim()) {
    updateItem(idx, { error: "Human description is required." });
    return;
  }

  updateItem(idx, { running: true });

  try {
    // -------------------- 1) GENERATE --------------------
    const genRes = await fetch(apiUrl("generate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sop, rubric, fsn }),
    });

    const genData = await genRes.json().catch(() => ({}));

    if (!genRes.ok) {
      const msg =
        typeof genData?.detail === "string"
          ? genData.detail
          : JSON.stringify(genData?.detail || "Generate failed");
      throw new Error(msg);
    }

    const extracted = genData?.extracted || {};
    const ai = (genData?.output || "").trim();

    // store generate result immediately
    updateItem(idx, { extracted, aiDescription: ai });

    if (!ai) {
      throw new Error(
        "Generate succeeded but output was empty (check backend formatting)."
      );
    }

    // -------------------- 2) EVALUATE --------------------
    const evalRes = await fetch(apiUrl("evaluate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sop,
        rubric,
        threshold: Number(threshold),
        human_features: [], // keep empty unless you add UI for it
        human_description: (it?.humanDescription || "").trim(),
        ai_description: ai,
        product_data: extracted,
      }),
    });

    const evalData = await evalRes.json().catch(() => ({}));

    if (!evalRes.ok) {
      const msg =
        typeof evalData?.detail === "string"
          ? evalData.detail
          : JSON.stringify(evalData?.detail || "Evaluation failed");
      throw new Error(msg);
    }

    // store evaluation
    updateItem(idx, {
      evaluation: evalData,
      rawEvalJson: JSON.stringify(evalData, null, 2),
    });
  } catch (err) {
    const msg = err?.message ? err.message : String(err);
    updateItem(idx, { error: msg });
  } finally {
    updateItem(idx, { running: false });
  }
};
const runAll = async () => {
  setBusy(true);
  setError("");

  try {
    for (let i = 0; i < items.length; i++) {
      const fsn = items[i]?.fsn?.trim();
      if (!fsn) continue;

      await runOne(i);

      // delay to reduce Flipkart blocking
      await sleep(3500);
    }
  } catch (err) {
    const msg = err?.message || String(err);
    setError(msg);
  } finally {
    setBusy(false);
  }
};


const summary = useMemo(() => {
  return rows.map((idx) => {
    const it = items[idx] || newItem();

    return {
      idx: idx + 1,
      fsn: it.fsn,
      humanOverall: it.evaluation?.human?.overall ?? null,
      humanPass: it.evaluation?.human?.pass ?? null,
      aiOverall: it.evaluation?.ai?.overall ?? null,
      aiPass: it.evaluation?.ai?.pass ?? null,
    };
  });
}, [items, rows]);

  // -------- Excel Export: S.No, FSN, AI Feature, AI Description --------
 function parseAiPairs(aiText, maxPairs = 6) {
  const text = (aiText || "").trim();
  if (!text) return [];

  // 1) Try tag-based parsing
  const pairsTagged = [];
  for (let i = 1; i <= maxPairs; i++) {
    const f = text.match(new RegExp(`<FEATURE_${i}>([\\s\\S]*?)</FEATURE_${i}>`, "i"));
    const d = text.match(new RegExp(`<DESC_${i}>([\\s\\S]*?)</DESC_${i}>`, "i"));
    if (f && d) {
      const feature = (f[1] || "").trim();
      const description = (d[1] || "").trim();
      if (feature || description) pairsTagged.push({ feature, description });
    } else {
      break;
    }
  }
  if (pairsTagged.length) return pairsTagged;

  // 2) Fallback: Plain text parsing (split by blank lines)
  // Expect blocks like: "Feature Title\nDescription..."
  const blocks = text.split(/\n\s*\n+/).map(b => b.trim()).filter(Boolean);

  const pairsPlain = [];
  for (let b of blocks) {
    const lines = b.split("\n").map(l => l.trim()).filter(Boolean);
    if (lines.length === 0) continue;

    const feature = lines[0];
    const description = lines.slice(1).join(" ").trim();

    // If no description line, keep entire block as description
    pairsPlain.push({
      feature: feature,
      description: description || ""
    });

    if (pairsPlain.length >= maxPairs) break;
  }

  // If still nothing sensible, return one pair with full text
  if (!pairsPlain.length) {
    return [{ feature: "", description: text }];
  }

  return pairsPlain;
}
const downloadExcel = () => {
  const rowsOut = [];

  items.forEach((it, idx) => {
    const fsn = (it?.fsn || "").trim() || "-";
    const aiText = (it?.aiDescription || "").trim();

    const pairs = parseAiPairs(aiText, 6);

    // Always create one row per pair (so FSN repeats)
    pairs.forEach((p, j) => {
      rowsOut.push({
        "S.No": idx + 1,
        "FSN": fsn,
        "AI Feature": (p.feature || "").trim(),
        "AI Description": (p.description || "").trim(),
      });
    });

    // If model returned empty somehow, still output one row
    if (pairs.length === 0) {
      rowsOut.push({
        "S.No": idx + 1,
        "FSN": fsn,
        "AI Feature": "",
        "AI Description": aiText,
      });
    }
  });

  const ws = XLSX.utils.json_to_sheet(rowsOut);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "AI_Output");

  const excelBuffer = XLSX.write(wb, { bookType: "xlsx", type: "array" });
  const blob = new Blob([excelBuffer], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });

  const name = `flipkart_ai_output_${new Date().toISOString().slice(0, 10)}.xlsx`;
  saveAs(blob, name);
};

  return (
    <div className="page">
      <div className="container">
        <div className="headerRow">
          <div>
            <h1 className="pageTitle">Flipkart Content Generation + Evaluation</h1>
            <div className="subTitle">Powered by Ollama</div>
          </div>
          <div className="rightPill">⚡ Powered by Ollama</div>
        </div>

        {globalErr ? (
          <div className="card errorCard">
            <b>Error:</b>
            <div style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>{globalErr}</div>
          </div>
        ) : null}

        <div className="card">
          <h2 className="cardTitle">Settings</h2>

          <div className="grid3">
            <div className="field">
              <label>PASS Cutoff (1–10)</label>
              <input
                type="number"
                min={1}
                max={10}
                step={0.1}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value || 7))}
              />
            </div>

            <div className="field">
              <label>Total Number of FSNs (1–50)</label>
              <input
                type="number"
                min={1}
                max={50}
                value={count}
                onChange={(e) => setCount(Number(e.target.value || 1))}
              />
              <div className="hint">Enter count and fill all FSNs.</div>
            </div>

            <div className="field">
              <label>Mode</label>
              <input value="Flipkart Product Pages" disabled />
              <div className="hint">FSN or full product URL accepted.</div>
            </div>
          </div>
        </div>

        <div className="card">
          <h2 className="cardTitle">Project SOP</h2>
          <textarea
            className="bigTextarea"
            value={sop}
            onChange={(e) => setSop(e.target.value)}
            placeholder="Paste SOP here..."
          />
        </div>

        <div className="card">
          <h2 className="cardTitle">Project Rubric</h2>
          <textarea
            className="bigTextarea"
            value={rubric}
            onChange={(e) => setRubric(e.target.value)}
            placeholder="Paste Rubric here..."
          />
        </div>

        {!step2 ? (
          <div className="card">
            <h2 className="cardTitle">Step 1: Enter FSNs / Product URLs</h2>
            <div className="hint">Fill all FSNs first, then continue.</div>

            <table className="table">
              <thead>
                <tr>
                  <th style={{ width: 120 }}>FSN</th>
                  <th>FSN / Product URL</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((idx) => {
                  const it = getItem(idx);
                  return (
                    <tr key={it.id || idx}>
                      <td>FSN {idx + 1}</td>
                      <td>
                        <input
                          value={it.fsn || ""}
                          onChange={(e) => updateItem(idx, { fsn: e.target.value })}
                          placeholder="Enter FSN or paste Flipkart URL"
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            <div className="actions">
              <button className="btnPrimary" disabled={anyLoading} onClick={continueToStep2}>
                Continue
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="card">
              <h2 className="cardTitle">Step 2: Enter Human Descriptions</h2>
              <div className="hint">Fill descriptions, then run all.</div>

              <table className="table">
                <thead>
                  <tr>
                    <th style={{ width: 120 }}>FSN</th>
                    <th>Human Description</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((idx) => {
                    const it = getItem(idx);
                    return (
                      <tr key={it.id || idx}>
                        <td>FSN {idx + 1}</td>
                        <td>
                          <textarea
                            className="rowTextarea"
                            value={it.humanDescription || ""}
                            onChange={(e) => updateItem(idx, { humanDescription: e.target.value })}
                            placeholder="Paste human description here..."
                          />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              <div className="actionsRow">
                <button className="btnPrimary" disabled={anyLoading} onClick={runAll}>
                  {anyLoading ? "Running..." : "Generate + Evaluate (All FSNs)"}
                </button>
                <button className="btnGhost" disabled={anyLoading} onClick={() => setStep2(false)}>
                  ⬅ Back
                </button>
              </div>
            </div>
<div className="card">
  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
    <h2 className="cardTitle">Results Summary</h2>
    <button className="btnGhost" onClick={downloadExcel} disabled={anyLoading}>
      Download Excel
    </button>
  </div>

  <table className="table">
    <thead>
      <tr>
        <th>#</th>
        <th>FSN/URL</th>
        <th>Human Score</th>
        <th>Human Status</th>
        <th>AI Score</th>
        <th>AI Status</th>
      </tr>
    </thead>

    <tbody>
      {summary.map((s) => (
        <tr key={s.idx}>
          <td>{s.idx}</td>
          <td style={{ wordBreak: "break-all" }}>{s.fsn || "-"}</td>

          <td>{s.humanOverall ?? "-"}</td>
          <td>
            {s.humanPass === null ? "-" : (
              <span className={`badge ${s.humanPass ? "pass" : "fail"}`}>
                {s.humanPass ? "PASS" : "FAIL"}
              </span>
            )}
          </td>

          <td>{s.aiOverall ?? "-"}</td>
          <td>
            {s.aiPass === null ? "-" : (
              <span className={`badge ${s.aiPass ? "pass" : "fail"}`}>
                {s.aiPass ? "PASS" : "FAIL"}
              </span>
            )}
          </td>
        </tr>
      ))}
    </tbody>
  </table>
</div>

            {rows.map((idx) => {
              const it = getItem(idx);
              return (
                <div className="card" key={it.id || idx}>
                  <div className="detailHeader">
                    <h2 className="cardTitle" style={{ margin: 0 }}>
                      FSN {idx + 1} Details
                    </h2>
                    <button
                      className="btnSmall"
                      disabled={anyLoading || it.running}
                      onClick={() => runOne(idx)}
                    >
                      {it.running ? "Running..." : "Run This FSN"}
                    </button>
                  </div>

                  {it.error ? (
                    <div className="errorInline">
                      <b>Error:</b> {it.error}
                    </div>
                  ) : null}

<div className="twoCol">
  <div className="panel">
    <h3 className="panelTitle">AI Description</h3>

    <div className="descBox">
      {(it.aiDescription || "").replace(/\r\n/g, "\n").split("\n\n")
        .filter(Boolean)
        .map((block, i) => {
          const lines = block.split("\n");

          return (
            <div key={i} style={{ marginBottom: "18px" }}>
              <div style={{ fontWeight: 600, marginBottom: "6px" }}>
                {lines[0]}
              </div>
              <div style={{ lineHeight: "1.7" }}>
                {lines.slice(1).join(" ")}
              </div>
            </div>
          );
        })}
    </div>
  </div>

  <div className="panel">
    <h3 className="panelTitle">Human Description</h3>

    <div
      className="descBox"
      style={{ whiteSpace: "pre-line", lineHeight: "1.7" }}
    >
      {it.humanDescription || "-"}
    </div>
  </div>
</div>
                  {it.rawEvalJson ? (
                    <div className="panel" style={{ marginTop: 14 }}>
                      <h3 className="panelTitle">Rubric Evaluation (Full JSON Output)</h3>
                      <pre className="jsonBox">{it.rawEvalJson}</pre>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </>
        )}
      </div>
    </div>
  );
}