import { useEffect, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  analyzeDatasetProfile,
  clusterDataset,
  deleteDataset,
  detectAnomalies,
  forecastDataset,
  getDatasetAutoAnalysis,
  getDatasetProfile,
  listDatasets,
  uploadDataset,
} from "./api/datasets.js";
import {
  confirmExperiment,
  createExperiment,
  generatePipelinePlan,
  trainExperiment,
  evaluateExperiment,
  optimizeExperiment,
  getExplainability,
  createPredictions,
  createReport,
  createPdfReport,
  createReflection,
  getReportHistory,
  updateAdaptivePipeline,
} from "./api/experiments.js";
import { API_BASE_URL } from "./api/client.js";
import {
  clearAccessToken,
  getAccessToken,
  setAccessToken,
} from "./api/client.js";
import {
  getCurrentUser,
  login,
  loginWithGoogleCredential,
  register,
} from "./api/auth.js";
import {
  startExperimentJob,
  getExperimentDetail,
  getExperimentHistory,
} from "./api/jobs.js";
import { useJobPolling } from "./hooks/useJobPolling.js";
import { getBackendHealth } from "./api/health.js";
import { askAssistant } from "./api/assistant.js";
import {
  downloadProtectedArtifact,
  viewProtectedArtifact,
} from "./api/artifacts.js";

const TASK_OPTIONS = [
  { value: "binary_classification", label: "Binary Classification" },
  { value: "multiclass_classification", label: "Multiclass Classification" },
  { value: "regression", label: "Regression" },
  { value: "clustering", label: "Clustering (no target required)" },
  {
    value: "anomaly_detection",
    label: "Anomaly Detection (no target required)",
  },
  { value: "time_series_forecasting", label: "Time-Series Forecasting" },
];

function taskLabel(value) {
  return (
    TASK_OPTIONS.find((option) => option.value === value)?.label ||
    "Needs confirmation"
  );
}

function displayLabel(value) {
  return (
    value
      ?.replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase()) || "—"
  );
}

function formatStatistic(value) {
  if (value === null || value === undefined) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 });
}

const PERCENT_METRICS = new Set([
  "accuracy",
  "precision",
  "recall",
  "f1",
  "roc_auc",
  "r2",
]);
function formatMetric(metric, value) {
  if (value === null || value === undefined) return "—";
  const numeric = Number(value);
  return PERCENT_METRICS.has(metric)
    ? `${(numeric * 100).toLocaleString(undefined, { maximumFractionDigits: 1 })}%`
    : numeric.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function historyStatusTone(status) {
  const normalized = String(status || "").toUpperCase();
  if (["EVALUATED", "COMPLETED"].includes(normalized)) return "complete";
  if (normalized === "FAILED") return "failed";
  if (
    normalized.includes("CONFIRM") ||
    ["CREATED", "ANALYZED", "PLANNED"].includes(normalized)
  )
    return "awaiting";
  return "running";
}

function trainingFailureMessage(message) {
  if (/train failed: FileNotFoundError\.?/i.test(message || "")) {
    return "Training couldn't start because the uploaded dataset is no longer available. Please upload the dataset again.";
  }
  return message || "Background job failed.";
}

function InlineMarkdown({ value }) {
  return value
    .split(/(\*\*[^*]+\*\*)/g)
    .map((part, index) =>
      part.startsWith("**") && part.endsWith("**") ? (
        <strong key={index}>{part.slice(2, -2)}</strong>
      ) : (
        part
      ),
    );
}

function MarkdownAnswer({ markdown }) {
  const lines = markdown.split("\n");
  const blocks = [];
  for (let index = 0; index < lines.length; ) {
    const line = lines[index];
    if (line.startsWith("|")) {
      const table = [];
      while (index < lines.length && lines[index].startsWith("|")) {
        table.push(lines[index]);
        index += 1;
      }
      const rows = table
        .filter((row) => !/^\|[-:| ]+\|$/.test(row))
        .map((row) =>
          row
            .split("|")
            .slice(1, -1)
            .map((cell) => cell.trim()),
        );
      blocks.push(
        <div className="table-scroll assistant-table" key={`table-${index}`}>
          <table>
            <thead>
              <tr>
                {rows[0].map((cell, cellIndex) => (
                  <th key={cellIndex}>
                    <InlineMarkdown value={cell} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(1).map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex}>
                      <InlineMarkdown value={cell} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
    } else if (line.startsWith("### ")) {
      blocks.push(<h3 key={index}>{line.slice(4)}</h3>);
      index += 1;
    } else if (line.startsWith("#### ")) {
      blocks.push(<h4 key={index}>{line.slice(5)}</h4>);
      index += 1;
    } else if (line.startsWith("*") && line.endsWith("*")) {
      blocks.push(
        <p className="assistant-note" key={index}>
          {line.slice(1, -1)}
        </p>,
      );
      index += 1;
    } else if (line.startsWith("- ")) {
      blocks.push(
        <ul key={index}>
          <li>
            <InlineMarkdown value={line.slice(2)} />
          </li>
        </ul>,
      );
      index += 1;
    } else if (line.trim()) {
      blocks.push(
        <p key={index}>
          <InlineMarkdown value={line} />
        </p>,
      );
      index += 1;
    } else {
      index += 1;
    }
  }
  return <div className="assistant-markdown">{blocks}</div>;
}

function VerifiedChart({
  specification,
  valueFormat = "number",
  metricLabel,
  groupLabel,
  isComparison = false,
}) {
  if (!specification?.labels?.length || !specification?.values?.length)
    return null;
  const data = specification.labels.map((label, index) => ({
    label,
    value: specification.values[index],
  }));
  const formatValue = (value) =>
    valueFormat === "percent" && typeof value === "number"
      ? `${(value * 100).toLocaleString(undefined, { maximumFractionDigits: 2 })}%`
      : formatStatistic(value);
  if (specification.chart_type === "heatmap")
    return (
      <div className="correlation-heatmap" aria-label={specification.title}>
        {data.map((item) => (
          <div
            key={item.label}
            className="heatmap-cell"
            style={{ "--strength": Math.abs(item.value) }}
          >
            <span>{item.label}</span>
            <strong>{formatValue(item.value)}</strong>
          </div>
        ))}
      </div>
    );
  if (specification.chart_type === "bar" && isComparison) {
    const height = Math.max(280, data.length * 38 + 55);
    return (
      <div
        className="chart-wrap chart-comparison"
        aria-label={specification.title}
      >
        <ResponsiveContainer width="100%" height={height}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ left: 12, right: 34, top: 12, bottom: 24 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              tickFormatter={formatValue}
              label={{
                value: metricLabel || "Verified value",
                position: "insideBottom",
                offset: -12,
              }}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={150}
              tick={{ fontSize: 12 }}
              label={{
                value: groupLabel || "Group",
                angle: -90,
                position: "insideLeft",
              }}
            />
            <Tooltip
              formatter={(value) => [
                formatValue(value),
                metricLabel || "Verified value",
              ]}
            />
            <Bar dataKey="value" fill="#1687d4" radius={[0, 5, 5, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }
  const Chart = specification.chart_type === "line" ? LineChart : BarChart;
  return (
    <div
      className={`chart-wrap chart-${specification.chart_type || "bar"}`}
      aria-label={specification.title}
    >
      <ResponsiveContainer width="100%" height={280}>
        <Chart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="label"
            interval={0}
            angle={data.length > 6 ? -25 : 0}
            textAnchor={data.length > 6 ? "end" : "middle"}
            height={data.length > 6 ? 70 : 30}
          />
          <YAxis tickFormatter={formatValue} />
          <Tooltip formatter={(value) => formatValue(value)} />
          {specification.chart_type === "line" ? (
            <Line
              type="monotone"
              dataKey="value"
              stroke="#44d7b6"
              strokeWidth={3}
            />
          ) : (
            <Bar dataKey="value" fill="#1687d4" radius={[5, 5, 0, 0]} />
          )}
        </Chart>
      </ResponsiveContainer>
    </div>
  );
}

function AnalyticsDashboard({
  result,
  onQuestion,
  showCalculationDetails = true,
}) {
  const verified = result?.structured_result;
  if (!verified) return <MarkdownAnswer markdown={result.answer} />;
  const rows = verified.rows || [];
  const metricResults = verified.metric_results || [];
  const isRate = verified.value_format === "percent";
  const formatVerifiedValue = (value) =>
    isRate && typeof value === "number"
      ? `${(value * 100).toLocaleString(undefined, { maximumFractionDigits: 2 })}%`
      : formatStatistic(value);
  const directValue =
    verified.value ??
    (rows[0]
      ? `${rows[0].label}: ${formatVerifiedValue(rows[0].value)}`
      : "Verified result ready");
  const insights = verified.insights?.length
    ? verified.insights
    : [
        `Analyzed ${formatStatistic(verified.rows_analyzed)} dataset rows.`,
        rows.length > 1
          ? `Returned ${rows.length} ranked or grouped results.`
          : "Result was calculated by the local analytics engine.",
        rows[0]
          ? `Leading result: ${rows[0].label} (${formatVerifiedValue(rows[0].value)}).`
          : "No unverified model calculation was used.",
      ];
  const plan = verified.plan?.steps || [];
  const isComparison = plan.some(
    (step) => step.operation === "group_aggregate",
  );
  const followUps = verified.follow_up_questions?.length
    ? verified.follow_up_questions
    : [
        "Show the dataset distribution.",
        "Which features are associated?",
        "Compare the top groups.",
      ];
  return (
    <div className="analytics-dashboard">
      <div className="verified-badge">✓ Verified from dataset</div>
      <div className="analytics-direct">
        <p>Direct answer</p>
        <h3>{verified.title}</h3>
        <strong>
          {typeof directValue === "number"
            ? formatVerifiedValue(directValue)
            : directValue}
        </strong>
      </div>
      <div className="analytics-metrics">
        <div>
          <span>Rows analyzed</span>
          <strong>{formatStatistic(verified.rows_analyzed)}</strong>
        </div>
        <div>
          <span>Results returned</span>
          <strong>{rows.length || 1}</strong>
        </div>
        <div>
          <span>Source</span>
          <strong>{verified.source_filename}</strong>
        </div>
      </div>
      <div className="analytics-insights">
        <h4>Insights</h4>
        <ul>
          {insights.map((insight) => (
            <li key={insight}>{insight}</li>
          ))}
        </ul>
      </div>
      {verified.total_groups > rows.length && (
        <p className="limit-note">
          Showing the first {rows.length} of {verified.total_groups} verified
          groups.
        </p>
      )}
      {rows.length > 0 && (
        <div className="table-scroll assistant-table">
          <table>
            <thead>
              <tr>
                <th>{verified.group_label || "Result"}</th>
                <th>{verified.metric_label || "Verified value"}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label}>
                  <td>{row.label}</td>
                  <td>{formatVerifiedValue(row.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {metricResults.length > 1 && (
        <div className="assistant-metric-results">
          {metricResults.map((metric) => (
            <section className="table-scroll assistant-table" key={metric.name}>
              <h4>{displayLabel(metric.name)}</h4>
              {metric.rows?.length ? (
                <table>
                  <thead>
                    <tr>
                      <th>{verified.group_label || "Result"}</th>
                      <th>{displayLabel(metric.name)}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metric.rows.map((row) => (
                      <tr key={row.label}>
                        <td>{row.label}</td>
                        <td>{formatVerifiedValue(row.value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p>{formatVerifiedValue(metric.value)}</p>
              )}
            </section>
          ))}
        </div>
      )}
      <VerifiedChart
        specification={verified.chart}
        valueFormat={verified.value_format}
        metricLabel={verified.metric_label}
        groupLabel={verified.group_label}
        isComparison={isComparison}
      />
      {showCalculationDetails && (
        <details className="calculation-details">
          <summary>How calculated</summary>
          <p>{verified.calculation}</p>
          <p>
            Plan:{" "}
            {plan
              .map((step) => step.operation.replaceAll("_", " "))
              .join(" → ") || "trusted local calculation"}
            .
          </p>
        </details>
      )}
      <div className="assistant-followups">
        <h4>Try next</h4>
        {followUps.map((question) => (
          <button
            type="button"
            key={question}
            onClick={() => onQuestion(question)}
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}

function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const googleButton = useRef(null);
  useEffect(() => {
    const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
    if (!clientId || !googleButton.current) return undefined;
    const render = () => {
      window.google?.accounts.id.initialize({
        client_id: clientId,
        callback: async ({ credential }) => {
          setBusy(true);
          setError("");
          try {
            const result = await loginWithGoogleCredential(credential);
            setAccessToken(result.access_token);
            onAuthenticated(result.user);
          } catch (requestError) {
            setError(requestError.message || "Google sign-in failed.");
          } finally {
            setBusy(false);
          }
        },
      });
      window.google?.accounts.id.renderButton(googleButton.current, {
        theme: "outline",
        size: "large",
        width: 320,
        text: "continue_with",
      });
    };
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.onload = render;
    document.head.appendChild(script);
    return () => script.remove();
  }, [onAuthenticated]);
  async function submit(event) {
    event.preventDefault();
    setError("");
    if (mode === "register" && password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (password.length < 12) {
      setError("Use a password with at least 12 characters.");
      return;
    }
    setBusy(true);
    try {
      const result =
        mode === "register"
          ? await register(email, password)
          : await login(email, password);
      setAccessToken(result.access_token);
      setPassword("");
      setConfirmPassword("");
      onAuthenticated(result.user);
    } catch (requestError) {
      setError(requestError.message || "Authentication failed.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="auth-shell">
      <div className="auth-ambient" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <section className="auth-intro" aria-labelledby="auth-project-title">
        <div className="auth-brand">
          <span className="auth-brand-mark">A</span>
          <strong>AutoDS-Agent</strong>
          <small>Intelligent Data Science</small>
        </div>
        <p className="eyebrow">Trusted automation · Explainable decisions</p>
        <h1 id="auth-project-title">
          Turn raw data into <span>decisions you can trust.</span>
        </h1>
        <p className="auth-intro-copy">
          A controlled multi-agent workspace that profiles datasets, plans safe
          pipelines, evaluates models, and explains every verified result.
        </p>
        <div className="auth-flow" aria-label="AutoDS workflow">
          <span>
            <i>01</i>Profile
          </span>
          <b aria-hidden="true">→</b>
          <span>
            <i>02</i>Plan
          </span>
          <b aria-hidden="true">→</b>
          <span>
            <i>03</i>Train
          </span>
          <b aria-hidden="true">→</b>
          <span>
            <i>04</i>Explain
          </span>
        </div>
        <div className="auth-trust-card">
          <span className="auth-live-dot" aria-hidden="true" />
          <div>
            <strong>AI-guided. Locally verified.</strong>
            <small>
              Models propose and interpret; trusted code calculates, trains, and
              evaluates.
            </small>
          </div>
        </div>
      </section>
      <section className="auth-card" aria-labelledby="auth-title">
        <p className="section-kicker">Secure workspace</p>
        <h2 id="auth-title">
          {mode === "login"
            ? "Enter your data-science workspace"
            : "Create your workspace"}
        </h2>
        <p className="auth-card-copy">
          {mode === "login"
            ? "Continue building verified experiments and analytics."
            : "Start creating trusted, explainable ML workflows."}
        </p>
        <form className="objective-form auth-form" onSubmit={submit}>
          <label>
            Email
            <input
              type="email"
              autoComplete="email"
              value={email}
              required
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
            />
          </label>
          <label>
            Password
            <input
              type="password"
              autoComplete={
                mode === "login" ? "current-password" : "new-password"
              }
              value={password}
              minLength="12"
              required
              onChange={(event) => setPassword(event.target.value)}
              placeholder="At least 12 characters"
            />
          </label>
          {mode === "register" && (
            <label>
              Confirm password
              <input
                type="password"
                autoComplete="new-password"
                value={confirmPassword}
                minLength="12"
                required
                onChange={(event) => setConfirmPassword(event.target.value)}
                placeholder="Repeat your password"
              />
            </label>
          )}
          <button className="auth-submit" disabled={busy}>
            {busy
              ? "Please wait…"
              : mode === "login"
                ? "Open workspace"
                : "Create account"}
            <span aria-hidden="true">→</span>
          </button>
          <div
            className="auth-google"
            ref={googleButton}
            aria-label="Continue with Google"
          />
          {!import.meta.env.VITE_GOOGLE_CLIENT_ID && (
            <p className="limit-note">
              Google sign-in is not configured for this environment.
            </p>
          )}
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
        </form>
        <div className="auth-switch">
          <span>
            {mode === "login"
              ? "New to AutoDS-Agent?"
              : "Already have a workspace?"}
          </span>
          <button
            type="button"
            onClick={() => {
              setMode(mode === "login" ? "register" : "login");
              setError("");
            }}
          >
            {mode === "login" ? "Create an account" : "Log in"}
          </button>
        </div>
        <p className="auth-security">
          Validated plans · Protected datasets · Verified metrics
        </p>
      </section>
    </main>
  );
}

function App() {
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [profile, setProfile] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [datasetsError, setDatasetsError] = useState("");
  const [datasetOpenError, setDatasetOpenError] = useState("");
  const [datasetAction, setDatasetAction] = useState("");
  const [datasetActionKind, setDatasetActionKind] = useState("");
  const [datasetAnalysis, setDatasetAnalysis] = useState(null);
  const [autoAnalysis, setAutoAnalysis] = useState(null);
  const [assistantContext, setAssistantContext] = useState("dataset");
  const [assistantExperimentId, setAssistantExperimentId] = useState("");
  const [assistantConversationId] = useState(() => {
    const key = "autods-assistant-conversation-id";
    const existing = window.localStorage.getItem(key);
    if (existing) return existing;
    const created =
      window.crypto?.randomUUID?.() || `conversation-${Date.now()}`;
    window.localStorage.setItem(key, created);
    return created;
  });
  const [objective, setObjective] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [selectedTask, setSelectedTask] = useState("");
  const [confirmedExperiment, setConfirmedExperiment] = useState(null);
  const [clustering, setClustering] = useState(null);
  const [specializedResult, setSpecializedResult] = useState(null);
  const [experimentError, setExperimentError] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isConfirming, setIsConfirming] = useState(false);
  const [pipelinePlan, setPipelinePlan] = useState(null);
  const [isPlanning, setIsPlanning] = useState(false);
  const [planningError, setPlanningError] = useState("");
  const [isUpdatingAdaptive, setIsUpdatingAdaptive] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingError, setTrainingError] = useState("");
  const [evaluation, setEvaluation] = useState(null);
  const [optimization, setOptimization] = useState(null);
  const [explainability, setExplainability] = useState(null);
  const [reflection, setReflection] = useState(null);
  const [predictionFile, setPredictionFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [report, setReport] = useState(null);
  const [pdfReport, setPdfReport] = useState(null);
  const [reportHistory, setReportHistory] = useState([]);
  const [reportHistoryError, setReportHistoryError] = useState("");
  const [reportSearch, setReportSearch] = useState("");
  const [reportFormatFilter, setReportFormatFilter] = useState("all");
  const [reportTaskFilter, setReportTaskFilter] = useState("all");
  const [reportDatasetFilter, setReportDatasetFilter] = useState("all");
  const [reportSort, setReportSort] = useState("newest");
  const [reportVisibleCount, setReportVisibleCount] = useState(8);
  const [selectedReportExperimentId, setSelectedReportExperimentId] =
    useState("");
  const [workflowError, setWorkflowError] = useState("");
  const [workflowAction, setWorkflowAction] = useState("");
  const [activeJob, setActiveJob] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyDetails, setHistoryDetails] = useState({});
  const [historyError, setHistoryError] = useState("");
  const [historySearch, setHistorySearch] = useState("");
  const [historyTaskFilter, setHistoryTaskFilter] = useState("all");
  const [historyStatusFilter, setHistoryStatusFilter] = useState("all");
  const [historySort, setHistorySort] = useState("newest");
  const [datasetSearch, setDatasetSearch] = useState("");
  const [datasetSort, setDatasetSort] = useState("newest");
  const [openedExperiment, setOpenedExperiment] = useState(null);
  const [openingExperimentId, setOpeningExperimentId] = useState("");
  const [assistantQuestion, setAssistantQuestion] = useState("");
  const [assistantDatasetSearch, setAssistantDatasetSearch] = useState("");
  const [assistantResult, setAssistantResult] = useState(null);
  const [assistantBusy, setAssistantBusy] = useState(false);
  const [assistantError, setAssistantError] = useState("");
  const [downloadAction, setDownloadAction] = useState("");
  const [downloadError, setDownloadError] = useState("");
  const [activeStage, setActiveStage] = useState("dashboard");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [compactLayout, setCompactLayout] = useState(false);
  const [showCalculationDetails, setShowCalculationDetails] = useState(true);
  const [lightTheme, setLightTheme] = useState(
    () => window.localStorage.getItem("autods-theme") === "light",
  );
  const [reduceMotion, setReduceMotion] = useState(
    () => window.localStorage.getItem("autods-reduce-motion") === "true",
  );

  useEffect(() => {
    window.localStorage.setItem("autods-theme", lightTheme ? "light" : "dark");
  }, [lightTheme]);

  useEffect(() => {
    window.localStorage.setItem("autods-reduce-motion", String(reduceMotion));
  }, [reduceMotion]);

  useEffect(() => {
    if (!getAccessToken()) {
      setAuthLoading(false);
      return;
    }
    getCurrentUser()
      .then(setUser)
      .catch(() => clearAccessToken())
      .finally(() => setAuthLoading(false));
    const expired = () => setUser(null);
    window.addEventListener("autods-auth-expired", expired);
    return () => window.removeEventListener("autods-auth-expired", expired);
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    getBackendHealth(controller.signal)
      .then(setHealth)
      .catch((requestError) => {
        if (requestError.name !== "AbortError") {
          setError("Backend health is currently unavailable.");
        }
      });

    return () => controller.abort();
  }, []);

  async function refreshHistory() {
    try {
      const value = await getExperimentHistory();
      setHistory(value.items);
      const details = await Promise.all(
        value.items.map(async (item) => {
          try {
            return [
              item.experiment_id,
              await getExperimentDetail(item.experiment_id),
            ];
          } catch {
            return [item.experiment_id, null];
          }
        }),
      );
      setHistoryDetails(
        Object.fromEntries(details.filter(([, detail]) => detail)),
      );
      setHistoryError("");
    } catch {
      setHistory([]);
      setHistoryDetails({});
      setHistoryError("Experiment history is currently unavailable.");
    }
  }

  async function refreshDatasets() {
    try {
      const value = await listDatasets();
      setDatasets(value.items || []);
      setDatasetsError("");
    } catch (requestError) {
      setDatasets([]);
      setDatasetsError(
        requestError.message ||
          "Your saved datasets are currently unavailable.",
      );
    }
  }

  async function refreshReportHistory() {
    try {
      const value = await getReportHistory();
      setReportHistory(value.items || []);
      const experimentIds = [
        ...new Set((value.items || []).map((item) => item.experiment_id)),
      ];
      const details = await Promise.all(
        experimentIds.map(async (experimentId) => {
          try {
            return [experimentId, await getExperimentDetail(experimentId)];
          } catch {
            return [experimentId, null];
          }
        }),
      );
      setHistoryDetails((current) => ({
        ...current,
        ...Object.fromEntries(details.filter(([, detail]) => detail)),
      }));
      setReportHistoryError("");
    } catch (requestError) {
      setReportHistory([]);
      setSelectedReportExperimentId("");
      setReportHistoryError(
        requestError.message || "Saved reports are currently unavailable.",
      );
    }
  }

  useEffect(() => {
    if (!user) {
      setHistory([]);
      setHistoryDetails({});
      setHistoryError("");
      setDatasets([]);
      setDatasetsError("");
      setReportHistory([]);
      setReportHistoryError("");
      return;
    }
    refreshHistory();
    refreshDatasets();
    refreshReportHistory();
  }, [user]);

  function resetDatasetWorkflow() {
    setDatasetAnalysis(null);
    setAutoAnalysis(null);
    setObjective("");
    setAnalysis(null);
    setSelectedTarget("");
    setSelectedTask("");
    setConfirmedExperiment(null);
    setExperimentError("");
    setPipelinePlan(null);
    setPlanningError("");
    setTrainingResult(null);
    setTrainingError("");
    setEvaluation(null);
    setOptimization(null);
    setExplainability(null);
    setPrediction(null);
    setReport(null);
    setPdfReport(null);
    setWorkflowError("");
    setAssistantResult(null);
    setAssistantError("");
    setOpenedExperiment(null);
  }

  async function openDataset(datasetId, { preserveExperiment = false } = {}) {
    if (!datasetId || datasetAction) return false;
    setDatasetAction(datasetId);
    setDatasetActionKind("open");
    setDatasetOpenError("");
    setUploadError("");
    try {
      const profiled = await getDatasetProfile(datasetId);
      if (!preserveExperiment) resetDatasetWorkflow();
      else {
        setAssistantResult(null);
        setAssistantError("");
      }
      setProfile(profiled);
      setAssistantContext("dataset");
      analyzeDatasetProfile(datasetId)
        .then(setDatasetAnalysis)
        .catch(() => setDatasetAnalysis(null));
      getDatasetAutoAnalysis(datasetId)
        .then(setAutoAnalysis)
        .catch(() => setAutoAnalysis(null));
      return true;
    } catch (requestError) {
      const message =
        requestError.message || "Could not load this saved dataset.";
      setDatasetOpenError(message);
      setUploadError(message);
      return false;
    } finally {
      setDatasetAction("");
      setDatasetActionKind("");
    }
  }

  async function openSavedDataset(datasetId, destination = "analytics") {
    if (!(await openDataset(datasetId))) return;
    openStage("workspace");
    const targetId =
      destination === "experiment"
        ? "experiment-objective-panel"
        : "dataset-analytics-panel";
    window.requestAnimationFrame(() =>
      window.requestAnimationFrame(() => {
        document
          .getElementById(targetId)
          ?.scrollIntoView({ behavior: "smooth", block: "start" });
      }),
    );
  }

  async function removeDataset(dataset) {
    const consequence =
      " This also permanently deletes any experiments using it, along with their saved models, predictions, and reports.";
    if (
      !window.confirm(
        `Delete ${dataset.original_filename}?${consequence} This cannot be undone.`,
      )
    )
      return;
    setDatasetAction(dataset.id);
    setDatasetActionKind("delete");
    setDatasetsError("");
    try {
      await deleteDataset(dataset.id, true);
      setDatasetOpenError("");
      if (profile?.dataset.id === dataset.id) {
        setProfile(null);
        resetDatasetWorkflow();
      }
      if (openedExperiment?.dataset_id === dataset.id)
        setOpenedExperiment(null);
      await Promise.all([
        refreshDatasets(),
        refreshHistory(),
        refreshReportHistory(),
      ]);
    } catch (requestError) {
      setDatasetsError(
        requestError.message || "Could not delete this dataset.",
      );
    } finally {
      setDatasetAction("");
      setDatasetActionKind("");
    }
  }

  async function submitAssistant(event) {
    event.preventDefault();
    if (!assistantQuestion.trim()) return;
    const datasetId =
      assistantContext === "dataset" ? profile?.dataset.id : undefined;
    const experimentId =
      assistantContext === "experiment"
        ? assistantExperimentId ||
          openedExperiment?.experiment_id ||
          confirmedExperiment?.experiment_id
        : undefined;
    setAssistantBusy(true);
    setAssistantError("");
    try {
      setAssistantResult(
        await askAssistant(assistantQuestion, {
          contextType: assistantContext,
          datasetId,
          experimentId,
          conversationId: assistantConversationId,
        }),
      );
    } catch (requestError) {
      setAssistantError(requestError.message || "Assistant request failed.");
    } finally {
      setAssistantBusy(false);
    }
  }

  useJobPolling(
    activeJob?.job_id,
    async (job) => {
      setActiveJob(job);
      if (job.status === "COMPLETED" && job.job_type === "train") {
        try {
          setTrainingResult(await trainExperiment(job.experiment_id));
        } catch (error) {
          setTrainingError(error.message);
        }
        refreshHistory();
      }
      if (job.status === "FAILED")
        setTrainingError(trainingFailureMessage(job.error_information));
    },
    (error) =>
      setTrainingError(error.message || "Could not check background job."),
  );

  async function handleUpload(event) {
    event.preventDefault();
    if (!selectedFile) {
      setUploadError("Choose a CSV or XLSX file before uploading.");
      return;
    }

    setIsUploading(true);
    setUploadError("");
    setProfile(null);
    resetDatasetWorkflow();

    try {
      const dataset = await uploadDataset(selectedFile);
      await refreshDatasets();
      await openDataset(dataset.id);
    } catch (requestError) {
      setUploadError(requestError.message || "Dataset upload failed.");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleObjective(event) {
    event.preventDefault();
    if (!objective.trim()) {
      setExperimentError("Describe what you want to predict.");
      return;
    }

    setIsAnalyzing(true);
    setExperimentError("");
    setConfirmedExperiment(null);
    setPipelinePlan(null);
    setPlanningError("");
    setTrainingResult(null);
    setTrainingError("");
    setEvaluation(null);
    setOptimization(null);
    setExplainability(null);
    setPrediction(null);
    setReport(null);
    setWorkflowError("");
    try {
      const result = await createExperiment(
        profile.dataset.id,
        objective.trim(),
      );
      setAnalysis(result);
      setSelectedTarget(result.suggested_target_column || "");
      setSelectedTask(result.suggested_task_type || "");
    } catch (requestError) {
      setExperimentError(requestError.message || "Objective analysis failed.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  function handleTargetChange(event) {
    const target = event.target.value;
    setSelectedTarget(target);
    setConfirmedExperiment(null);
    setPipelinePlan(null);
    setPlanningError("");
    setTrainingResult(null);
    setTrainingError("");
    setEvaluation(null);
    setOptimization(null);
    setExplainability(null);
    setPrediction(null);
    setReport(null);
    setWorkflowError("");
    const candidate = analysis?.target_candidates.find(
      (item) => item.name === target,
    );
    setSelectedTask(candidate?.suggested_task_type || "");
  }

  async function handleConfirmation(event) {
    event.preventDefault();
    if (
      !selectedTask ||
      (!["clustering", "anomaly_detection"].includes(selectedTask) &&
        !selectedTarget)
    ) {
      setExperimentError("Choose both a target column and task type.");
      return;
    }

    setIsConfirming(true);
    setExperimentError("");
    try {
      if (selectedTask === "clustering") {
        setClustering(await clusterDataset(profile.dataset.id));
        return;
      }
      if (selectedTask === "anomaly_detection") {
        setSpecializedResult(await detectAnomalies(profile.dataset.id));
        return;
      }
      if (selectedTask === "time_series_forecasting") {
        const timeColumn = autoAnalysis?.feature_roles.find(
          (item) => item.role === "datetime",
        )?.column;
        if (!timeColumn)
          throw new Error(
            "Time-series forecasting requires a detected datetime column.",
          );
        setSpecializedResult(
          await forecastDataset(profile.dataset.id, timeColumn, selectedTarget),
        );
        return;
      }
      const result = await confirmExperiment(
        analysis.experiment_id,
        selectedTarget,
        selectedTask,
      );
      setConfirmedExperiment(result);
      setAssistantExperimentId(result.experiment_id);
      setPipelinePlan(null);
      setPlanningError("");
      setTrainingResult(null);
      setTrainingError("");
    } catch (requestError) {
      setExperimentError(
        requestError.message || "Objective confirmation failed.",
      );
    } finally {
      setIsConfirming(false);
    }
  }

  async function handlePlanning() {
    setIsPlanning(true);
    setPlanningError("");
    try {
      setPipelinePlan(
        await generatePipelinePlan(confirmedExperiment.experiment_id),
      );
      setTrainingResult(null);
      setTrainingError("");
    } catch (requestError) {
      setPlanningError(requestError.message || "Pipeline planning failed.");
    } finally {
      setIsPlanning(false);
    }
  }

  async function handleAdaptiveWeightOverride(modelName, enabled) {
    if (!confirmedExperiment?.experiment_id || !pipelinePlan) return;
    const current = pipelinePlan.plan.balanced_class_weight_models || [];
    const next = enabled
      ? [...new Set([...current, modelName])]
      : current.filter((name) => name !== modelName);
    setIsUpdatingAdaptive(true);
    setPlanningError("");
    try {
      setPipelinePlan(
        await updateAdaptivePipeline(confirmedExperiment.experiment_id, next),
      );
    } catch (requestError) {
      setPlanningError(
        requestError.message ||
          "Could not save the adaptive pipeline override.",
      );
    } finally {
      setIsUpdatingAdaptive(false);
    }
  }

  async function handleTraining() {
    setIsTraining(true);
    setTrainingError("");
    try {
      setActiveJob(await startExperimentJob(confirmedExperiment.experiment_id));
      setEvaluation(null);
      setOptimization(null);
      setExplainability(null);
      setPrediction(null);
      setReport(null);
    } catch (requestError) {
      setTrainingError(requestError.message || "Baseline training failed.");
    } finally {
      setIsTraining(false);
    }
  }

  async function handleReflection() {
    if (!confirmedExperiment?.experiment_id) return;
    setWorkflowAction("reflection");
    setWorkflowError("");
    try {
      setReflection(await createReflection(confirmedExperiment.experiment_id));
    } catch (requestError) {
      setWorkflowError(
        requestError.message || "AI review is currently unavailable.",
      );
    } finally {
      setWorkflowAction("");
    }
  }

  async function handleOpenExperiment(experimentId) {
    setOpeningExperimentId(experimentId);
    setHistoryError("");
    try {
      setOpenedExperiment(await getExperimentDetail(experimentId));
      setAssistantExperimentId(experimentId);
      setActiveStage("history-detail");
      setSidebarOpen(false);
    } catch (requestError) {
      setHistoryError(
        requestError.message || "Could not reopen this experiment.",
      );
    } finally {
      setOpeningExperimentId("");
    }
  }

  async function runWorkflow(action, request) {
    setWorkflowAction(action);
    setWorkflowError("");
    try {
      return await request();
    } catch (requestError) {
      setWorkflowError(requestError.message || `${action} failed.`);
      return null;
    } finally {
      setWorkflowAction("");
    }
  }

  async function handleEvaluation() {
    const result = await runWorkflow("evaluate", () =>
      evaluateExperiment(confirmedExperiment.experiment_id),
    );
    if (result) {
      setEvaluation(result);
      setOptimization(null);
      setExplainability(null);
      setPrediction(null);
      setReport(null);
    }
  }

  async function handleOptimization() {
    const result = await runWorkflow("optimize", () =>
      optimizeExperiment(confirmedExperiment.experiment_id),
    );
    if (result) {
      setOptimization(result);
      const refreshed = await runWorkflow("refresh evaluation", () =>
        evaluateExperiment(confirmedExperiment.experiment_id),
      );
      if (refreshed) setEvaluation(refreshed);
    }
  }

  async function handleExplainability() {
    const result = await runWorkflow("explainability", () =>
      getExplainability(confirmedExperiment.experiment_id),
    );
    if (result) setExplainability(result);
  }

  async function handlePredictions(event) {
    event.preventDefault();
    if (!predictionFile) {
      setWorkflowError("Choose a compatible prediction CSV first.");
      return;
    }
    const result = await runWorkflow("predictions", () =>
      createPredictions(confirmedExperiment.experiment_id, predictionFile),
    );
    if (result) setPrediction(result);
  }

  async function handleReport() {
    const result = await runWorkflow("report", () =>
      createReport(confirmedExperiment.experiment_id),
    );
    if (result) {
      setReport(result);
      refreshReportHistory();
    }
  }
  async function handlePdfReport() {
    const result = await runWorkflow("PDF report", () =>
      createPdfReport(confirmedExperiment.experiment_id),
    );
    if (result) {
      setPdfReport(result);
      refreshReportHistory();
    }
  }

  async function handleArtifactDownload(kind, artifact, filename) {
    if (!artifact || downloadAction) return;
    setDownloadAction(kind);
    setDownloadError("");
    try {
      await downloadProtectedArtifact(artifact.download_url, filename);
    } catch (requestError) {
      setDownloadError(
        requestError.message || "Download failed. Please try again.",
      );
    } finally {
      setDownloadAction("");
    }
  }

  async function handleArtifactView(kind, artifact) {
    if (!artifact || downloadAction) return;
    setDownloadAction(kind);
    setDownloadError("");
    try {
      await viewProtectedArtifact(artifact.download_url);
    } catch (requestError) {
      setDownloadError(requestError.message || "Could not open this report.");
    } finally {
      setDownloadAction("");
    }
  }

  const numericalProfiles =
    profile?.columns.filter((column) => column.numeric_statistics) || [];
  const modelIsTraining =
    isTraining ||
    (activeJob?.job_type === "train" &&
      ["PENDING", "STARTED", "RUNNING"].includes(activeJob.status));
  const workflowSteps = [
    "Dataset",
    "Objective",
    "Plan",
    "Train",
    "Evaluate",
    "Results",
  ];
  const completedStep = evaluation
    ? 5
    : trainingResult
      ? 3
      : pipelinePlan
        ? 2
        : confirmedExperiment
          ? 1
          : profile
            ? 0
            : -1;
  const isRegression =
    evaluation &&
    Object.prototype.hasOwnProperty.call(
      evaluation.final_test_metrics || {},
      "rmse",
    );
  const visibleMetricNames = isRegression
    ? ["mae", "rmse", "r2"]
    : ["accuracy", "precision", "recall", "f1", "roc_auc"];
  const selectedValidation = evaluation?.validation_comparison?.find(
    (item) => item.model_run_id === evaluation.selected_model_run_id,
  );
  const validationPrimaryMetric =
    selectedValidation?.metrics?.[evaluation?.primary_metric];
  const testPrimaryMetric =
    evaluation?.final_test_metrics?.[evaluation?.primary_metric];
  const generalizationDifference =
    validationPrimaryMetric !== undefined && testPrimaryMetric !== undefined
      ? testPrimaryMetric - validationPrimaryMetric
      : null;
  const generalizationPercentage =
    generalizationDifference !== null && validationPrimaryMetric
      ? (generalizationDifference / Math.abs(validationPrimaryMetric)) * 100
      : null;
  const comparisonChartData =
    evaluation?.validation_comparison?.map((item) => ({
      model: displayLabel(item.model_name),
      score: item.metrics[evaluation.primary_metric],
      selected: item.model_run_id === evaluation.selected_model_run_id,
    })) || [];
  const performanceCostData =
    evaluation?.validation_comparison
      ?.filter(
        (item) =>
          item.metrics?.[evaluation.primary_metric] !== null &&
          item.metrics?.[evaluation.primary_metric] !== undefined &&
          item.training_seconds > 0,
      )
      .map((item) => ({
        model: displayLabel(item.model_name),
        training_seconds: item.training_seconds,
        validation_score: item.metrics[evaluation.primary_metric],
        selected: item.model_run_id === evaluation.selected_model_run_id,
      })) || [];
  const optimizationBaseline =
    selectedValidation?.metrics?.[evaluation?.primary_metric];
  const primaryMetricIsHigherBetter = ![
    "mae",
    "rmse",
    "mse",
    "log_loss",
  ].includes(evaluation?.primary_metric);
  const optimizationImprovement =
    optimization &&
    optimizationBaseline !== undefined &&
    optimization.best_validation_score !== undefined
      ? primaryMetricIsHigherBetter
        ? optimization.best_validation_score - optimizationBaseline
        : optimizationBaseline - optimization.best_validation_score
      : null;
  // Preserve older records for linked experiments, but show one current entry per uploaded filename.
  const visibleDatasets = datasets.filter(
    (dataset, index, all) =>
      all.findIndex(
        (candidate) =>
          candidate.original_filename.toLowerCase() ===
          dataset.original_filename.toLowerCase(),
      ) === index,
  );
  const filteredAssistantDatasets = visibleDatasets.filter((dataset) =>
    dataset.original_filename
      .toLowerCase()
      .includes(assistantDatasetSearch.trim().toLowerCase()),
  );
  const assistantExperimentOptions = [
    ...history,
    openedExperiment,
    confirmedExperiment,
  ]
    .filter((item) => item?.experiment_id)
    .filter(
      (item, index, all) =>
        all.findIndex(
          (candidate) => candidate.experiment_id === item.experiment_id,
        ) === index,
    );
  const selectedAssistantExperimentId =
    assistantExperimentId ||
    openedExperiment?.experiment_id ||
    confirmedExperiment?.experiment_id ||
    "";
  const selectedAssistantExperiment = assistantExperimentOptions.find(
    (item) => item.experiment_id === selectedAssistantExperimentId,
  );
  const selectedDatasetTarget =
    autoAnalysis?.target_recommendation?.recommended_target;
  const selectedDatasetTask =
    autoAnalysis?.target_recommendation?.recommended_task;
  const selectedDatasetNumeric = profile?.summary?.numerical_columns?.[0];
  const selectedDatasetCategory = profile?.summary?.categorical_columns?.[0];
  const datasetQuestionSuggestions = profile
    ? [
        `Show summary statistics for ${selectedDatasetNumeric || "the numerical columns"}.`,
        selectedDatasetCategory
          ? `Compare ${selectedDatasetNumeric || "values"} by ${selectedDatasetCategory}.`
          : `Rank ${selectedDatasetNumeric || "the numerical columns"}.`,
        selectedDatasetNumeric
          ? `Show the distribution of ${selectedDatasetNumeric}.`
          : "Show the dataset distribution.",
        selectedDatasetTarget
          ? `Which features are associated with ${selectedDatasetTarget}?`
          : `Which features are associated with ${selectedDatasetNumeric || "the main numerical column"}?`,
        selectedDatasetTarget
          ? `Which ${selectedDatasetCategory || "group"} has the highest average ${selectedDatasetTarget}?`
          : `What are the top values in ${selectedDatasetCategory || "this dataset"}?`,
      ]
    : [];
  const completedExperimentCount = history.filter((item) =>
    ["EVALUATED", "COMPLETED"].includes(item.status),
  ).length;
  const inProgressExperimentCount = history.filter(
    (item) => !["EVALUATED", "COMPLETED", "FAILED"].includes(item.status),
  ).length;
  const historyTaskOptions = [
    ...new Set(history.map((item) => item.task_type).filter(Boolean)),
  ];
  const historyStatusOptions = [
    ...new Set(history.map((item) => item.status).filter(Boolean)),
  ];
  const historyRows = history
    .filter((item) => {
      const datasetName =
        visibleDatasets.find((dataset) => dataset.id === item.dataset_id)
          ?.original_filename || "";
      const query = historySearch.trim().toLowerCase();
      return (
        (!query ||
          [item.objective, datasetName, item.task_type, item.target_column]
            .filter(Boolean)
            .join(" ")
            .toLowerCase()
            .includes(query)) &&
        (historyTaskFilter === "all" || item.task_type === historyTaskFilter) &&
        (historyStatusFilter === "all" || item.status === historyStatusFilter)
      );
    })
    .sort((left, right) =>
      historySort === "newest"
        ? new Date(right.created_at) - new Date(left.created_at)
        : new Date(left.created_at) - new Date(right.created_at),
    );
  const datasetRows = visibleDatasets
    .filter((dataset) =>
      dataset.original_filename
        .toLowerCase()
        .includes(datasetSearch.trim().toLowerCase()),
    )
    .sort((left, right) =>
      datasetSort === "name"
        ? left.original_filename.localeCompare(right.original_filename)
        : datasetSort === "oldest"
          ? new Date(left.created_at) - new Date(right.created_at)
          : new Date(right.created_at) - new Date(left.created_at),
    );
  const reportGroups = Object.values(
    reportHistory.reduce((groups, item) => {
      if (!["html", "pdf"].includes(item.format)) return groups;
      const group = groups[item.experiment_id] || {
        experimentId: item.experiment_id,
        objective: item.objective,
        items: [],
      };
      group.items.push(item);
      groups[item.experiment_id] = group;
      return groups;
    }, {}),
  ).map((group) => ({
    ...group,
    items: group.items.sort(
      (left, right) => new Date(right.created_at) - new Date(left.created_at),
    ),
    latestByFormat: group.items.reduce(
      (formats, item) => ({
        ...formats,
        [item.format]: formats[item.format] || item,
      }),
      {},
    ),
  }));
  const reportTaskOptions = [
    ...new Set(
      reportGroups
        .map(
          (group) =>
            historyDetails[group.experimentId]?.task_type ||
            history.find((item) => item.experiment_id === group.experimentId)
              ?.task_type,
        )
        .filter(Boolean),
    ),
  ];
  const reportDatasetOptions = [
    ...new Set(
      reportGroups
        .map(
          (group) =>
            historyDetails[group.experimentId]?.dataset_id ||
            history.find((item) => item.experiment_id === group.experimentId)
              ?.dataset_id,
        )
        .filter(Boolean),
    ),
  ];
  const filteredReportGroups = reportGroups
    .filter((group) => {
      const detail = historyDetails[group.experimentId];
      const historyItem = history.find(
        (item) => item.experiment_id === group.experimentId,
      );
      const datasetId = detail?.dataset_id || historyItem?.dataset_id;
      const datasetName =
        visibleDatasets.find((dataset) => dataset.id === datasetId)
          ?.original_filename || "Saved dataset";
      const task = detail?.task_type || historyItem?.task_type;
      const query = reportSearch.trim().toLowerCase();
      return (
        (!query ||
          [datasetName, group.objective, task]
            .filter(Boolean)
            .join(" ")
            .toLowerCase()
            .includes(query)) &&
        (reportFormatFilter === "all" ||
          group.latestByFormat[reportFormatFilter]) &&
        (reportTaskFilter === "all" || task === reportTaskFilter) &&
        (reportDatasetFilter === "all" || datasetId === reportDatasetFilter)
      );
    })
    .sort((left, right) =>
      reportSort === "newest"
        ? new Date(right.items[0].created_at) -
          new Date(left.items[0].created_at)
        : new Date(left.items[0].created_at) -
          new Date(right.items[0].created_at),
    );
  const selectedReportGroup =
    filteredReportGroups.find(
      (group) => group.experimentId === selectedReportExperimentId,
    ) ||
    reportGroups.find(
      (group) => group.experimentId === selectedReportExperimentId,
    );
  const taskActivity = Object.entries(
    history.reduce((counts, item) => {
      const task = item.task_type || "not_confirmed";
      counts[task] = (counts[task] || 0) + 1;
      return counts;
    }, {}),
  ).map(([task, count]) => ({ task: displayLabel(task), count }));

  function openStage(stage) {
    setActiveStage(stage);
    setSidebarOpen(false);
  }

  function toggleSidebar() {
    if (window.matchMedia("(max-width: 720px)").matches)
      setSidebarOpen((open) => !open);
    else setSidebarCollapsed((collapsed) => !collapsed);
  }

  function SavedDatasetsPanel() {
    return (
      <section
        className="saved-datasets history-datasets"
        aria-labelledby="saved-datasets-title"
      >
        <div className="section-heading">
          <div>
            <p className="section-kicker">Persistent workspace</p>
            <h3 id="saved-datasets-title">Saved datasets</h3>
            <p className="limit-note">
              Same-name uploads are shown once here; their original saved
              records are preserved.
            </p>
          </div>
          <button
            type="button"
            className="secondary-action"
            onClick={refreshDatasets}
          >
            Refresh
          </button>
        </div>
        {datasetsError && (
          <p className="form-error" role="alert">
            {datasetsError}
          </p>
        )}
        {datasetOpenError && (
          <p className="form-error" role="alert">
            Could not open the selected dataset: {datasetOpenError}
          </p>
        )}
        <div className="history-controls dataset-controls">
          <label>
            <span>Search datasets</span>
            <input
              value={datasetSearch}
              onChange={(event) => setDatasetSearch(event.target.value)}
              placeholder="Search dataset name…"
            />
          </label>
          <label>
            <span>Sort</span>
            <select
              value={datasetSort}
              onChange={(event) => setDatasetSort(event.target.value)}
            >
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
              <option value="name">Name A–Z</option>
            </select>
          </label>
        </div>
        {datasetRows.length ? (
          <div className="table-scroll">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Shape</th>
                  <th>Uploaded</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {datasetRows.map((dataset) => {
                  const isBusy = datasetAction === dataset.id;
                  const linkedExperiment = history.some(
                    (experiment) => experiment.dataset_id === dataset.id,
                  );
                  const savedVersions = datasets.filter(
                    (candidate) =>
                      candidate.original_filename.toLowerCase() ===
                      dataset.original_filename.toLowerCase(),
                  );
                  const savedVersionCount = savedVersions.length;
                  const isOpening = isBusy && datasetActionKind === "open";
                  const isDeleting = isBusy && datasetActionKind === "delete";
                  return (
                    <tr key={dataset.id}>
                      <td>
                        <strong>{dataset.original_filename}</strong>
                        <br />
                        <span className="file-type-badge">
                          {dataset.source_format.toUpperCase()}
                        </span>
                        {savedVersionCount > 1 && (
                          <details className="dataset-versions">
                            <summary>
                              {savedVersionCount} saved versions
                            </summary>
                            {savedVersions.map((version, index) => {
                              const versionLinked = history.some(
                                (experiment) =>
                                  experiment.dataset_id === version.id,
                              );
                              const versionBusy = datasetAction === version.id;
                              return (
                                <div key={version.id}>
                                  <span>
                                    {index === 0
                                      ? "Newest"
                                      : `Version ${savedVersionCount - index}`}{" "}
                                    ·{" "}
                                    {new Date(
                                      version.created_at,
                                    ).toLocaleString()}
                                    {versionLinked ? " · linked" : ""}
                                  </span>
                                  <button
                                    type="button"
                                    className="secondary-action"
                                    onClick={() => openSavedDataset(version.id)}
                                    disabled={Boolean(datasetAction)}
                                  >
                                    {versionBusy && datasetActionKind === "open"
                                      ? "Opening…"
                                      : "Open"}
                                  </button>
                                  <button
                                    type="button"
                                    className="secondary-action danger-action"
                                    onClick={() => removeDataset(version)}
                                    disabled={Boolean(datasetAction)}
                                  >
                                    {versionBusy &&
                                    datasetActionKind === "delete"
                                      ? "Deleting…"
                                      : "Delete version"}
                                  </button>
                                </div>
                              );
                            })}
                          </details>
                        )}
                      </td>
                      <td>
                        {dataset.row_count.toLocaleString()} ×{" "}
                        {dataset.column_count.toLocaleString()}
                      </td>
                      <td>
                        {new Date(dataset.created_at).toLocaleDateString()}
                      </td>
                      <td>
                        <span className="status-badge status-complete">
                          {displayLabel(dataset.status)}
                        </span>
                        {linkedExperiment && (
                          <small className="dataset-in-use">
                            Deleting also removes linked experiment
                          </small>
                        )}
                      </td>
                      <td>
                        <div className="history-actions">
                          <button
                            type="button"
                            className="secondary-action"
                            onClick={() => openSavedDataset(dataset.id)}
                            disabled={Boolean(datasetAction)}
                          >
                            {isOpening ? "Opening…" : "Open"}
                          </button>
                          <button
                            type="button"
                            className="secondary-action"
                            onClick={() =>
                              openSavedDataset(dataset.id, "analytics")
                            }
                            disabled={Boolean(datasetAction)}
                          >
                            Analytics
                          </button>
                          <button
                            type="button"
                            onClick={() =>
                              openSavedDataset(dataset.id, "experiment")
                            }
                            disabled={Boolean(datasetAction)}
                          >
                            New Experiment
                          </button>
                          <button
                            type="button"
                            title="Delete this saved dataset version and any linked experiment outputs"
                            aria-label={`Delete ${dataset.original_filename}`}
                            className="secondary-action danger-action compact-delete"
                            onClick={() => removeDataset(dataset)}
                            disabled={Boolean(datasetAction)}
                          >
                            {isDeleting
                              ? "Deleting…"
                              : savedVersionCount > 1
                                ? "Delete newest"
                                : "Delete"}
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <section className="history-empty">
            <h4>
              {visibleDatasets.length
                ? "No datasets match this search"
                : "No saved datasets yet"}
            </h4>
            <p>
              {visibleDatasets.length
                ? "Try a different dataset name."
                : "Upload a CSV or XLSX file from New Experiment to begin."}
            </p>
          </section>
        )}
      </section>
    );
  }

  function DashboardPanel() {
    const recentExperiments = history.slice(0, 5);
    const recentDatasets = visibleDatasets.slice(0, 5);
    return (
      <section
        className="dashboard-workspace"
        id="dashboard"
        aria-labelledby="page-title"
      >
        <div className="dashboard-page-heading">
          <div>
            <p className="section-kicker">AutoDS-Agent</p>
            <h1 id="page-title">Dashboard</h1>
            <p>Your AutoDS data-science workspace</p>
          </div>
          <button type="button" onClick={() => openStage("workspace")}>
            New Experiment
          </button>
        </div>
        <div className="dashboard-summary-grid">
          <article>
            <span>Total datasets</span>
            <strong>{visibleDatasets.length}</strong>
            <small>Saved in your workspace</small>
          </article>
          <article>
            <span>Total experiments</span>
            <strong>{history.length}</strong>
            <small>Persisted experiment records</small>
          </article>
          <article>
            <span>Completed experiments</span>
            <strong>{completedExperimentCount}</strong>
            <small>Evaluated or completed</small>
          </article>
          <article>
            <span>Reports</span>
            <strong>{reportHistory.length}</strong>
            <small>Available for download</small>
          </article>
        </div>
        {!history.length && !visibleDatasets.length ? (
          <section className="dashboard-empty">
            <h2>No experiments yet</h2>
            <p>Upload a dataset and start your first verified experiment.</p>
            <button type="button" onClick={() => openStage("workspace")}>
              Start your first experiment
            </button>
          </section>
        ) : (
          <div className="dashboard-grid">
            <section className="dashboard-card dashboard-wide">
              <div className="section-heading">
                <div>
                  <p className="section-kicker">Recent work</p>
                  <h2>Recent experiments</h2>
                </div>
                <button
                  type="button"
                  className="secondary-action"
                  onClick={() => openStage("history")}
                >
                  View all experiments
                </button>
              </div>
              {recentExperiments.length ? (
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Dataset</th>
                        <th>Objective</th>
                        <th>Task</th>
                        <th>Target</th>
                        <th>Model</th>
                        <th>Status</th>
                        <th>Primary metric</th>
                        <th>Created</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentExperiments.map((item) => (
                        <tr
                          key={item.experiment_id}
                          className="clickable-row"
                          onClick={() => {
                            openStage("history");
                            handleOpenExperiment(item.experiment_id);
                          }}
                        >
                          <td>
                            {visibleDatasets.find(
                              (dataset) => dataset.id === item.dataset_id,
                            )?.original_filename || "Saved dataset"}
                          </td>
                          <td>{item.objective}</td>
                          <td>{displayLabel(item.task_type)}</td>
                          <td>{item.target_column || "—"}</td>
                          <td>
                            {item.selected_model_run_id ? "Selected" : "—"}
                          </td>
                          <td>{displayLabel(item.status)}</td>
                          <td>{displayLabel(item.primary_metric)}</td>
                          <td>
                            {new Date(item.created_at).toLocaleDateString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="limit-note">No experiments yet.</p>
              )}
            </section>
            <section className="dashboard-card">
              <p className="section-kicker">Activity</p>
              <h2>Experiments by task</h2>
              {taskActivity.length ? (
                <ResponsiveContainer width="100%" height={230}>
                  <BarChart data={taskActivity}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="task"
                      interval={0}
                      angle={taskActivity.length > 3 ? -20 : 0}
                      textAnchor={taskActivity.length > 3 ? "end" : "middle"}
                      height={taskActivity.length > 3 ? 60 : 30}
                    />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#1687d4" radius={[5, 5, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="limit-note">
                  Activity appears after your first experiment.
                </p>
              )}
            </section>
            <section className="dashboard-card dashboard-wide">
              <div className="section-heading">
                <div>
                  <p className="section-kicker">Data library</p>
                  <h2>Recent datasets</h2>
                </div>
              </div>
              {recentDatasets.length ? (
                <div className="dataset-list">
                  {recentDatasets.map((dataset) => (
                    <article key={dataset.id}>
                      <div>
                        <strong>{dataset.original_filename}</strong>
                        <span>
                          {dataset.row_count.toLocaleString()} rows ×{" "}
                          {dataset.column_count} columns ·{" "}
                          {new Date(dataset.created_at).toLocaleDateString()}
                        </span>
                        {profile?.dataset.id === dataset.id &&
                          autoAnalysis?.target_recommendation && (
                            <small>
                              {displayLabel(
                                autoAnalysis.target_recommendation
                                  .recommended_task,
                              )}{" "}
                              · target:{" "}
                              {autoAnalysis.target_recommendation
                                .recommended_target || "not recommended"}
                            </small>
                          )}
                      </div>
                      <button
                        type="button"
                        className="secondary-action"
                        onClick={() =>
                          openSavedDataset(dataset.id, "analytics")
                        }
                      >
                        Analytics
                      </button>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="limit-note">No saved datasets yet.</p>
              )}
            </section>
          </div>
        )}
        <section className="dashboard-card agent-status-card">
          <div className="section-heading">
            <div>
              <p className="section-kicker">Multi-agent status</p>
              <h2>Controlled Multi-Agent Workflow</h2>
            </div>
          </div>
          <div className="agent-flow">
            <article>
              <span className="agent-icon" aria-hidden="true">
                ◫
              </span>
              <strong>Data Analyst Agent</strong>
              <p>Profiles data and identifies useful patterns.</p>
            </article>
            <span className="agent-arrow" aria-hidden="true">
              →
            </span>
            <article>
              <span className="agent-icon" aria-hidden="true">
                ⌘
              </span>
              <strong>ML Planner Agent</strong>
              <p>Creates a validated ML pipeline plan.</p>
            </article>
            <span className="agent-arrow" aria-hidden="true">
              →
            </span>
            <article>
              <span className="agent-icon" aria-hidden="true">
                ✓
              </span>
              <strong>Trusted ML Engine</strong>
              <p>Trains, evaluates and verifies models using trusted code.</p>
            </article>
            <span className="agent-arrow" aria-hidden="true">
              →
            </span>
            <article>
              <span className="agent-icon" aria-hidden="true">
                ✦
              </span>
              <strong>Insight &amp; Reflection Agent</strong>
              <p>Explains verified results and recommends next steps.</p>
            </article>
          </div>
          <p className="agent-trust-note">
            AI plans and interprets. Verified local code performs calculations,
            training and evaluation.
          </p>
        </section>
      </section>
    );
  }

  function ReportsPage() {
    const htmlReportCount = reportHistory.filter(
      (item) => item.format === "html",
    ).length;
    const pdfReportCount = reportHistory.filter(
      (item) => item.format === "pdf",
    ).length;
    const visibleGroups = filteredReportGroups.slice(0, reportVisibleCount);
    const previewDetail =
      selectedReportGroup && historyDetails[selectedReportGroup.experimentId];
    const previewHistory =
      selectedReportGroup &&
      history.find(
        (item) => item.experiment_id === selectedReportGroup.experimentId,
      );
    const previewDatasetId =
      previewDetail?.dataset_id || previewHistory?.dataset_id;
    const previewSelectedRun = previewDetail?.model_runs?.find(
      (run) => run.id === previewDetail.selected_model_run_id,
    );
    const previewMetric =
      previewDetail?.evaluation?.primary_metric ||
      previewHistory?.primary_metric;
    const previewComparison =
      previewDetail?.evaluation?.validation_comparison?.find(
        (item) => item.model_run_id === previewDetail?.selected_model_run_id,
      );
    return (
      <section
        className="history-panel reports-page reports-workspace"
        id="reports"
        aria-labelledby="reports-title"
      >
        <div className="section-heading">
          <div>
            <p className="section-kicker">Saved outputs</p>
            <h2 id="reports-title">Report history</h2>
          </div>
          <button
            type="button"
            className="secondary-action"
            onClick={refreshReportHistory}
          >
            Refresh
          </button>
        </div>
        <p className="report-info-message">
          Generate reports from a completed experiment in New Experiment.
        </p>
        <div className="report-summary-grid">
          <article>
            <span>Total reports</span>
            <strong>{reportHistory.length}</strong>
          </article>
          <article>
            <span>HTML reports</span>
            <strong>{htmlReportCount}</strong>
          </article>
          <article>
            <span>PDF reports</span>
            <strong>{pdfReportCount}</strong>
          </article>
          <article>
            <span>Experiments with reports</span>
            <strong>{reportGroups.length}</strong>
          </article>
        </div>
        {reportHistoryError && (
          <p className="form-error" role="alert">
            {reportHistoryError}
          </p>
        )}
        {reportGroups.length ? (
          <>
            <div className="report-controls">
              <label>
                <span>Search reports</span>
                <input
                  value={reportSearch}
                  onChange={(event) => {
                    setReportSearch(event.target.value);
                    setReportVisibleCount(8);
                  }}
                  placeholder="Dataset, objective, or task…"
                />
              </label>
              <label>
                <span>Format</span>
                <select
                  value={reportFormatFilter}
                  onChange={(event) => {
                    setReportFormatFilter(event.target.value);
                    setReportVisibleCount(8);
                  }}
                >
                  <option value="all">All formats</option>
                  <option value="html">HTML available</option>
                  <option value="pdf">PDF available</option>
                </select>
              </label>
              <label>
                <span>Task</span>
                <select
                  value={reportTaskFilter}
                  onChange={(event) => {
                    setReportTaskFilter(event.target.value);
                    setReportVisibleCount(8);
                  }}
                >
                  <option value="all">All tasks</option>
                  {reportTaskOptions.map((task) => (
                    <option key={task} value={task}>
                      {displayLabel(task)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Dataset</span>
                <select
                  value={reportDatasetFilter}
                  onChange={(event) => {
                    setReportDatasetFilter(event.target.value);
                    setReportVisibleCount(8);
                  }}
                >
                  <option value="all">All datasets</option>
                  {reportDatasetOptions.map((datasetId) => (
                    <option key={datasetId} value={datasetId}>
                      {visibleDatasets.find(
                        (dataset) => dataset.id === datasetId,
                      )?.original_filename || "Saved dataset"}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Sort</span>
                <select
                  value={reportSort}
                  onChange={(event) => setReportSort(event.target.value)}
                >
                  <option value="newest">Newest first</option>
                  <option value="oldest">Oldest first</option>
                </select>
              </label>
            </div>
            {visibleGroups.length ? (
              <div className="table-scroll">
                <table className="reports-table">
                  <thead>
                    <tr>
                      <th>Dataset / objective</th>
                      <th>Task / model</th>
                      <th>Generated</th>
                      <th>Formats</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleGroups.map((group) => {
                      const detail = historyDetails[group.experimentId];
                      const historyItem = history.find(
                        (item) => item.experiment_id === group.experimentId,
                      );
                      const datasetId =
                        detail?.dataset_id || historyItem?.dataset_id;
                      const datasetName =
                        visibleDatasets.find(
                          (dataset) => dataset.id === datasetId,
                        )?.original_filename || "Saved dataset";
                      const selectedRun = detail?.model_runs?.find(
                        (run) => run.id === detail?.selected_model_run_id,
                      );
                      const html = group.latestByFormat.html;
                      const pdf = group.latestByFormat.pdf;
                      return (
                        <tr
                          key={group.experimentId}
                          className={
                            selectedReportExperimentId === group.experimentId
                              ? "selected-report-row"
                              : ""
                          }
                          onClick={() =>
                            setSelectedReportExperimentId(group.experimentId)
                          }
                        >
                          <td>
                            <strong>{datasetName}</strong>
                            <br />
                            <span
                              title={group.objective}
                              className="short-objective"
                            >
                              {group.objective}
                            </span>
                          </td>
                          <td>
                            {displayLabel(
                              detail?.task_type || historyItem?.task_type,
                            )}{" "}
                            ·{" "}
                            {selectedRun?.model_name
                              ? displayLabel(selectedRun.model_name)
                              : detail
                                ? "Not selected"
                                : "Loading…"}
                          </td>
                          <td>
                            {new Date(
                              group.items[0].created_at,
                            ).toLocaleString()}
                          </td>
                          <td>
                            <div className="format-badges">
                              {html && (
                                <span className="format-badge">HTML</span>
                              )}
                              {pdf && <span className="format-badge">PDF</span>}
                              {group.items.length >
                                Object.keys(group.latestByFormat).length && (
                                <small>
                                  Previous versions (
                                  {group.items.length -
                                    Object.keys(group.latestByFormat).length}
                                  )
                                </small>
                              )}
                            </div>
                          </td>
                          <td>
                            <div className="report-actions">
                              {html && (
                                <button
                                  type="button"
                                  className="secondary-action"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    handleArtifactView(
                                      `view-report-${html.report_id}`,
                                      html,
                                    );
                                  }}
                                  disabled={Boolean(downloadAction)}
                                >
                                  {downloadAction ===
                                  `view-report-${html.report_id}`
                                    ? "Opening…"
                                    : "View HTML"}
                                </button>
                              )}
                              {html && (
                                <button
                                  type="button"
                                  className="secondary-action"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    handleArtifactDownload(
                                      `report-${html.report_id}`,
                                      html,
                                      "autods-agent-report.html",
                                    );
                                  }}
                                  disabled={Boolean(downloadAction)}
                                >
                                  {downloadAction === `report-${html.report_id}`
                                    ? "Downloading…"
                                    : "Download HTML"}
                                </button>
                              )}
                              {pdf && (
                                <button
                                  type="button"
                                  className="secondary-action"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    handleArtifactDownload(
                                      `report-${pdf.report_id}`,
                                      pdf,
                                      "autods-agent-report.pdf",
                                    );
                                  }}
                                  disabled={Boolean(downloadAction)}
                                >
                                  {downloadAction === `report-${pdf.report_id}`
                                    ? "Downloading…"
                                    : "Download PDF"}
                                </button>
                              )}
                              <button
                                type="button"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  openStage("history");
                                  handleOpenExperiment(group.experimentId);
                                }}
                              >
                                Open Experiment
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <section className="history-empty">
                <h3>No reports match these filters</h3>
                <p>Try a different dataset, task, format, or search.</p>
              </section>
            )}
            {filteredReportGroups.length > visibleGroups.length && (
              <button
                type="button"
                className="load-more"
                onClick={() => setReportVisibleCount((count) => count + 8)}
              >
                Load more reports
              </button>
            )}
            {selectedReportGroup && (
              <section className="report-preview">
                <div className="section-heading">
                  <div>
                    <p className="section-kicker">Selected report</p>
                    <h3>
                      {visibleDatasets.find(
                        (dataset) => dataset.id === previewDatasetId,
                      )?.original_filename || "Saved dataset"}
                    </h3>
                  </div>
                  <button
                    type="button"
                    className="secondary-action"
                    onClick={() => setSelectedReportExperimentId("")}
                  >
                    Close preview
                  </button>
                </div>
                <dl>
                  <div>
                    <dt>Objective</dt>
                    <dd title={selectedReportGroup.objective}>
                      {selectedReportGroup.objective}
                    </dd>
                  </div>
                  <div>
                    <dt>Task</dt>
                    <dd>
                      {displayLabel(
                        previewDetail?.task_type || previewHistory?.task_type,
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>Target</dt>
                    <dd>
                      {previewDetail?.target_column ||
                        previewHistory?.target_column ||
                        "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>Selected model</dt>
                    <dd>
                      {previewSelectedRun?.model_name
                        ? displayLabel(previewSelectedRun.model_name)
                        : previewDetail
                          ? "Not selected"
                          : "Loading…"}
                    </dd>
                  </div>
                  <div>
                    <dt>Primary metric</dt>
                    <dd>
                      {previewMetric
                        ? `${displayLabel(previewMetric)} ${formatMetric(previewMetric, previewComparison?.metrics?.[previewMetric])}`
                        : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>Generated</dt>
                    <dd>
                      {new Date(
                        selectedReportGroup.items[0].created_at,
                      ).toLocaleString()}
                    </dd>
                  </div>
                  <div>
                    <dt>Available formats</dt>
                    <dd>
                      {Object.keys(selectedReportGroup.latestByFormat)
                        .map((format) => format.toUpperCase())
                        .join(" · ")}
                    </dd>
                  </div>
                </dl>
              </section>
            )}
          </>
        ) : (
          <section className="history-empty">
            <h3>No reports generated yet</h3>
            <p>
              Complete an experiment, then generate an HTML or PDF report from
              its results.
            </p>
            <button type="button" onClick={() => openStage("workspace")}>
              Open New Experiment
            </button>
          </section>
        )}
      </section>
    );
  }

  function ReportGenerationPanel() {
    if (!evaluation) return null;
    return (
      <div
        className="evaluation-card report-download-panel"
        aria-labelledby="generate-reports-title"
      >
        <p className="section-kicker">Reports and downloads</p>
        <h3 id="generate-reports-title">
          Generate and download verified reports
        </h3>
        <p className="limit-note">
          Choose an HTML report for interactive charts, or a PDF for sharing and
          submission.
        </p>
        <div className="report-format-grid">
          <article className="report-format-card">
            <span className="format-icon">HTML</span>
            <h4>Interactive report</h4>
            <p>Detailed browser report with charts and analysis.</p>
            <button
              type="button"
              onClick={handleReport}
              disabled={workflowAction === "report"}
            >
              {workflowAction === "report"
                ? "Generating HTML…"
                : "Generate HTML report"}
            </button>
            {report ? (
              <button
                type="button"
                className="download-action"
                onClick={() =>
                  handleArtifactDownload(
                    "html-report",
                    report,
                    "autods-agent-report.html",
                  )
                }
                disabled={Boolean(downloadAction)}
              >
                {downloadAction === "html-report"
                  ? "Downloading…"
                  : "Download HTML report"}
              </button>
            ) : (
              <small className="report-status">Not generated yet</small>
            )}
          </article>
          <article className="report-format-card">
            <span className="format-icon">PDF</span>
            <h4>Professional PDF</h4>
            <p>Polished format for submission, printing, and sharing.</p>
            <button
              type="button"
              onClick={handlePdfReport}
              disabled={workflowAction === "PDF report"}
            >
              {workflowAction === "PDF report"
                ? "Generating PDF…"
                : "Generate PDF report"}
            </button>
            {pdfReport ? (
              <button
                type="button"
                className="download-action"
                onClick={() =>
                  handleArtifactDownload(
                    "pdf-report",
                    pdfReport,
                    "autods-agent-report.pdf",
                  )
                }
                disabled={Boolean(downloadAction)}
              >
                {downloadAction === "pdf-report"
                  ? "Downloading…"
                  : "Download PDF report"}
              </button>
            ) : (
              <small className="report-status">Not generated yet</small>
            )}
          </article>
        </div>
        {downloadError && (
          <p className="form-error" role="alert">
            {downloadError}
          </p>
        )}
        <div className="report-panel-footer">
          <span>
            {report || pdfReport
              ? "Your reports are saved in Report history."
              : "Reports use verified experiment results."}
          </span>
          <button
            type="button"
            className="stage-next"
            onClick={() => openStage("reports")}
          >
            View report history →
          </button>
        </div>
      </div>
    );
  }

  if (authLoading)
    return (
      <main className="page-shell">
        <section className="hero">
          <p>Restoring secure session…</p>
        </section>
      </main>
    );
  if (!user) return <AuthScreen onAuthenticated={setUser} />;

  return (
    <main
      className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""} ${sidebarOpen ? "sidebar-open" : ""} ${lightTheme ? "light-theme" : ""} ${reduceMotion ? "reduce-motion" : ""}`}
    >
      <aside
        className="dashboard-sidebar"
        aria-label="AutoDS workspace navigation"
      >
        <div className="sidebar-brand">
          AutoDS-Agent<span>Intelligent Data Science</span>
        </div>
        <nav className="sidebar-nav">
          <button
            aria-label="Dashboard"
            title="Dashboard"
            className={activeStage === "dashboard" ? "active" : ""}
            onClick={() => openStage("dashboard")}
          >
            <span aria-hidden="true" className="nav-icon">
              D
            </span>
            <span className="sidebar-nav-label">Dashboard</span>
          </button>
          <button
            aria-label="New Experiment"
            title="New Experiment"
            className={activeStage === "workspace" ? "active" : ""}
            onClick={() => openStage("workspace")}
          >
            <span aria-hidden="true" className="nav-icon">
              N
            </span>
            <span className="sidebar-nav-label">New Experiment</span>
          </button>
          <button
            aria-label="History"
            title="History"
            className={
              ["history", "history-detail"].includes(activeStage)
                ? "active"
                : ""
            }
            onClick={() => openStage("history")}
          >
            <span aria-hidden="true" className="nav-icon">
              H
            </span>
            <span className="sidebar-nav-label">History</span>
          </button>
          <button
            aria-label="Ask Q&amp;A"
            title="Ask Q&amp;A"
            className={activeStage === "ask" ? "active" : ""}
            onClick={() => openStage("ask")}
          >
            <span aria-hidden="true" className="nav-icon">
              Q
            </span>
            <span className="sidebar-nav-label">Ask Q&amp;A</span>
          </button>
          <button
            aria-label="Reports"
            title="Reports"
            className={activeStage === "reports" ? "active" : ""}
            onClick={() => openStage("reports")}
          >
            <span aria-hidden="true" className="nav-icon">
              R
            </span>
            <span className="sidebar-nav-label">Reports</span>
          </button>
          <button
            aria-label="Settings"
            title="Settings"
            className={activeStage === "settings" ? "active" : ""}
            onClick={() => openStage("settings")}
          >
            <span aria-hidden="true" className="nav-icon">
              S
            </span>
            <span className="sidebar-nav-label">Settings</span>
          </button>
        </nav>
        <div className="sidebar-status">
          <span
            className={`status-dot ${health ? "online" : error ? "offline" : ""}`}
          />
          {health ? "Analytics online" : "Checking analytics"}
        </div>
      </aside>
      {sidebarOpen && (
        <button
          type="button"
          className="sidebar-scrim"
          aria-label="Close sidebar"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <div
        className={`dashboard-content ${compactLayout ? "compact-layout" : ""}`}
      >
        <header className="dashboard-topbar">
          <button
            type="button"
            className="sidebar-toggle"
            title="Toggle sidebar"
            aria-label="Toggle sidebar"
            aria-expanded={sidebarOpen || !sidebarCollapsed}
            onClick={toggleSidebar}
          >
            <span className="sidebar-toggle-glyph" aria-hidden="true">
              <i />
              <i />
            </span>
          </button>
          <span>{user.email}</span>
          <button
            type="button"
            onClick={() => {
              clearAccessToken();
              setProfile(null);
              setDatasets([]);
              setUser(null);
            }}
          >
            Log out
          </button>
        </header>
        {activeStage === "dashboard" && <DashboardPanel />}
        <section className="hero" id="dashboard" aria-labelledby="page-title">
          {false && activeStage === "dashboard" && (
            <>
              <div className="stage-trail" aria-label="Workflow stages">
                Overview <span>→</span> Upload <span>→</span> Objective{" "}
                <span>→</span> Plan <span>→</span> Run <span>→</span> Ask{" "}
                <span>→</span> Reports
              </div>
              <p className="eyebrow">Intelligent data science automation</p>
              <h1 id="page-title">AutoDS-Agent</h1>
              <p className="summary">
                A safe, modular foundation for turning structured data and
                natural-language objectives into trusted data-science workflows.
              </p>

              <div className="status-card" aria-live="polite">
                <span
                  className={`status-dot ${health ? "online" : error ? "offline" : ""}`}
                />
                <div>
                  <p className="status-label">Backend status</p>
                  {!health && !error && <p>Checking connection…</p>}
                  {health && <p className="healthy">Healthy</p>}
                  {error && <p className="error">{error}</p>}
                </div>
              </div>

              <section
                className="feature-overview"
                aria-label="AutoDS-Agent capabilities"
              >
                <article>
                  <span>01</span>
                  <h3>Understand your data</h3>
                  <p>
                    Upload CSV or Excel data for validated profiling, quality
                    checks, feature roles, and task recommendations.
                  </p>
                </article>
                <article>
                  <span>02</span>
                  <h3>Build trusted pipelines</h3>
                  <p>
                    Confirm the objective, review the generated plan, train
                    baselines, evaluate, optimize, and explain results.
                  </p>
                </article>
                <article>
                  <span>03</span>
                  <h3>Ask and report</h3>
                  <p>
                    Use verified dataset analytics, then create downloadable
                    predictions and HTML/PDF reports.
                  </p>
                </article>
              </section>
              <div className="stage-actions">
                <button type="button" onClick={() => openStage("workspace")}>
                  Start with a dataset
                </button>
                <button
                  type="button"
                  className="secondary-action"
                  onClick={() => openStage("ask")}
                >
                  Ask AutoDS
                </button>
              </div>
            </>
          )}
          {activeStage === "settings" && (
            <section
              className="settings-overview"
              id="settings"
              aria-labelledby="settings-title"
            >
              <p className="section-kicker">Settings</p>
              <h2 id="settings-title">Workspace controls</h2>
              <p>
                Manage your view and refresh saved workspace data. These
                controls never alter datasets, pipelines, or reports.
              </p>
              <div className="settings-status-grid">
                <article>
                  <span>Saved datasets</span>
                  <strong>{visibleDatasets.length}</strong>
                  <small>Available for trusted Q&amp;A</small>
                </article>
                <article>
                  <span>Saved reports</span>
                  <strong>{reportHistory.length}</strong>
                  <small>Available for download</small>
                </article>
                <article>
                  <span>Analytics service</span>
                  <strong className={health ? "healthy" : "error"}>
                    {health ? "Online" : "Checking"}
                  </strong>
                  <small>Protected backend connection</small>
                </article>
              </div>
              <label className="setting-toggle">
                <span>
                  <strong>Light theme</strong>
                  <small>
                    Switch to a brighter workspace view. Your choice is
                    remembered.
                  </small>
                </span>
                <input
                  type="checkbox"
                  checked={lightTheme}
                  onChange={(event) => setLightTheme(event.target.checked)}
                />
              </label>
              <label className="setting-toggle">
                <span>
                  <strong>Compact workspace layout</strong>
                  <small>Reduce spacing between dashboard cards.</small>
                </span>
                <input
                  type="checkbox"
                  checked={compactLayout}
                  onChange={(event) => setCompactLayout(event.target.checked)}
                />
              </label>
              <label className="setting-toggle">
                <span>
                  <strong>Show calculation details</strong>
                  <small>
                    Display trusted calculation steps in Ask Q&amp;A results.
                  </small>
                </span>
                <input
                  type="checkbox"
                  checked={showCalculationDetails}
                  onChange={(event) =>
                    setShowCalculationDetails(event.target.checked)
                  }
                />
              </label>
              <label className="setting-toggle">
                <span>
                  <strong>Reduce motion</strong>
                  <small>
                    Turn off visual motion, including training indicators, for a
                    calmer workspace.
                  </small>
                </span>
                <input
                  type="checkbox"
                  checked={reduceMotion}
                  onChange={(event) => setReduceMotion(event.target.checked)}
                />
              </label>
              <div className="settings-actions">
                <button
                  type="button"
                  onClick={() => {
                    refreshDatasets();
                    refreshHistory();
                    refreshReportHistory();
                  }}
                >
                  Refresh workspace data
                </button>
                <button
                  type="button"
                  className="secondary-action"
                  onClick={() => openStage("history")}
                >
                  Open history
                </button>
                <button
                  type="button"
                  className="secondary-action"
                  onClick={() => openStage("ask")}
                >
                  Open Ask Q&amp;A
                </button>
              </div>
              <div className="settings-readonly">
                <strong>System safeguards</strong>
                <span>
                  Authenticated datasets · validated plans · protected downloads
                </span>
              </div>
            </section>
          )}

          {false && activeStage === "dashboard" && (
            <section
              className="history-panel"
              aria-labelledby="agent-workflow-title"
            >
              <div className="section-heading">
                <div>
                  <p className="section-kicker">AI workflow</p>
                  <h2 id="agent-workflow-title">Controlled multi-agent flow</h2>
                </div>
              </div>
              <div className="plan-grid">
                <div>
                  <dt>Data Analyst Agent</dt>
                  <dd>
                    {datasetAnalysis
                      ? `Completed · ${displayLabel(datasetAnalysis.provider_used)}`
                      : "Waiting for dataset profile"}
                  </dd>
                </div>
                <div>
                  <dt>ML Planner Agent</dt>
                  <dd>
                    {pipelinePlan
                      ? `Completed · ${displayLabel(pipelinePlan.provider_used)}`
                      : "Waiting for confirmation"}
                  </dd>
                </div>
                <div>
                  <dt>Trusted ML Engine</dt>
                  <dd>
                    {evaluation
                      ? "Models trained and evaluated"
                      : "Waiting for evaluation"}
                  </dd>
                </div>
                <div>
                  <dt>Insight & Reflection Agent</dt>
                  <dd>
                    {reflection
                      ? `Reviewed · ${displayLabel(reflection.provider_used)}`
                      : "Waiting for AI review"}
                  </dd>
                </div>
              </div>
            </section>
          )}

          {activeStage === "history" && (
            <section
              className="history-panel history-workspace"
              id="history"
              aria-labelledby="history-title"
            >
              <div className="section-heading">
                <div>
                  <p className="section-kicker">Your data-science journey</p>
                  <h2 id="history-title">Discover what you’ve built</h2>
                  <p className="limit-note">
                    Revisit datasets, explore verified results, and continue any
                    experiment without rerunning its pipeline.
                  </p>
                </div>
              </div>
              <div className="history-summary-grid">
                <article>
                  <span>Total experiments</span>
                  <strong>{history.length}</strong>
                </article>
                <article>
                  <span>Completed / evaluated</span>
                  <strong>{completedExperimentCount}</strong>
                </article>
                <article>
                  <span>In progress</span>
                  <strong>{inProgressExperimentCount}</strong>
                </article>
                <article>
                  <span>Saved datasets</span>
                  <strong>{visibleDatasets.length}</strong>
                </article>
              </div>
              {historyError ? (
                <p className="form-error" role="alert">
                  {historyError}
                </p>
              ) : (
                <>
                  <div className="history-controls">
                    <label>
                      <span>Search experiments</span>
                      <input
                        value={historySearch}
                        onChange={(event) =>
                          setHistorySearch(event.target.value)
                        }
                        placeholder="Objective, dataset, task, or target…"
                      />
                    </label>
                    <label>
                      <span>Task</span>
                      <select
                        value={historyTaskFilter}
                        onChange={(event) =>
                          setHistoryTaskFilter(event.target.value)
                        }
                      >
                        <option value="all">All tasks</option>
                        {historyTaskOptions.map((task) => (
                          <option value={task} key={task}>
                            {displayLabel(task)}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      <span>Status</span>
                      <select
                        value={historyStatusFilter}
                        onChange={(event) =>
                          setHistoryStatusFilter(event.target.value)
                        }
                      >
                        <option value="all">All statuses</option>
                        {historyStatusOptions.map((status) => (
                          <option value={status} key={status}>
                            {displayLabel(status)}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      <span>Sort</span>
                      <select
                        value={historySort}
                        onChange={(event) => setHistorySort(event.target.value)}
                      >
                        <option value="newest">Newest first</option>
                        <option value="oldest">Oldest first</option>
                      </select>
                    </label>
                  </div>
                  {historyRows.length ? (
                    <div className="table-scroll">
                      <table className="history-table">
                        <thead>
                          <tr>
                            <th>Objective</th>
                            <th>Dataset</th>
                            <th>Task / Target</th>
                            <th>Status</th>
                            <th>Selected model</th>
                            <th>Primary metric</th>
                            <th>Created</th>
                            <th>Open</th>
                          </tr>
                        </thead>
                        <tbody>
                          {historyRows.map((item) => {
                            const detail = historyDetails[item.experiment_id];
                            const selectedRun = detail?.model_runs?.find(
                              (run) => run.id === item.selected_model_run_id,
                            );
                            const primaryMetric =
                              detail?.evaluation?.primary_metric ||
                              item.primary_metric;
                            const selectedComparison =
                              detail?.evaluation?.validation_comparison?.find(
                                (comparison) =>
                                  comparison.model_run_id ===
                                  item.selected_model_run_id,
                              );
                            const metricValue = primaryMetric
                              ? selectedComparison?.metrics?.[primaryMetric]
                              : null;
                            const tone = historyStatusTone(item.status);
                            const isOpening =
                              openingExperimentId === item.experiment_id;
                            return (
                              <tr key={item.experiment_id}>
                                <td>
                                  <strong>{item.objective}</strong>
                                </td>
                                <td>
                                  {visibleDatasets.find(
                                    (dataset) => dataset.id === item.dataset_id,
                                  )?.original_filename || "Saved dataset"}
                                </td>
                                <td>
                                  {displayLabel(item.task_type)}
                                  <br />
                                  <small>
                                    Target: {item.target_column || "—"}
                                  </small>
                                </td>
                                <td>
                                  <span
                                    className={`status-badge status-${tone}`}
                                  >
                                    {displayLabel(item.status)}
                                  </span>
                                </td>
                                <td>
                                  {selectedRun?.model_name
                                    ? displayLabel(selectedRun.model_name)
                                    : item.selected_model_run_id
                                      ? "Loading model…"
                                      : "Not selected"}
                                </td>
                                <td>
                                  {primaryMetric ? (
                                    <>
                                      <strong>
                                        {displayLabel(primaryMetric)}
                                      </strong>
                                      <br />
                                      <small>
                                        {formatMetric(
                                          primaryMetric,
                                          metricValue,
                                        )}
                                      </small>
                                    </>
                                  ) : (
                                    "—"
                                  )}
                                </td>
                                <td>
                                  {new Date(
                                    item.created_at,
                                  ).toLocaleDateString()}
                                </td>
                                <td>
                                  <button
                                    type="button"
                                    onClick={() =>
                                      handleOpenExperiment(item.experiment_id)
                                    }
                                    disabled={isOpening}
                                  >
                                    {isOpening ? "Opening…" : "Open"}
                                  </button>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <section className="history-empty">
                      <h3>
                        {history.length
                          ? "No experiments match these filters"
                          : "No experiments yet"}
                      </h3>
                      <p>
                        {history.length
                          ? "Try a different search, task, or status."
                          : "Start with a saved dataset to create your first verified experiment."}
                      </p>
                      {!history.length && (
                        <button
                          type="button"
                          onClick={() => openStage("workspace")}
                        >
                          Start your first experiment
                        </button>
                      )}
                    </section>
                  )}
                </>
              )}
              <SavedDatasetsPanel />
            </section>
          )}

          {activeStage === "ask" && (
            <section
              className="history-panel ask-workspace"
              id="ask"
              aria-labelledby="assistant-title"
            >
              <div className="section-heading">
                <div>
                  <p className="section-kicker">Trusted analytics workspace</p>
                  <h2 id="assistant-title">Ask Q&amp;A</h2>
                  <p className="ask-subtitle">
                    Ask natural-language questions and explore only verified
                    dataset or experiment results.
                  </p>
                </div>
              </div>
              <div
                className="assistant-tabs"
                role="tablist"
                aria-label="Question context"
              >
                <button
                  type="button"
                  role="tab"
                  aria-selected={assistantContext === "dataset"}
                  className={assistantContext === "dataset" ? "active" : ""}
                  onClick={() => {
                    setAssistantContext("dataset");
                    setAssistantResult(null);
                    setAssistantError("");
                  }}
                  disabled={!visibleDatasets.length}
                >
                  Dataset Questions
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={assistantContext === "experiment"}
                  className={assistantContext === "experiment" ? "active" : ""}
                  onClick={() => {
                    setAssistantContext("experiment");
                    setAssistantResult(null);
                    setAssistantError("");
                  }}
                  disabled={!assistantExperimentOptions.length}
                >
                  Experiment Questions
                </button>
              </div>
              {assistantContext === "dataset" ? (
                <>
                  <section className="ask-context-card">
                    <div>
                      <p className="section-kicker">Dataset context</p>
                      <h3>Choose a saved dataset</h3>
                    </div>
                    <label className="dataset-search">
                      <span>Search datasets</span>
                      <input
                        value={assistantDatasetSearch}
                        onChange={(event) =>
                          setAssistantDatasetSearch(event.target.value)
                        }
                        placeholder="Search by dataset name…"
                      />
                    </label>
                    <div
                      className="dataset-selector"
                      role="listbox"
                      aria-label="Saved datasets"
                    >
                      {filteredAssistantDatasets.length ? (
                        filteredAssistantDatasets.map((dataset) => (
                          <button
                            type="button"
                            role="option"
                            aria-selected={profile?.dataset.id === dataset.id}
                            className={
                              profile?.dataset.id === dataset.id
                                ? "selected"
                                : ""
                            }
                            key={dataset.id}
                            onClick={() =>
                              openDataset(dataset.id, {
                                preserveExperiment: true,
                              })
                            }
                            disabled={Boolean(datasetAction)}
                          >
                            <strong>{dataset.original_filename}</strong>
                            <span>
                              {dataset.row_count.toLocaleString()} rows ×{" "}
                              {dataset.column_count.toLocaleString()} columns
                            </span>
                          </button>
                        ))
                      ) : (
                        <p className="limit-note">
                          No saved datasets match your search.
                        </p>
                      )}
                    </div>
                  </section>
                  {profile ? (
                    <section className="selected-dataset-card">
                      <div>
                        <span className="verified-badge">Selected dataset</span>
                        <h3>{profile.dataset.original_filename}</h3>
                      </div>
                      <dl>
                        <div>
                          <dt>Rows</dt>
                          <dd>{profile.summary.row_count.toLocaleString()}</dd>
                        </div>
                        <div>
                          <dt>Columns</dt>
                          <dd>
                            {profile.summary.column_count.toLocaleString()}
                          </dd>
                        </div>
                        <div>
                          <dt>Missing values</dt>
                          <dd>
                            {profile.summary.total_missing_values.toLocaleString()}
                          </dd>
                        </div>
                        <div>
                          <dt>Recommended task</dt>
                          <dd>
                            {selectedDatasetTask
                              ? displayLabel(selectedDatasetTask)
                              : "Exploratory analysis"}
                          </dd>
                        </div>
                        <div>
                          <dt>Recommended target</dt>
                          <dd>{selectedDatasetTarget || "Not recommended"}</dd>
                        </div>
                      </dl>
                    </section>
                  ) : (
                    <section className="ask-empty-state">
                      <h3>Select a dataset to begin</h3>
                      <p>
                        Its profile and verified question suggestions will
                        appear here.
                      </p>
                    </section>
                  )}
                  <section className="suggested-questions">
                    <p className="section-kicker">Suggested questions</p>
                    <div>
                      {datasetQuestionSuggestions.map((question) => (
                        <button
                          type="button"
                          key={question}
                          onClick={() => setAssistantQuestion(question)}
                        >
                          {question}
                        </button>
                      ))}
                    </div>
                  </section>
                </>
              ) : (
                <section className="selected-dataset-card experiment-context-card">
                  <label className="dataset-search">
                    <span>Select experiment</span>
                    <select
                      value={selectedAssistantExperimentId}
                      onChange={(event) => {
                        setAssistantExperimentId(event.target.value);
                        setAssistantResult(null);
                        setAssistantError("");
                      }}
                    >
                      <option value="">Choose a saved experiment…</option>
                      {assistantExperimentOptions.map((item) => (
                        <option
                          key={item.experiment_id}
                          value={item.experiment_id}
                        >
                          {item.objective || "Untitled experiment"}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div>
                    <span className="verified-badge">
                      Verified experiment context
                    </span>
                    <h3>
                      {selectedAssistantExperiment?.objective ||
                        "Select a saved experiment"}
                    </h3>
                  </div>
                  {selectedAssistantExperiment ? (
                    <dl>
                      <div>
                        <dt>Dataset</dt>
                        <dd>
                          {visibleDatasets.find(
                            (dataset) =>
                              dataset.id ===
                              selectedAssistantExperiment.dataset_id,
                          )?.original_filename || "Saved dataset"}
                        </dd>
                      </div>
                      <div>
                        <dt>Task</dt>
                        <dd>
                          {displayLabel(selectedAssistantExperiment.task_type)}
                        </dd>
                      </div>
                      <div>
                        <dt>Target</dt>
                        <dd>
                          {selectedAssistantExperiment.target_column || "—"}
                        </dd>
                      </div>
                      <div>
                        <dt>Status</dt>
                        <dd>
                          {displayLabel(selectedAssistantExperiment.status)}
                        </dd>
                      </div>
                      <div>
                        <dt>Scope</dt>
                        <dd>Persisted verified results only</dd>
                      </div>
                    </dl>
                  ) : (
                    <p>
                      Select an evaluated experiment to ask about its persisted
                      model results.
                    </p>
                  )}
                </section>
              )}
              <div className="assistant-active-context" aria-live="polite">
                <strong>
                  {assistantContext === "dataset"
                    ? "Dataset Questions"
                    : "Experiment Questions"}
                </strong>
                <span>
                  {assistantContext === "dataset"
                    ? `Dataset: ${profile?.dataset.original_filename || "Select a dataset above"}`
                    : `Experiment: ${selectedAssistantExperiment?.objective || "Select an experiment above"}`}
                </span>
              </div>
              <form className="assistant-composer" onSubmit={submitAssistant}>
                <label htmlFor="assistant-question">Question</label>
                <div>
                  <textarea
                    id="assistant-question"
                    value={assistantQuestion}
                    placeholder={
                      assistantContext === "dataset"
                        ? "Ask anything about this dataset…"
                        : "Ask about the selected experiment’s verified results…"
                    }
                    onChange={(event) =>
                      setAssistantQuestion(event.target.value)
                    }
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        event.currentTarget.form?.requestSubmit();
                      }
                    }}
                    rows="3"
                  />
                  <button
                    type="submit"
                    aria-label="Ask AutoDS"
                    disabled={
                      assistantBusy ||
                      !assistantQuestion.trim() ||
                      (assistantContext === "dataset" && !profile) ||
                      (assistantContext === "experiment" &&
                        !selectedAssistantExperimentId)
                    }
                  >
                    {assistantBusy ? (
                      <>
                        <span className="loading-spinner" />
                        Analyzing…
                      </>
                    ) : (
                      <>
                        Send <span aria-hidden="true">→</span>
                      </>
                    )}
                  </button>
                </div>
                <small>
                  Press Enter to submit · Shift + Enter for a new line
                </small>
              </form>
              {assistantContext === "experiment" && (
                <p className="limit-note">
                  Experiment Questions answer only from persisted, verified
                  results after training and evaluation.
                </p>
              )}
              {assistantError && (
                <p className="form-error" role="alert">
                  {assistantError}
                </p>
              )}
              {assistantBusy && (
                <section className="ask-loading-state" aria-live="polite">
                  <span className="loading-spinner" />
                  Building and executing a validated analytics request…
                </section>
              )}
              {assistantResult && (
                <div className="assistant-answer">
                  <AnalyticsDashboard
                    result={assistantResult}
                    onQuestion={setAssistantQuestion}
                    showCalculationDetails={showCalculationDetails}
                  />
                  <div className="assistant-meta">
                    <span>Grounded in verified AutoDS results</span>
                    <span className="provider-badge">
                      {displayLabel(assistantResult.provider_used)}
                    </span>
                  </div>
                </div>
              )}
            </section>
          )}

          {activeStage === "history-detail" && openedExperiment && (
            <section
              className="history-panel history-detail"
              aria-labelledby="reopened-title"
            >
              <div className="section-heading">
                <div>
                  <p className="section-kicker">Experiment history</p>
                  <h2 id="reopened-title">Persisted experiment details</h2>
                </div>
                <div className="history-detail-actions">
                  <span className="success-badge">
                    {displayLabel(openedExperiment.status)}
                  </span>
                  <button
                    type="button"
                    className="secondary-action"
                    onClick={() => openStage("history")}
                  >
                    ← Back to history
                  </button>
                </div>
              </div>
              <dl className="plan-grid">
                <div>
                  <dt>Objective</dt>
                  <dd>{openedExperiment.objective}</dd>
                </div>
                <div>
                  <dt>Task</dt>
                  <dd>{displayLabel(openedExperiment.task_type)}</dd>
                </div>
                <div>
                  <dt>Target</dt>
                  <dd>{openedExperiment.target_column || "—"}</dd>
                </div>
                <div>
                  <dt>Selected model</dt>
                  <dd>
                    {openedExperiment.model_runs?.find(
                      (run) =>
                        run.id === openedExperiment.selected_model_run_id,
                    )?.model_name
                      ? displayLabel(
                          openedExperiment.model_runs.find(
                            (run) =>
                              run.id === openedExperiment.selected_model_run_id,
                          ).model_name,
                        )
                      : "Not selected yet"}
                  </dd>
                </div>
                {openedExperiment.pipeline_plan && (
                  <div>
                    <dt>Validated plan metric</dt>
                    <dd>
                      {displayLabel(
                        openedExperiment.pipeline_plan.primary_metric,
                      )}
                    </dd>
                  </div>
                )}
              </dl>
              {openedExperiment.evaluation ? (
                <>
                  <h3>Saved evaluation</h3>
                  <p>
                    Primary metric:{" "}
                    {displayLabel(openedExperiment.evaluation.primary_metric)}.
                    Final test:{" "}
                    {Object.entries(
                      openedExperiment.evaluation.final_test_metrics,
                    )
                      .map(
                        ([name, value]) =>
                          `${displayLabel(name)} ${formatStatistic(value)}`,
                      )
                      .join(" · ")}
                  </p>
                </>
              ) : (
                <p className="limit-note">No evaluation has been saved yet.</p>
              )}
              {openedExperiment.optimization && (
                <p>
                  Optimization:{" "}
                  {displayLabel(openedExperiment.optimization.status)} ·{" "}
                  {openedExperiment.optimization.trial_count} trials.
                </p>
              )}
              <p className="limit-note">
                This view reads saved metadata and artifacts; it does not rerun
                the experiment.
              </p>
            </section>
          )}

          {activeStage === "workspace" && (
            <ol className="experiment-stepper" aria-label="Experiment workflow">
              {workflowSteps.map((step, index) => (
                <li
                  key={step}
                  className={index <= completedStep ? "complete" : ""}
                >
                  {step}
                </li>
              ))}
            </ol>
          )}
          {activeStage === "workspace" && activeJob && (
            <section className="job-panel" aria-live="polite">
              <p className="section-kicker">Experiment progress</p>
              <strong>{displayLabel(activeJob.stage)}</strong>
              <p>
                {displayLabel(activeJob.status)} — background job{" "}
                {activeJob.job_type}
              </p>
            </section>
          )}

          {activeStage === "workspace" && (
            <section
              className="upload-panel"
              id="workspace"
              aria-labelledby="upload-title"
            >
              <div className="section-heading">
                <div>
                  <p className="section-kicker">Dataset intake</p>
                  <h2 id="upload-title">
                    Step 1 — Upload a CSV or Excel dataset
                  </h2>
                </div>
                <span className="limit-note">CSV and XLSX</span>
              </div>

              <form className="upload-form" onSubmit={handleUpload}>
                <label className="file-picker">
                  <span>
                    {selectedFile
                      ? selectedFile.name
                      : "Choose a structured CSV or XLSX file"}
                  </span>
                  <input
                    type="file"
                    accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    onChange={(event) => {
                      setSelectedFile(event.target.files?.[0] || null);
                      setUploadError("");
                    }}
                  />
                </label>
                <button type="submit" disabled={!selectedFile || isUploading}>
                  {isUploading
                    ? "Validating and profiling…"
                    : "Upload and analyze"}
                </button>
              </form>
              {uploadError && (
                <p className="form-error" role="alert">
                  {uploadError}
                </p>
              )}
            </section>
          )}

          {activeStage === "workspace" && profile && (
            <section
              className="profile-panel"
              id="dataset-analytics-panel"
              aria-labelledby="profile-title"
            >
              <div className="section-heading">
                <div>
                  <p className="section-kicker">Dataset profile</p>
                  <h2 id="profile-title">
                    {profile.dataset.original_filename}
                  </h2>
                </div>
                <span className="success-badge">Ready</span>
              </div>

              <div className="summary-grid">
                <article>
                  <span>Rows</span>
                  <strong>{profile.summary.row_count}</strong>
                </article>
                <article>
                  <span>Columns</span>
                  <strong>{profile.summary.column_count}</strong>
                </article>
                <article>
                  <span>Duplicates</span>
                  <strong>{profile.summary.duplicate_row_count}</strong>
                </article>
                <article>
                  <span>Missing values</span>
                  <strong>{profile.summary.total_missing_values}</strong>
                </article>
              </div>

              {datasetAnalysis && (
                <section
                  className="analysis-card"
                  aria-labelledby="analysis-title"
                >
                  <div className="section-heading">
                    <div>
                      <p className="section-kicker">AI dataset analysis</p>
                      <h3 id="analysis-title">Data Analyst Agent</h3>
                    </div>
                    <span className="provider-badge">
                      {displayLabel(datasetAnalysis.provider_used)}
                    </span>
                  </div>
                  <p>{datasetAnalysis.summary}</p>
                  {datasetAnalysis.quality_issues.length > 0 && (
                    <ul>
                      {datasetAnalysis.quality_issues
                        .slice(0, 5)
                        .map((issue) => (
                          <li key={`${issue.column}-${issue.issue}`}>
                            <strong>{issue.column}:</strong> {issue.explanation}
                          </li>
                        ))}
                    </ul>
                  )}
                  {datasetAnalysis.feature_observations.length > 0 && (
                    <ul>
                      {datasetAnalysis.feature_observations
                        .slice(0, 5)
                        .map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                    </ul>
                  )}
                </section>
              )}

              {autoAnalysis && (
                <section
                  className="analysis-card"
                  aria-labelledby="auto-analysis-title"
                >
                  <p className="section-kicker">Automatic recommendation</p>
                  <h3 id="auto-analysis-title">
                    {autoAnalysis.target_recommendation.recommended_task
                      ? displayLabel(
                          autoAnalysis.target_recommendation.recommended_task,
                        )
                      : "Exploratory analysis"}
                  </h3>
                  <dl className="plan-grid">
                    <div>
                      <dt>Recommended target</dt>
                      <dd>
                        {autoAnalysis.target_recommendation
                          .recommended_target ||
                          "No suitable supervised target"}
                      </dd>
                    </div>
                    <div>
                      <dt>Confidence</dt>
                      <dd>
                        {displayLabel(
                          autoAnalysis.target_recommendation.confidence,
                        )}
                      </dd>
                    </div>
                    <div>
                      <dt>Reason</dt>
                      <dd>{autoAnalysis.target_recommendation.reason}</dd>
                    </div>
                    <div>
                      <dt>Recommended models</dt>
                      <dd>
                        {autoAnalysis.recommended_algorithms
                          .map((item) => item.display_name)
                          .join(", ") ||
                          "Clustering or anomaly detection available"}
                      </dd>
                    </div>
                  </dl>
                  {autoAnalysis.target_recommendation.recommended_target && (
                    <button
                      type="button"
                      onClick={() => {
                        const target =
                          autoAnalysis.target_recommendation.recommended_target;
                        setObjective(`Predict ${target}`);
                        setSelectedTarget(target);
                        setSelectedTask(
                          autoAnalysis.target_recommendation.recommended_task,
                        );
                      }}
                    >
                      Accept recommendation
                    </button>
                  )}
                  <details>
                    <summary>Advanced / Manual Settings</summary>
                    <ul>
                      {autoAnalysis.feature_roles.map((item) => (
                        <li key={item.column}>
                          <strong>{item.column}</strong> →{" "}
                          {displayLabel(item.role)}
                          {item.excluded_by_default
                            ? " (excluded by default)"
                            : ""}
                        </li>
                      ))}
                    </ul>
                  </details>
                </section>
              )}

              {!profile.dataset.has_header && (
                <p className="header-notice" role="status">
                  No CSV header was detected. Temporary column names were
                  generated.
                </p>
              )}

              <details className="advanced-details dataset-profile-details">
                <summary>View full dataset profile / Advanced details</summary>
                <div className="table-block">
                  <h3>Columns</h3>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Type</th>
                          <th>Missing</th>
                          <th>Missing %</th>
                          <th>Unique</th>
                          <th>Possible ID</th>
                        </tr>
                      </thead>
                      <tbody>
                        {profile.columns.map((column) => (
                          <tr key={column.name}>
                            <td>{column.name}</td>
                            <td>
                              <span className="type-pill">
                                {column.logical_type}
                              </span>
                            </td>
                            <td>{column.missing_count}</td>
                            <td>
                              {formatStatistic(column.missing_percentage)}%
                            </td>
                            <td>{column.unique_count}</td>
                            <td>{column.is_possible_id ? "Yes" : "No"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {numericalProfiles.length > 0 && (
                  <div className="table-block">
                    <h3>Numerical summary</h3>
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Column</th>
                            <th>Count</th>
                            <th>Mean</th>
                            <th>Std</th>
                            <th>Min</th>
                            <th>25%</th>
                            <th>Median</th>
                            <th>75%</th>
                            <th>Max</th>
                          </tr>
                        </thead>
                        <tbody>
                          {numericalProfiles.map((column) => {
                            const stats = column.numeric_statistics;
                            return (
                              <tr key={column.name}>
                                <td>{column.name}</td>
                                <td>{stats.count}</td>
                                <td>{formatStatistic(stats.mean)}</td>
                                <td>{formatStatistic(stats.std)}</td>
                                <td>{formatStatistic(stats.min)}</td>
                                <td>{formatStatistic(stats.percentile_25)}</td>
                                <td>{formatStatistic(stats.median)}</td>
                                <td>{formatStatistic(stats.percentile_75)}</td>
                                <td>{formatStatistic(stats.max)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </details>

              <section
                className="objective-panel"
                id="experiment-objective-panel"
                aria-labelledby="objective-title"
              >
                <div className="section-heading">
                  <div>
                    <p className="section-kicker">Define objective</p>
                    <h3 id="objective-title">
                      What should AutoDS-Agent predict?
                    </h3>
                  </div>
                  <span className="limit-note">
                    Deterministic target analysis
                  </span>
                </div>

                <form className="objective-form" onSubmit={handleObjective}>
                  <label htmlFor="objective">Natural-language objective</label>
                  <textarea
                    id="objective"
                    value={objective}
                    rows="4"
                    maxLength="4000"
                    placeholder="Predict which customers are likely to churn."
                    onChange={(event) => {
                      setObjective(event.target.value);
                      setAnalysis(null);
                      setConfirmedExperiment(null);
                      setExperimentError("");
                      setPipelinePlan(null);
                      setPlanningError("");
                      setTrainingResult(null);
                      setTrainingError("");
                    }}
                  />
                  <button
                    type="submit"
                    disabled={isAnalyzing || !objective.trim()}
                  >
                    {isAnalyzing ? "Analyzing objective…" : "Analyze objective"}
                  </button>
                </form>

                {analysis && (
                  <div className="confirmation-block">
                    <div className="suggestion-grid">
                      <article>
                        <span>Suggested task</span>
                        <strong>
                          {taskLabel(analysis.suggested_task_type)}
                        </strong>
                      </article>
                      <article>
                        <span>Suggested target</span>
                        <strong>
                          {analysis.suggested_target_column ||
                            "Select manually"}
                        </strong>
                      </article>
                      <article>
                        <span>Confidence</span>
                        <strong>{analysis.confidence}</strong>
                      </article>
                    </div>

                    {analysis.ambiguity_reason && (
                      <p className="ambiguity-notice" role="status">
                        {analysis.ambiguity_reason}
                      </p>
                    )}

                    <form
                      className="confirmation-form"
                      onSubmit={handleConfirmation}
                    >
                      <label>
                        Target column
                        <select
                          value={selectedTarget}
                          onChange={handleTargetChange}
                          disabled={[
                            "clustering",
                            "anomaly_detection",
                          ].includes(selectedTask)}
                        >
                          <option value="">Choose a dataset column</option>
                          {profile.columns.map((column) => (
                            <option key={column.name} value={column.name}>
                              {column.name}
                              {column.is_constant ? " — constant" : ""}
                              {column.is_possible_id ? " — possible ID" : ""}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Task type
                        <select
                          value={selectedTask}
                          onChange={(event) => {
                            setSelectedTask(event.target.value);
                            setConfirmedExperiment(null);
                            setPipelinePlan(null);
                            setPlanningError("");
                            setTrainingResult(null);
                            setTrainingError("");
                          }}
                        >
                          <option value="">Choose a task type</option>
                          {TASK_OPTIONS.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <button
                        type="submit"
                        disabled={
                          isConfirming ||
                          !selectedTask ||
                          (!["clustering", "anomaly_detection"].includes(
                            selectedTask,
                          ) &&
                            !selectedTarget)
                        }
                      >
                        {isConfirming ? "Confirming…" : "Confirm objective"}
                      </button>
                    </form>
                    {clustering && (
                      <div className="confirmed-card">
                        <p className="section-kicker">
                          Trusted clustering result
                        </p>
                        <h3>
                          {clustering.selected_cluster_count} clusters selected
                        </h3>
                        <p>
                          Silhouette score:{" "}
                          {formatStatistic(clustering.silhouette_score)} ·
                          Features used: {clustering.feature_count}
                        </p>
                        <p>
                          Excluded identifiers:{" "}
                          {clustering.excluded_columns.join(", ") || "None"}
                        </p>
                        <ul>
                          {clustering.cluster_summaries.map((item) => (
                            <li key={item.cluster}>
                              Cluster {item.cluster}: {item.row_count} rows (
                              {formatStatistic(item.percentage)}%)
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {specializedResult?.anomaly_count !== undefined && (
                      <div className="confirmed-card">
                        <p className="section-kicker">Trusted anomaly result</p>
                        <h3>
                          {specializedResult.anomaly_count} unusual observations
                        </h3>
                        <p>
                          {formatStatistic(
                            specializedResult.anomaly_percentage,
                          )}
                          % of {specializedResult.analyzed_rows} rows ·{" "}
                          {displayLabel(specializedResult.algorithm)}
                        </p>
                        <p className="limit-note">
                          These are anomalous observations, not verified fraud
                          labels.
                        </p>
                      </div>
                    )}
                    {specializedResult?.metrics && (
                      <div className="confirmed-card">
                        <p className="section-kicker">
                          Chronological forecast evaluation
                        </p>
                        <h3>{displayLabel(specializedResult.algorithm)}</h3>
                        <p>
                          MAE {formatStatistic(specializedResult.metrics.mae)} ·
                          RMSE {formatStatistic(specializedResult.metrics.rmse)}
                          {specializedResult.metrics.mape !== null
                            ? ` · MAPE ${formatStatistic(specializedResult.metrics.mape)}%`
                            : ""}
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {experimentError && (
                  <p className="form-error" role="alert">
                    {experimentError}
                  </p>
                )}

                {confirmedExperiment && (
                  <div className="confirmed-card" role="status">
                    <p className="section-kicker">Objective confirmed</p>
                    <dl>
                      <div>
                        <dt>Task</dt>
                        <dd>
                          {taskLabel(confirmedExperiment.confirmed_task_type)}
                        </dd>
                      </div>
                      <div>
                        <dt>Target</dt>
                        <dd>{confirmedExperiment.confirmed_target_column}</dd>
                      </div>
                      <div>
                        <dt>Status</dt>
                        <dd>Ready for planning</dd>
                      </div>
                    </dl>
                    <button
                      type="button"
                      onClick={handlePlanning}
                      disabled={isPlanning}
                    >
                      {isPlanning ? "Planning…" : "Generate pipeline plan"}
                    </button>
                  </div>
                )}

                {planningError && (
                  <p className="form-error" role="alert">
                    Planning failed: {planningError}
                  </p>
                )}

                {pipelinePlan && (
                  <div className="plan-card" aria-labelledby="plan-title">
                    <div className="section-heading">
                      <div>
                        <p className="section-kicker">Plan generated</p>
                        <h3 id="plan-title">Pipeline plan</h3>
                      </div>
                      <span className="provider-badge">
                        {pipelinePlan.provider_used === "ollama"
                          ? "Fallback: Ollama"
                          : "Gemini"}
                      </span>
                    </div>
                    <details className="advanced-details">
                      <summary>
                        Advanced details: preprocessing and model plan
                      </summary>
                      <dl className="plan-grid">
                        <div>
                          <dt>Task</dt>
                          <dd>{displayLabel(pipelinePlan.plan.task_type)}</dd>
                        </div>
                        <div>
                          <dt>Target</dt>
                          <dd>{pipelinePlan.plan.target_column}</dd>
                        </div>
                        <div>
                          <dt>Numeric imputation</dt>
                          <dd>
                            {displayLabel(pipelinePlan.plan.numeric_imputation)}
                          </dd>
                        </div>
                        <div>
                          <dt>Categorical imputation</dt>
                          <dd>
                            {displayLabel(
                              pipelinePlan.plan.categorical_imputation,
                            )}
                          </dd>
                        </div>
                        <div>
                          <dt>Encoding</dt>
                          <dd>
                            {displayLabel(
                              pipelinePlan.plan.categorical_encoding,
                            )}
                          </dd>
                        </div>
                        <div>
                          <dt>Scaling</dt>
                          <dd>
                            {displayLabel(pipelinePlan.plan.numeric_scaling)}
                          </dd>
                        </div>
                        <div>
                          <dt>Primary metric</dt>
                          <dd>
                            {displayLabel(pipelinePlan.plan.primary_metric)}
                          </dd>
                        </div>
                        <div className="model-list">
                          <dt>Baseline models</dt>
                          <dd>
                            {pipelinePlan.plan.models
                              .map(displayLabel)
                              .join(", ")}
                          </dd>
                        </div>
                      </dl>
                    </details>
                    {pipelinePlan.model_recommendations?.length > 0 && (
                      <section className="adaptive-decision-list">
                        <h4>Why These Models?</h4>
                        <p className="limit-note">
                          Candidates are task-compatible; final selection
                          remains based only on the configured primary
                          validation metric.
                        </p>
                        {pipelinePlan.model_recommendations.map((item) => (
                          <article
                            className="analysis-card"
                            key={item.model_name}
                          >
                            <strong>{displayLabel(item.model_name)}</strong>
                            <p>{item.reason}</p>
                            <small>
                              Dataset evidence: {item.dataset_evidence}
                            </small>
                          </article>
                        ))}
                      </section>
                    )}
                    {pipelinePlan.adaptive_decisions?.length > 0 && (
                      <section className="adaptive-decision-list">
                        <h4>Adaptive Pipeline Decisions</h4>
                        {pipelinePlan.adaptive_decisions.map((item, index) => (
                          <article
                            className="analysis-card"
                            key={`${item.decision}-${index}`}
                          >
                            <strong>{item.decision}</strong>
                            <p>{item.reason}</p>
                            <small>Evidence: {item.evidence}</small>
                            <p>
                              <strong>Action taken:</strong> {item.action_taken}{" "}
                              {item.applied ? "· Applied" : "· Advisory only"}
                            </p>
                          </article>
                        ))}
                      </section>
                    )}
                    {(pipelinePlan.plan.task_type === "binary_classification" ||
                      pipelinePlan.plan.task_type ===
                        "multiclass_classification") &&
                      pipelinePlan.plan.models.some((name) =>
                        ["logistic_regression", "random_forest"].includes(name),
                      ) && (
                        <div className="setting-toggle">
                          <span>
                            <strong>Manual override: class weighting</strong>
                            <small>
                              Toggle balanced class weights for supported
                              classifiers. Locked after training begins.
                            </small>
                          </span>
                          <div>
                            {pipelinePlan.plan.models
                              .filter((name) =>
                                [
                                  "logistic_regression",
                                  "random_forest",
                                ].includes(name),
                              )
                              .map((name) => (
                                <label key={name}>
                                  <input
                                    type="checkbox"
                                    checked={(
                                      pipelinePlan.plan
                                        .balanced_class_weight_models || []
                                    ).includes(name)}
                                    onChange={(event) =>
                                      handleAdaptiveWeightOverride(
                                        name,
                                        event.target.checked,
                                      )
                                    }
                                    disabled={
                                      isUpdatingAdaptive ||
                                      isTraining ||
                                      Boolean(trainingResult)
                                    }
                                  />{" "}
                                  {displayLabel(name)}
                                </label>
                              ))}
                          </div>
                          {isUpdatingAdaptive && (
                            <small>Saving override…</small>
                          )}
                        </div>
                      )}
                    {modelIsTraining && (
                      <p className="training-progress" role="status">
                        <span className="loading-spinner" aria-hidden="true" />
                        Models are training. Please wait…
                      </p>
                    )}
                    <button
                      type="button"
                      onClick={handleTraining}
                      disabled={modelIsTraining}
                    >
                      {modelIsTraining
                        ? "Models are training…"
                        : "Run experiment"}
                    </button>
                  </div>
                )}

                {trainingError && (
                  <p className="form-error" role="alert">
                    Training failed: {trainingError}
                  </p>
                )}

                {trainingResult && (
                  <div
                    className="training-card"
                    aria-labelledby="training-title"
                  >
                    <div className="section-heading">
                      <div>
                        <p className="section-kicker">Baseline training</p>
                        <h3 id="training-title">Model runs</h3>
                      </div>
                      <span className="success-badge">
                        {displayLabel(trainingResult.status)}
                      </span>
                    </div>
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Model</th>
                            <th>Status</th>
                            <th>Training duration</th>
                          </tr>
                        </thead>
                        <tbody>
                          {trainingResult.model_runs.map((run) => (
                            <tr key={run.model_run_id}>
                              <td>{displayLabel(run.model_name)}</td>
                              <td>{displayLabel(run.status)}</td>
                              <td>
                                {formatStatistic(run.training_duration_seconds)}{" "}
                                seconds
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="training-disclaimer">
                      Baselines are fitted on training data. Compare them only
                      on the validation split.
                    </p>
                    <button
                      type="button"
                      onClick={handleEvaluation}
                      disabled={workflowAction === "evaluate"}
                    >
                      {workflowAction === "evaluate"
                        ? "Evaluating validation results…"
                        : "Evaluate models"}
                    </button>
                  </div>
                )}

                {workflowError && (
                  <p className="form-error" role="alert">
                    {workflowError}
                  </p>
                )}

                {evaluation && (
                  <div
                    className="evaluation-card"
                    aria-labelledby="evaluation-title"
                  >
                    <p className="section-kicker">Results</p>
                    <h3 id="evaluation-title">
                      Best model:{" "}
                      {evaluation.selected_model_name
                        ? displayLabel(evaluation.selected_model_name)
                        : displayLabel(selectedValidation?.model_name)}
                    </h3>
                    <div className="result-summary-grid">
                      <article>
                        <span>Why selected</span>
                        <strong>
                          Best validation{" "}
                          {displayLabel(evaluation.primary_metric)}
                        </strong>
                        <small>
                          {formatMetric(
                            evaluation.primary_metric,
                            validationPrimaryMetric,
                          )}
                        </small>
                      </article>
                      <article>
                        <span>
                          Validation {displayLabel(evaluation.primary_metric)}
                        </span>
                        <strong>
                          {formatMetric(
                            evaluation.primary_metric,
                            validationPrimaryMetric,
                          )}
                        </strong>
                        <small>
                          {primaryMetricIsHigherBetter
                            ? "Higher is better"
                            : "Lower is better"}
                        </small>
                      </article>
                      <article>
                        <span>
                          Final test {displayLabel(evaluation.primary_metric)}
                        </span>
                        <strong>
                          {formatMetric(
                            evaluation.primary_metric,
                            testPrimaryMetric,
                          )}
                        </strong>
                        <small>Untouched test split</small>
                      </article>
                      <article>
                        <span>Validation vs test difference</span>
                        <strong>
                          {generalizationDifference === null
                            ? "—"
                            : formatMetric(
                                evaluation.primary_metric,
                                Math.abs(generalizationDifference),
                              )}
                        </strong>
                        <small>
                          {generalizationPercentage === null
                            ? "—"
                            : `${generalizationPercentage >= 0 ? "+" : ""}${generalizationPercentage.toFixed(1)}% test change`}
                        </small>
                      </article>
                      {visibleMetricNames
                        .filter(
                          (metric) => metric !== evaluation.primary_metric,
                        )
                        .map((metric) => (
                          <article key={metric}>
                            <span>Final test {displayLabel(metric)}</span>
                            <strong>
                              {formatMetric(
                                metric,
                                evaluation.final_test_metrics[metric],
                              )}
                            </strong>
                            <small>Untouched test split</small>
                          </article>
                        ))}
                    </div>
                    <p className="training-disclaimer">
                      {isRegression
                        ? `Validation RMSE and final test RMSE differ by ${generalizationDifference === null ? "—" : formatMetric("rmse", Math.abs(generalizationDifference))} (${generalizationPercentage === null ? "—" : `${Math.abs(generalizationPercentage).toFixed(1)}%`}). This is a verified generalization comparison, not an automatic overfitting diagnosis.`
                        : "The selected model was chosen using validation only; final classification metrics come from the untouched test split."}
                    </p>
                    <h4>Model Benchmark</h4>
                    <div className="comparison-chart">
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart data={comparisonChartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="model" />
                          <YAxis />
                          <Tooltip
                            formatter={(value) =>
                              formatMetric(evaluation.primary_metric, value)
                            }
                          />
                          <Bar
                            dataKey="score"
                            radius={[5, 5, 0, 0]}
                            fill="#1687d4"
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    {performanceCostData.length > 1 && (
                      <>
                        <h4>Performance vs Training Cost</h4>
                        <p className="limit-note">
                          Measured training wall time vs. the configured
                          validation metric. No composite score is applied.
                        </p>
                        <div className="comparison-chart">
                          <ResponsiveContainer width="100%" height={260}>
                            <ScatterChart>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis
                                type="number"
                                dataKey="training_seconds"
                                name="Training time"
                                unit=" s"
                              />
                              <YAxis
                                type="number"
                                dataKey="validation_score"
                                name={displayLabel(evaluation.primary_metric)}
                              />
                              <Tooltip
                                cursor={{ strokeDasharray: "3 3" }}
                                formatter={(value, name) =>
                                  name === "validation_score"
                                    ? formatMetric(
                                        evaluation.primary_metric,
                                        value,
                                      )
                                    : `${formatStatistic(value)} seconds`
                                }
                                labelFormatter={(_, payload) =>
                                  payload?.[0]?.payload?.model || ""
                                }
                              />
                              <Scatter
                                name="Model runs"
                                data={performanceCostData}
                                fill="#1687d4"
                              />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </div>
                      </>
                    )}
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Model</th>
                            <th>
                              Validation{" "}
                              {displayLabel(evaluation.primary_metric)}
                            </th>
                            <th>Train time</th>
                            <th>Inference / row</th>
                            <th>Complexity</th>
                            <th>Generalization</th>
                          </tr>
                        </thead>
                        <tbody>
                          {evaluation.validation_comparison.map((item) => (
                            <tr
                              key={item.model_run_id}
                              className={
                                item.model_run_id ===
                                evaluation.selected_model_run_id
                                  ? "selected-row"
                                  : ""
                              }
                            >
                              <td>
                                {displayLabel(item.model_name)}
                                {item.model_run_id ===
                                evaluation.selected_model_run_id
                                  ? " — selected"
                                  : ""}
                              </td>
                              <td>
                                {formatMetric(
                                  evaluation.primary_metric,
                                  item.metrics[evaluation.primary_metric],
                                )}
                              </td>
                              <td>
                                {item.training_seconds == null
                                  ? "—"
                                  : `${formatStatistic(item.training_seconds)} s`}
                              </td>
                              <td>
                                {item.inference_seconds_per_row == null
                                  ? "—"
                                  : `${formatStatistic(item.inference_seconds_per_row * 1000)} ms`}
                              </td>
                              <td>{item.complexity_summary || "—"}</td>
                              <td>
                                {item.generalization_change == null
                                  ? item.generalization_status ||
                                    "Not available"
                                  : `${formatMetric(evaluation.primary_metric, item.generalization_change)} test − validation`}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {selectedValidation && (
                      <section className="analysis-card">
                        <h4>Why This Model Was Selected</h4>
                        <p>
                          <strong>
                            {displayLabel(evaluation.selected_model_name)}
                          </strong>{" "}
                          was selected by the configured primary validation
                          metric ({displayLabel(evaluation.primary_metric)}):{" "}
                          {formatMetric(
                            evaluation.primary_metric,
                            validationPrimaryMetric,
                          )}
                          . The other candidates remain available in the
                          benchmark above; no arbitrary weighted score was used.
                        </p>
                        <p>
                          Training took{" "}
                          {formatStatistic(selectedValidation.training_seconds)}{" "}
                          seconds; measured inference was{" "}
                          {selectedValidation.inference_seconds_per_row == null
                            ? "unavailable"
                            : `${formatStatistic(selectedValidation.inference_seconds_per_row * 1000)} ms per row`}
                          . Its final test-minus-validation change was{" "}
                          {selectedValidation.generalization_change == null
                            ? "not available"
                            : formatMetric(
                                evaluation.primary_metric,
                                selectedValidation.generalization_change,
                              )}
                          . This generalization evidence did not affect
                          selection.
                        </p>
                        <small>
                          Training time and model structure are local
                          measurements/proxies, not monetary cost estimates.
                        </small>
                      </section>
                    )}
                    {evaluation.final_test_confusion_matrix && (
                      <div className="table-block">
                        <h3>Final test confusion matrix</h3>
                        <div className="table-scroll">
                          <table>
                            <thead>
                              <tr>
                                <th>Actual \ Predicted</th>
                                {evaluation.final_test_confusion_matrix.labels.map(
                                  (label) => (
                                    <th key={label}>{label}</th>
                                  ),
                                )}
                              </tr>
                            </thead>
                            <tbody>
                              {evaluation.final_test_confusion_matrix.matrix.map(
                                (row, index) => (
                                  <tr
                                    key={
                                      evaluation.final_test_confusion_matrix
                                        .labels[index]
                                    }
                                  >
                                    <td>
                                      {
                                        evaluation.final_test_confusion_matrix
                                          .labels[index]
                                      }
                                    </td>
                                    {row.map((value, cell) => (
                                      <td key={cell}>{value}</td>
                                    ))}
                                  </tr>
                                ),
                              )}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                    <button
                      type="button"
                      onClick={handleReflection}
                      disabled={workflowAction === "reflection"}
                    >
                      {workflowAction === "reflection"
                        ? "Generating AI review…"
                        : "Generate AI Review"}
                    </button>
                    {reflection && (
                      <div className="assistant-answer">
                        <p className="section-kicker">AI experiment review</p>
                        <h3>
                          Best Model:{" "}
                          {evaluation.selected_model_name
                            ? displayLabel(evaluation.selected_model_name)
                            : "Selected baseline"}
                        </h3>
                        <div className="verified-review">
                          <h4>Verified evidence</h4>
                          <ul>
                            <li>
                              Validation{" "}
                              {displayLabel(evaluation.primary_metric)}:{" "}
                              {formatMetric(
                                evaluation.primary_metric,
                                selectedValidation?.metrics?.[
                                  evaluation.primary_metric
                                ],
                              )}
                              ; final test:{" "}
                              {formatMetric(
                                evaluation.primary_metric,
                                evaluation.final_test_metrics[
                                  evaluation.primary_metric
                                ],
                              )}
                              .
                            </li>
                            <li>
                              {isRegression
                                ? `Final MAE ${formatMetric("mae", evaluation.final_test_metrics.mae)}, RMSE ${formatMetric("rmse", evaluation.final_test_metrics.rmse)}, and R² ${formatMetric("r2", evaluation.final_test_metrics.r2)}.`
                                : `Final test metrics are calculated on an untouched split after validation-based model selection.`}
                            </li>
                            <li>
                              Limitation: this evaluation uses one persisted
                              train/validation/test split.
                            </li>
                            {explainability?.global_feature_importance
                              ?.slice(0, 3)
                              .map((item) => (
                                <li key={item.feature_name}>
                                  Important feature: {item.feature_name} (
                                  {formatStatistic(item.importance)}{" "}
                                  importance).
                                </li>
                              ))}
                          </ul>
                        </div>
                        <p>
                          <strong>Overall Assessment</strong>
                          <br />
                          {reflection.model_assessment}
                        </p>
                        {reflection.strengths.length > 0 && (
                          <>
                            <h4>Strengths</h4>
                            <ul>
                              {reflection.strengths.map((item) => (
                                <li key={item}>{item}</li>
                              ))}
                            </ul>
                          </>
                        )}
                        {reflection.weaknesses.length > 0 && (
                          <>
                            <h4>Possible limitations</h4>
                            <ul>
                              {reflection.weaknesses.map((item) => (
                                <li key={item}>{item}</li>
                              ))}
                            </ul>
                          </>
                        )}
                        {reflection.important_features.length > 0 && (
                          <>
                            <h4>Important Features</h4>
                            <ul>
                              {reflection.important_features.map((item) => (
                                <li key={item.feature}>
                                  <strong>{item.feature}:</strong>{" "}
                                  {item.explanation}
                                </li>
                              ))}
                            </ul>
                          </>
                        )}
                        {reflection.decision_explanations?.length > 0 && (
                          <>
                            <h4>Why the workflow made these decisions</h4>
                            {reflection.decision_explanations.map((item) => (
                              <article
                                className="analysis-card"
                                key={`${item.stage}-${item.decision}`}
                              >
                                <strong>
                                  {item.stage}: {item.decision}
                                </strong>
                                <p>{item.rationale}</p>
                                <small>Evidence: {item.evidence_source}</small>
                              </article>
                            ))}
                          </>
                        )}
                        {reflection.similar_experiments?.length > 0 && (
                          <>
                            <h4>Related verified experiments</h4>
                            {reflection.similar_experiments.map((item) => (
                              <article
                                className="analysis-card"
                                key={item.experiment_id}
                              >
                                <strong>
                                  {item.dataset_filename ||
                                    "Previous experiment"}{" "}
                                  · {displayLabel(item.selected_model || "")}
                                </strong>
                                <p>
                                  {item.match_reason}. {item.objective}
                                </p>
                                <p>
                                  {displayLabel(item.task_type || "")} · target{" "}
                                  {item.target_column || "—"} ·{" "}
                                  {displayLabel(item.primary_metric || "")}:
                                  validation{" "}
                                  {formatMetric(
                                    item.primary_metric,
                                    item.validation_score,
                                  )}
                                  , test{" "}
                                  {formatMetric(
                                    item.primary_metric,
                                    item.test_score,
                                  )}
                                </p>
                                <small>
                                  Historical context only; not a performance
                                  guarantee.
                                </small>
                              </article>
                            ))}
                          </>
                        )}
                        {reflection.research_evidence?.length > 0 ? (
                          <>
                            <h4>Curated research (advisory)</h4>
                            {reflection.research_evidence.map((item) => (
                              <article
                                className="analysis-card"
                                key={item.source_url}
                              >
                                <strong>{item.title}</strong>
                                <p>{item.evidence_summary}</p>
                                <small>
                                  {item.venue} · {item.year} ·{" "}
                                  <a
                                    href={item.source_url}
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    Source paper
                                  </a>
                                </small>
                              </article>
                            ))}
                          </>
                        ) : (
                          <p className="limit-note">
                            No relevant curated research evidence was found for
                            this experiment. No citations were generated.
                          </p>
                        )}
                        {reflection.recommendations.length > 0 && (
                          <>
                            <h4>Next recommendations</h4>
                            {reflection.recommendations
                              .slice(0, 3)
                              .map((item) => (
                                <article
                                  className="analysis-card"
                                  key={item.title}
                                >
                                  <strong>{item.title}</strong>
                                  <p>{item.reason}</p>
                                  <p>{item.suggested_action}</p>
                                  <small>
                                    Requires retraining:{" "}
                                    {item.requires_retraining ? "Yes" : "No"} —
                                    advisory only
                                  </small>
                                </article>
                              ))}
                          </>
                        )}
                        <div className="assistant-meta">
                          <span>Grounded in verified AutoDS results</span>
                          <span className="provider-badge">
                            {displayLabel(reflection.provider_used)}
                          </span>
                        </div>
                      </div>
                    )}
                    {reflection?.recommendations?.some(
                      (item) => item.research_sources?.length,
                    ) && (
                      <div className="analysis-card">
                        <strong>Research linked to recommendations</strong>
                        {reflection.recommendations
                          .flatMap((item) =>
                            (item.research_sources || []).map((source) => ({
                              ...source,
                              recommendation: item.title,
                            })),
                          )
                          .map((source) => (
                            <p
                              key={`${source.recommendation}-${source.source_url}`}
                            >
                              {source.recommendation}:{" "}
                              <a
                                href={source.source_url}
                                target="_blank"
                                rel="noreferrer"
                              >
                                {source.title}
                                {source.year ? ` (${source.year})` : ""}
                              </a>
                            </p>
                          ))}
                      </div>
                    )}
                    <div className="completion-actions">
                      <button
                        type="button"
                        onClick={handleOptimization}
                        disabled={workflowAction === "optimize"}
                      >
                        Optimize Model
                      </button>
                      <button
                        type="button"
                        onClick={handleExplainability}
                        disabled={workflowAction === "explainability"}
                      >
                        {workflowAction === "explainability"
                          ? "Explaining…"
                          : "Explain Model"}
                      </button>
                      <button
                        type="button"
                        onClick={handleReport}
                        disabled={workflowAction === "report"}
                      >
                        Generate Report
                      </button>
                      <button
                        type="button"
                        className="stage-next"
                        onClick={() => openStage("ask")}
                      >
                        Ask AutoDS
                      </button>
                    </div>
                    <form
                      className="upload-form prediction-form"
                      onSubmit={handlePredictions}
                    >
                      <label className="file-picker">
                        <span>
                          {predictionFile
                            ? predictionFile.name
                            : "Choose a prediction CSV"}
                        </span>
                        <input
                          type="file"
                          accept=".csv,text/csv"
                          onChange={(event) =>
                            setPredictionFile(event.target.files?.[0] || null)
                          }
                        />
                      </label>
                      <button
                        type="submit"
                        disabled={
                          !predictionFile || workflowAction === "predictions"
                        }
                      >
                        {workflowAction === "predictions"
                          ? "Predicting…"
                          : "Create Predictions"}
                      </button>
                    </form>
                    {prediction && (
                      <p>
                        <button
                          type="button"
                          onClick={() =>
                            handleArtifactDownload(
                              "prediction",
                              prediction,
                              "predictions.csv",
                            )
                          }
                          disabled={Boolean(downloadAction)}
                        >
                          Download prediction CSV ({prediction.row_count} rows)
                        </button>
                      </p>
                    )}
                  </div>
                )}

                <ReportGenerationPanel />

                {optimization && (
                  <div className="evaluation-card">
                    <p className="section-kicker">Optimization</p>
                    <h3>{displayLabel(optimization.status)}</h3>
                    <div className="result-summary-grid">
                      <article>
                        <span>
                          Before {displayLabel(evaluation?.primary_metric)}
                        </span>
                        <strong>
                          {formatMetric(
                            evaluation?.primary_metric,
                            optimizationBaseline,
                          )}
                        </strong>
                      </article>
                      <article>
                        <span>
                          After {displayLabel(evaluation?.primary_metric)}
                        </span>
                        <strong>
                          {formatMetric(
                            evaluation?.primary_metric,
                            optimization.best_validation_score,
                          )}
                        </strong>
                      </article>
                      <article>
                        <span>Absolute improvement</span>
                        <strong>
                          {optimizationImprovement === null
                            ? "—"
                            : formatMetric(
                                evaluation?.primary_metric,
                                optimizationImprovement,
                              )}
                        </strong>
                        <small>
                          {primaryMetricIsHigherBetter
                            ? "Higher is better"
                            : "Lower is better"}
                        </small>
                      </article>
                      <article>
                        <span>Percentage improvement</span>
                        <strong>
                          {optimizationImprovement === null ||
                          !optimizationBaseline
                            ? "—"
                            : `${((optimizationImprovement / Math.abs(optimizationBaseline)) * 100).toFixed(1)}%`}
                        </strong>
                      </article>
                    </div>
                    {optimization.cost_profile && (
                      <div className="verified-review">
                        <h4>Compute cost profile</h4>
                        <p>
                          Search used a{" "}
                          {optimization.cost_profile.budget_seconds ??
                            "bounded"}{" "}
                          second time budget; average completed trial:{" "}
                          {formatStatistic(
                            optimization.cost_profile.mean_trial_seconds,
                          )}{" "}
                          seconds.
                        </p>
                        {optimization.cost_profile.fastest_near_optimal && (
                          <p>
                            Fastest candidate within 1% relative validation
                            score:{" "}
                            {formatStatistic(
                              optimization.cost_profile.fastest_near_optimal
                                .duration_seconds,
                            )}{" "}
                            seconds. This is a cost/quality comparison only; the
                            trained model remains the validation-quality-best
                            candidate.
                          </p>
                        )}
                        <small>
                          Measured wall-clock time; no monetary cost or
                          hardware-normalized estimate.
                        </small>
                      </div>
                    )}
                    <details className="advanced-details">
                      <summary>Advanced details: best parameters</summary>
                      <p>
                        {displayLabel(optimization.model_name)} ·{" "}
                        {optimization.trial_count} trials
                      </p>
                      <pre>
                        {JSON.stringify(optimization.best_parameters, null, 2)}
                      </pre>
                    </details>
                  </div>
                )}

                {false && evaluation && (
                  <div className="evaluation-card" id="reports">
                    <p className="section-kicker">
                      Step 4 — Reports and downloads
                    </p>
                    <h3>Explain, predict, and report</h3>
                    <button
                      type="button"
                      onClick={handleExplainability}
                      disabled={workflowAction === "explainability"}
                    >
                      Show feature importance
                    </button>
                    {explainability && (
                      <ol className="importance-list">
                        {explainability.global_feature_importance
                          .slice(0, 12)
                          .map((item) => (
                            <li key={item.feature_name}>
                              {item.feature_name}:{" "}
                              {formatStatistic(item.importance)}{" "}
                              {item.direction ? `(${item.direction})` : ""}
                            </li>
                          ))}
                      </ol>
                    )}
                    <form
                      className="upload-form prediction-form"
                      onSubmit={handlePredictions}
                    >
                      <label className="file-picker">
                        <span>
                          {predictionFile
                            ? predictionFile.name
                            : "Choose prediction CSV"}
                        </span>
                        <input
                          type="file"
                          accept=".csv,text/csv"
                          onChange={(event) =>
                            setPredictionFile(event.target.files?.[0] || null)
                          }
                        />
                      </label>
                      <button
                        type="submit"
                        disabled={
                          !predictionFile || workflowAction === "predictions"
                        }
                      >
                        {workflowAction === "predictions"
                          ? "Predicting…"
                          : "Create predictions"}
                      </button>
                    </form>
                    {prediction && (
                      <p>
                        <button
                          type="button"
                          onClick={() =>
                            handleArtifactDownload(
                              "prediction",
                              prediction,
                              "predictions.csv",
                            )
                          }
                          disabled={Boolean(downloadAction)}
                        >
                          {downloadAction === "prediction"
                            ? "Downloading…"
                            : `Download prediction CSV (${prediction.row_count} rows)`}
                        </button>
                      </p>
                    )}
                    <button
                      type="button"
                      onClick={handleReport}
                      disabled={workflowAction === "report"}
                    >
                      {workflowAction === "report"
                        ? "Generating report…"
                        : "Generate HTML report"}
                    </button>
                    {report && (
                      <p>
                        <button
                          type="button"
                          onClick={() =>
                            handleArtifactDownload(
                              "html-report",
                              report,
                              "autods-agent-report.html",
                            )
                          }
                          disabled={Boolean(downloadAction)}
                        >
                          {downloadAction === "html-report"
                            ? "Downloading…"
                            : "Download HTML report"}
                        </button>
                      </p>
                    )}
                    <button
                      type="button"
                      onClick={handlePdfReport}
                      disabled={workflowAction === "PDF report"}
                    >
                      {workflowAction === "PDF report"
                        ? "Generating PDF…"
                        : "Generate PDF report"}
                    </button>
                    {pdfReport && (
                      <p>
                        <button
                          type="button"
                          onClick={() =>
                            handleArtifactDownload(
                              "pdf-report",
                              pdfReport,
                              "autods-agent-report.pdf",
                            )
                          }
                          disabled={Boolean(downloadAction)}
                        >
                          {downloadAction === "pdf-report"
                            ? "Downloading…"
                            : "Download PDF report"}
                        </button>
                      </p>
                    )}
                    {downloadError && (
                      <p className="form-error" role="alert">
                        {downloadError}
                      </p>
                    )}
                  </div>
                )}
              </section>
            </section>
          )}
          {activeStage === "reports" && <ReportsPage />}
        </section>
      </div>
    </main>
  );
}

export default App;
