"use client";

import dynamic from "next/dynamic";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), {
  ssr: false,
});

type Phase = "welcome" | "thinking" | "planning" | "interview" | "complete";

type PlanSection = {
  id: string;
  title: string;
  description: string;
};

type InterviewContext = {
  plan: PlanSection[];
  coding_question: string;
};

type BackendFeedback = {
  highlights: string[];
  recommendations: string[];
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type PendingAction = {
  type: "plan";
  iteration: number;
  feedback?: string;
};

type InterviewUtterance = {
  id: string;
  speaker: "interviewer" | "candidate";
  content: string;
  timestamp: string;
};

type InterviewReport = {
  overview: string;
  highlights: string[];
  recommendations: string[];
  nextSteps: string;
};

const PLANNER_URL = "/generate_plan";

const DEFAULT_EDITOR_VALUE =
  "# Start coding here. Narrate your thought process as you go.\n";

// #TODO - CODEX SUGGESTION: Remove this artificial delay once the planner response arrives directly from the backend.

const createId = (prefix: string) =>
  `${prefix}-${Math.random().toString(36).slice(2, 10)}`;

const timestamp = () =>
  new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

function planMessage(
  plan: PlanSection[],
  iteration: number,
  feedback?: string,
  supplementalAvailable?: boolean,
  resumeName?: string,
) {
  const intro =
    iteration === 0
      ? "Here’s the interview outline I drafted:"
      : "Thanks for the direction—here’s the refreshed outline:";
  const bullets = plan
    .map((section, index) => `${index + 1}. ${section.title}`)
    .join("\n");
  const extras: string[] = [];
  if (resumeName) {
    extras.push(`Resume in scope: ${resumeName}`);
  }
  if (supplementalAvailable) {
    extras.push(
      "Supplemental prep notes loaded—they'll shape follow-ups without cluttering the outline.",
    );
  }
  if (feedback) {
    extras.push(`Incorporated feedback: ${feedback}`);
  }
  const extrasBlock = extras.length ? `\n\n${extras.join("\n")}` : "";
  return `${intro}\n\n${bullets}${extrasBlock}\n\nLet me know if you’d like to adjust anything else.`;
}

// The new async function that calls the backend
async function generateBackendReport(
  plan: PlanSection[],
  editorValue: string,
): Promise<BackendFeedback> {
  const payload: InterviewContext = {
    plan: plan,
    coding_question: editorValue, // Send the final code as the "question" for analysis
  };

  const response = await fetch("/generate_feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to generate feedback");
  }

  return await response.json();
}

function synthesizeMockReport(opts: {
  plan: PlanSection[];
  transcript: InterviewUtterance[];
  editorValue: string;
}): InterviewReport {
  const { plan, transcript, editorValue } = opts;
  const totalTurns = transcript.length;
  const interviewerTurns = transcript.filter(
    (entry) => entry.speaker === "interviewer",
  ).length;
  const candidateTurns = totalTurns - interviewerTurns;
  const codeLines = editorValue
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const technicalStage =
    plan.find((section) =>
      section.title.toLowerCase().includes("technical"),
    ) || plan[2];
  const resumeStage =
    plan.find((section) => section.title.toLowerCase().includes("resume")) ||
    plan.find((section) => section.id.includes("resume"));

  const highlights = [
    `Moved through ${plan.length} planned segments with ${interviewerTurns} interviewer prompts and ${candidateTurns} responses.`,
    codeLines.length
      ? `Authored ~${codeLines.length} lines in the workspace; opening focus: ${codeLines
        .slice(0, 2)
        .join(" ")
        .slice(0, 120)}${codeLines.length > 2 ? "…" : "."}`
      : "Relied primarily on verbal reasoning—no substantive code committed during this run.",
    `Kept alignment with the closing loop by acknowledging next steps before ending early.`,
  ];

  const recommendations = [
    technicalStage
      ? `During “${technicalStage.title}”, narrate explicit test cases and edge handling to mirror production readiness.`
      : "Narrate explicit test cases and edge handling to mirror production readiness.",
    resumeStage
      ? `For “${resumeStage.title}”, connect each highlight back to measurable impact and supporting anecdotes.`
      : "Connect resume highlights back to measurable impact and supporting anecdotes.",
  ];

  return {
    overview: `Session wrapped manually after ${totalTurns} conversational turns.`,
    highlights,
    recommendations,
    nextSteps:
      "Schedule another mock run or export this outline to Modal tooling to keep iterating with real voice IO.",
  };
}

export default function HomePage() {
  const [phase, setPhase] = useState<Phase>("welcome");
  const [pendingAction, setPendingAction] = useState<PendingAction | null>(
    null,
  );
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [resumeFileName, setResumeFileName] = useState<string>("");
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [supplementalInfo, setSupplementalInfo] = useState<string>("");
  const [planSections, setPlanSections] = useState<PlanSection[]>([]);
  const [codingQuestion, setCodingQuestion] = useState<string>("");
  const [planIteration, setPlanIteration] = useState<number>(0);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [feedbackDraft, setFeedbackDraft] = useState<string>("");
  const [editorValue, setEditorValue] =
    useState<string>(DEFAULT_EDITOR_VALUE); // Use the default value
  const [transcript, setTranscript] = useState<InterviewUtterance[]>([]);
  const [interviewReport, setInterviewReport] =
    useState<InterviewReport | null>(null);
  const conversationEndRef = useRef<HTMLDivElement | null>(null);


  useEffect(() => {
    if (phase === "planning") {
      conversationEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, phase]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const controller = new AbortController();
    const target = `${window.location.origin}/editor-state`;

    const syncEditorState = async () => {
      try {
        console.log("[editor-sync] sending update");
        await fetch(target, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ document: editorValue }),
          signal: controller.signal,
        });
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          console.error("[editor-sync] failed", error);
        }
      }
    };

    syncEditorState();

    return () => controller.abort();
  }, [editorValue]);

  const thinkingHeadline = useMemo(() => {
    if (!pendingAction) {
      return "Synthesizing the next step...";
    }
    return pendingAction.iteration === 0
      ? "Drafting your interview plan..."
      : "Reworking the interview plan...";
  }, [pendingAction]);

  const thinkingSubhead = useMemo(() => {
    if (!pendingAction?.feedback) {
      return "Pulling in role expectations, competencies, and your current skills.";
    }
    const clipped =
      pendingAction.feedback.length > 160
        ? `${pendingAction.feedback.slice(0, 157)}…`
        : pendingAction.feedback;
    return `Integrating your note: “${clipped}”`;
  }, [pendingAction]);

  // --- REPLACE THIS FUNCTION ---
  const handleWelcomeSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!resumeFile) { // Check for the File object
      setResumeError("Please upload your resume to continue.");
      return;
    }
    setResumeError(null);

    const summaryParts = [
      resumeFileName ? `Resume uploaded: ${resumeFileName}` : null,
      supplementalInfo.trim()
        ? `Supplemental context: ${supplementalInfo.trim()}`
        : null,
    ].filter(Boolean);

    const userMessage: ChatMessage = {
      id: createId("user"),
      role: "user",
      content:
        summaryParts.join("\n") ||
        "No extra materials—just eager to get started.",
    };

    const assistantAck: ChatMessage = {
      id: createId("assistant"),
      role: "assistant",
      content:
        "Great! Give me a few seconds to tailor the interview flow around what you shared.",
    };

    setMessages([userMessage, assistantAck]);
    const action: PendingAction = { type: "plan", iteration: 0 };
    setPendingAction(action);
    setPhase("thinking");

    // --- Start Backend Integration ---
    const formData = new FormData();
    formData.append("resume", resumeFile);
    formData.append("job_description", supplementalInfo.trim() || "No job description provided.");
    formData.append("feedback", ""); // No feedback on first run

    try {
      const response = await fetch(PLANNER_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to generate plan");
      }

      // Destructure the new object
      const context: InterviewContext = await response.json();
      const plan = context.plan;

      // Success: Update state with real plan from backend
      setPlanSections(plan);
      setCodingQuestion(context.coding_question);
      setPlanIteration(action.iteration);
      setMessages((prev) => [
        ...prev,
        {
          id: createId("assistant"),
          role: "assistant",
          content: planMessage(
            plan,
            action.iteration,
            undefined,
            supplementalInfo.trim().length > 0,
            resumeFileName,
          ),
        },
      ]);
      setPhase("planning");
      setPendingAction(null);
    } catch (error) {
      console.error("Failed to fetch plan:", error);
      const errorMessage =
        error instanceof Error ? error.message : "An unknown error occurred.";
      // Show error on the welcome screen
      setResumeError(
        `Error from backend: ${errorMessage}. Please try again.`,
      );
      // Revert to welcome screen
      setPhase("welcome");
      setMessages([]);
      setPendingAction(null);
    }
    // --- End Backend Integration ---
  };

  const handleAcceptPlan = () => {
    setInterviewReport(null);
    setMessages((prev) => [
      ...prev,
      {
        id: createId("assistant"),
        role: "assistant",
        content:
          "Amazing. I’ll launch the live interview workspace—feel free to narrate your thought process once you’re ready.",
      },
    ]);
    setEditorValue(codingQuestion);

    setTranscript([
      {
        id: createId("interviewer"),
        speaker: "interviewer",
        content:
          "Welcome to the coding canvas. I’ll listen for how you reason aloud, just like the Modal Quillman voice demo.",
        timestamp: timestamp(),
      },
    ]);

    setPhase("interview");
  };

  const handleManualPrompt = () => {
    // #TODO - CODEX SUGGESTION: Trigger the backend interviewer turn instead of mutating transcript locally.
    setTranscript((prev) => [
      ...prev,
      {
        id: createId("candidate"),
        speaker: "candidate",
        content: "Manual prompt sent—ready for the next question.",
        timestamp: timestamp(),
      },
      {
        id: createId("interviewer"),
        speaker: "interviewer",
        content:
          "Copy that. Walk me through what you’re thinking next when you’re ready.",
        timestamp: timestamp(),
      },
    ]);
  };

  const handleEndInterview = async () => {
    // #TODO - CODEX SUGGESTION: Persist the wrap-up through the backend and hydrate the report from its response.
    const closingExchange: InterviewUtterance[] = [
      {
        id: createId("candidate"),
        speaker: "candidate",
        content:
          "Thanks for the mock run—let’s end here and debrief with a written report.",
        timestamp: timestamp(),
      },
      {
        id: createId("interviewer"),
        speaker: "interviewer",
        content:
          "Understood. Summarizing strengths, coaching notes, and next steps now.",
        timestamp: timestamp(),
      },
    ];

    const updatedTranscript = [...transcript, ...closingExchange];
    setTranscript(updatedTranscript);
    try {
      // 1. Try to get the real feedback from the backend
      const feedback = await generateBackendReport(
        planSections,
        editorValue,
      );

      // 2. Success: Populate the report with real data
      setInterviewReport({
        overview: `Session complete. AI-generated feedback based on your code and our plan.`,
        highlights: feedback.highlights,
        recommendations: feedback.recommendations,
        nextSteps:
          "Review the feedback, then schedule another run or practice these concepts.",
      });
    } catch (error) {
      console.error("Failed to generate backend report, using mock:", error);

      // 3. Failure: Fall back to the original mock report
      setInterviewReport(
        synthesizeMockReport({
          plan: planSections,
          transcript: updatedTranscript,
          editorValue,
        }),
      );
    }

    setPhase("complete");
  };

  const resetExperience = () => {
    setPhase("welcome");
    setPendingAction(null);
    setResumeFile(null);
    setResumeFileName("");
    setResumeError(null);
    setSupplementalInfo("");
    setPlanSections([]);
    setPlanIteration(0);
    setCodingQuestion("");
    setMessages([]);
    setFeedbackDraft("");
    setEditorValue(DEFAULT_EDITOR_VALUE);
    setTranscript([]);
    setInterviewReport(null);
  };

  if (phase === "welcome") {
    return (
      <div className="flex min-h-screen items-center justify-center px-6 py-16 text-slate-100">
        <div className="w-full max-w-3xl rounded-3xl border border-white/10 bg-slate-900/60 p-10 shadow-2xl backdrop-blur">
          <header className="space-y-3">
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-emerald-400/80">
              Syntherview
            </p>
            <h1 className="text-4xl font-semibold tracking-tight">
              Prep with a voice-first AI interviewer
            </h1>
            <p className="text-sm text-slate-300">
              Upload your resume, add any job context, and let the system plan a
              tailored interview session.
            </p>
          </header>

          <form className="mt-10 space-y-8" onSubmit={handleWelcomeSubmit}>
            <div className="space-y-3">
              <label className="block text-sm font-medium text-slate-200">
                <span className="flex items-center gap-2">
                  <span>Resume or CV</span>
                  <span className="rounded-full border border-emerald-500/40 bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.25em] text-emerald-200">
                    Required
                  </span>
                </span>
              </label>
              <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-slate-700 bg-slate-900/60 px-6 py-8 text-center text-sm text-slate-300 transition hover:border-emerald-400/80 hover:text-emerald-200">
                <span>
                  Drag & drop or{" "}
                  <span className="font-semibold text-emerald-300">
                    browse files
                  </span>
                </span>
                <span className="text-xs text-slate-400">
                  PDF only — max 10MB
                </span>
                <input
                  type="file"
                  accept=".pdf"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    console.log("File selected:", file);
                    setResumeFile(file ? file : null);
                    setResumeFileName(file ? file.name : "");
                    setResumeError(
                      file ? null : "Please upload your resume to continue.",
                    );
                  }}
                />
              </label>
              {resumeFileName ? (
                <p className="rounded-xl border border-emerald-400/30 bg-emerald-500/10 px-4 py-2 text-xs text-emerald-200">
                  Detected: {resumeFileName}
                </p>
              ) : (
                <p className="text-xs text-slate-400">
                  Upload your resume to enable the interview planner.
                </p>
              )}
              {resumeError && (
                <p className="text-xs font-semibold text-rose-300">
                  {resumeError}
                </p>
              )}
            </div>

            <div className="space-y-3">
              <label
                htmlFor="supplemental"
                className="block text-sm font-medium text-slate-200"
              >
                Supplemental context
              </label>
              <textarea
                id="supplemental"
                value={supplementalInfo}
                onChange={(event) => setSupplementalInfo(event.target.value)}
                placeholder="Team culture, role focus, question areas you want to drill..."
                className="min-h-[140px] w-full rounded-2xl border border-slate-800 bg-slate-950/70 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/40"
              />
              <p className="text-xs text-slate-400">
                This helps the planner prioritize topics during the interview.
              </p>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="text-xs text-slate-400">
                Powered by Modal and OpenRouter.
              </div>
              <button
                type="submit"
                disabled={!resumeFile}
                className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-emerald-500 via-emerald-400 to-cyan-400 px-6 py-3 text-sm font-semibold text-slate-950 shadow-lg transition hover:shadow-emerald-500/40 focus:outline-none focus:ring-2 focus:ring-emerald-300/70 focus:ring-offset-2 focus:ring-offset-slate-950 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Generate interview plan
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  if (phase === "thinking") {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center px-6 text-slate-100">
        <div className="w-full max-w-lg rounded-3xl border border-white/10 bg-slate-900/70 p-12 text-center shadow-2xl backdrop-blur">
          <div className="mx-auto mb-8 h-16 w-16 animate-spin rounded-full border-2 border-slate-700 border-t-emerald-400" />
          <h2 className="text-2xl font-semibold tracking-tight">
            {thinkingHeadline}
          </h2>
          <p className="mt-4 text-sm text-slate-300">{thinkingSubhead}</p>
          <p className="mt-8 text-xs uppercase tracking-[0.35em] text-slate-500">
            Parsing your resume...
          </p>
        </div>
      </div>
    );
  }

  if (phase === "planning") {
    return (
      <div className="min-h-screen px-6 py-14 text-slate-100">
        <div className="mx-auto flex max-w-6xl flex-col gap-8">
          <header className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-slate-900/60 p-8 backdrop-blur">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-emerald-400/80">
              Interview plan {planIteration + 1}
            </p>
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <h2 className="text-3xl font-semibold tracking-tight">
                Review the outline before we go live
              </h2>
              <div className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-4 py-2 text-xs font-semibold uppercase text-emerald-300">
                Interactive voice chat to follow
              </div>
            </div>
            <div className="flex flex-wrap gap-4 text-xs text-slate-400">
              {resumeFileName ? (
                <span className="rounded-full border border-white/10 bg-slate-800/60 px-3 py-1">
                  Resume: {resumeFileName}
                </span>
              ) : (
                <span className="rounded-full border border-white/10 bg-slate-800/60 px-3 py-1">
                  No resume uploaded
                </span>
              )}
              {supplementalInfo.trim() ? (
                <span className="rounded-full border border-white/10 bg-slate-800/60 px-3 py-1">
                  Notes in scope
                </span>
              ) : (
                <span className="rounded-full border border-white/10 bg-slate-800/60 px-3 py-1">
                  No supplemental notes
                </span>
              )}
            </div>
          </header>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
            <section className="flex flex-col gap-6 rounded-3xl border border-white/10 bg-slate-900/60 p-6 backdrop-blur">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Planned flow</h3>
                <span className="text-xs text-slate-400">
                  Built from past interviews at major companies
                </span>
              </div>
              <div className="flex flex-col gap-4">
                {planSections.map((section, index) => (
                  <article
                    key={section.id}
                    className="flex gap-4 rounded-2xl border border-white/10 bg-slate-950/40 px-5 py-4"
                  >
                    <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-emerald-500/15 text-sm font-semibold text-emerald-200">
                      {index + 1}
                    </div>
                    <div>
                      <h4 className="text-base font-semibold">
                        {section.title}
                      </h4>
                      <p className="mt-2 text-sm text-slate-300">
                        {section.description}
                      </p>
                    </div>
                  </article>
                ))}
              </div>
              <button
                type="button"
                onClick={handleAcceptPlan}
                className="mt-2 inline-flex items-center justify-center gap-2 self-start rounded-full bg-gradient-to-r from-emerald-500 via-emerald-400 to-cyan-400 px-6 py-3 text-sm font-semibold text-slate-950 shadow-lg transition hover:shadow-emerald-500/40 focus:outline-none focus:ring-2 focus:ring-emerald-300/70 focus:ring-offset-2 focus:ring-offset-slate-950"
              >
                Accept plan & start interview
              </button>
            </section>
          </div>
        </div>
      </div>
    );
  }

  if (phase === "complete" && interviewReport) {
    return (
      <div className="min-h-screen px-6 py-14 text-slate-100">
        <div className="mx-auto flex max-w-5xl flex-col gap-8">
          <header className="rounded-3xl border border-white/10 bg-slate-900/60 p-8 backdrop-blur">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-emerald-400/80">
              Mock interview report
            </p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight">
              Interview Summary & Coaching Cues
            </h2>
            <p className="mt-4 text-sm text-slate-300">
              {interviewReport.overview}
            </p>
          </header>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
            <section className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 backdrop-blur">
              <h3 className="text-lg font-semibold">Highlights</h3>
              <ul className="mt-4 space-y-3 text-sm text-slate-300">
                {interviewReport.highlights.map((item, index) => (
                  <li
                    key={`highlight-${index}`}
                    className="flex gap-3 rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3"
                  >
                    <span className="mt-1 h-6 w-6 flex-none rounded-full bg-emerald-500/20 text-center text-xs font-semibold text-emerald-200">
                      {index + 1}
                    </span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </section>

            <section className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-slate-900/60 p-6 backdrop-blur">
              <div>
                <h3 className="text-lg font-semibold">Recommendations</h3>
                <ul className="mt-4 space-y-3 text-sm text-slate-300">
                  {interviewReport.recommendations.map((item, index) => (
                    <li
                      key={`recommendation-${index}`}
                      className="rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3"
                    >
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
              <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-4 text-sm text-emerald-100">
                <p className="font-semibold uppercase tracking-[0.3em] text-emerald-300">
                  Next steps
                </p>
                <p className="mt-2 text-sm text-emerald-100">
                  {interviewReport.nextSteps}
                </p>
              </div>
              <button
                type="button"
                onClick={resetExperience}
                className="mt-auto inline-flex items-center justify-center rounded-full bg-gradient-to-r from-emerald-500 via-emerald-400 to-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 shadow-lg transition hover:shadow-emerald-500/40"
              >
                Restart with a fresh resume
              </button>
            </section>
          </div>

          <section className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 backdrop-blur">
            <h3 className="text-lg font-semibold">What was on deck</h3>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              {planSections.map((section, index) => (
                <div
                  key={`report-plan-${section.id}`}
                  className="rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3 text-sm text-slate-300"
                >
                  <p className="font-semibold text-slate-100">
                    {index + 1}. {section.title}
                  </p>
                  <p className="mt-2 text-xs text-slate-400">
                    {section.description}
                  </p>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    );
  }

  if (phase === "complete") {
    return (
      <div className="flex min-h-screen items-center justify-center px-6 py-12 text-slate-100">
        <div className="w-full max-w-xl rounded-3xl border border-white/10 bg-slate-900/60 p-8 text-center backdrop-blur">
          <h2 className="text-2xl font-semibold tracking-tight">
            Report unavailable
          </h2>
          <p className="mt-3 text-sm text-slate-300">
            We wrapped the session, but something went wrong while generating
            the summary. Let’s start a fresh run.
          </p>
          <button
            type="button"
            onClick={resetExperience}
            className="mt-6 inline-flex items-center justify-center rounded-full bg-gradient-to-r from-emerald-500 via-emerald-400 to-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 shadow-lg transition hover:shadow-emerald-500/40"
          >
            Restart session
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-6 py-14 text-slate-100">
      <div className="mx-auto flex max-w-6xl flex-col gap-8">
        <header className="rounded-3xl border border-white/10 bg-slate-900/60 p-8 backdrop-blur">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-emerald-400/80">
            Live mock interview
          </p>
          <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <h2 className="text-3xl font-semibold tracking-tight">
              Coding Canvas
            </h2>
            <div className="flex items-center gap-3 text-xs text-slate-400">
              <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              Voice link enabled
            </div>
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.9fr)_minmax(0,1fr)]">
          <section className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-slate-900/60 p-6 backdrop-blur">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Workspace</h3>
              <span className="text-xs text-slate-400">
                Language: Python (Monaco editor)
              </span>
            </div>
            <div className="overflow-hidden rounded-2xl border border-white/10">
              <MonacoEditor
                height="520px"
                defaultLanguage="python"
                theme="vs-dark"
                value={editorValue}
                onChange={(value) => setEditorValue(value ?? "")}
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  fontFamily: "JetBrains Mono, ui-monospace, SFMono-Regular",
                  scrollBeyondLastLine: false,
                  smoothScrolling: true,
                }}
              />
            </div>
            <div className="flex flex-col gap-3 rounded-2xl border border-white/10 bg-slate-950/60 p-4 text-sm text-slate-300">
              <p>
                Speak through your thought process. When you pause to code
                silently, use the button below to nudge the interviewer in lieu
                of the voice trigger.
              </p>
              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={handleManualPrompt}
                  className="rounded-full border border-emerald-500/40 bg-emerald-500/15 px-4 py-2 text-xs font-semibold text-emerald-100 transition hover:bg-emerald-500/25"
                >
                  Manually prompt the interviewer
                </button>
                <button
                  type="button"
                  onClick={handleEndInterview}
                  className="rounded-full border border-rose-500/50 bg-rose-500/15 px-4 py-2 text-xs font-semibold text-rose-100 transition hover:bg-rose-500/25"
                >
                  End interview & view report
                </button>
              </div>
            </div>
          </section>

          <section className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-slate-900/60 p-6 backdrop-blur">
            <h3 className="text-lg font-semibold">Conversation timeline</h3>
            <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">
                Live transcript
              </p>
              <div className="mt-3 max-h-[260px] space-y-4 overflow-y-auto pr-2">
                {transcript.map((entry) => (
                  <div
                    key={entry.id}
                    className={`rounded-2xl border px-4 py-3 text-sm ${entry.speaker === "interviewer"
                      ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-100"
                      : "border-white/10 bg-slate-800/70 text-slate-100"
                      }`}
                  >
                    <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.3em] text-slate-400">
                      <span>
                        {entry.speaker === "interviewer"
                          ? "Interviewer"
                          : "You"}
                      </span>
                      <span>{entry.timestamp}</span>
                    </div>
                    <p className="mt-2">{entry.content}</p>
                  </div>
                ))}
                {transcript.length === 0 && (
                  <div className="text-xs text-slate-500">
                    Transcript will populate once the interview kicks off.
                  </div>
                )}
              </div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">
                Current outline
              </p>
              <ul className="mt-3 space-y-3 text-sm text-slate-300">
                {planSections.map((section, index) => (
                  <li key={section.id} className="rounded-xl bg-slate-900/70 p-3">
                    <p className="font-semibold text-slate-100">
                      {index + 1}. {section.title}
                    </p>
                    <p className="mt-1 text-xs text-slate-400">
                      {section.description}
                    </p>
                  </li>
                ))}
              </ul>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
