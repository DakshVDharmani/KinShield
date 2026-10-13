function $(selector, root) { return (root || document).querySelector(selector); }
function $$(selector, root) { return [...(root || document).querySelectorAll(selector)]; }

// ── Lesson-exploration progress (unchanged) ────────────────────────────────
const progressKey = "kinshield-learning-v1";
let explored = new Set();
try { explored = new Set(JSON.parse(localStorage.getItem(progressKey) || "[]")); } catch { explored = new Set(); }
const progressBar = $(".progress-track");
function paintProgress() {
  const count = Math.min(explored.size, 7);
  $("#progress-count").textContent = `${count} of 7`;
  $("#progress-fill").style.width = `${count / 7 * 100}%`;
  progressBar.setAttribute("aria-valuenow", String(count));
}
paintProgress();

const lessonObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    const id = entry.target.dataset.lesson;
    if (!explored.has(id)) {
      explored.add(id);
      try { localStorage.setItem(progressKey, JSON.stringify([...explored])); } catch { /* Private browsing may block storage. */ }
      paintProgress();
    }
    $$(".lesson-nav a").forEach((link) => link.classList.toggle("active", link.hash === `#${id}`));
  });
}, { rootMargin: "-22% 0px -62% 0px" });
$$("[data-lesson]").forEach((section) => lessonObserver.observe(section));

// ── Topic search ────────────────────────────────────────────────────────────
const search = $("#topic-search");
search.addEventListener("input", () => {
  const query = search.value.trim().toLowerCase();
  let visible = 0;
  $$(".topic-card").forEach((card) => {
    const show = !query || `${card.dataset.search} ${card.textContent}`.toLowerCase().includes(query);
    card.hidden = !show;
    if (show) visible++;
  });
  $("#search-empty").hidden = visible > 0;
});

// ── Lesson 01 legacy quiz (the existing quiz-card) ──────────────────────────
const quizFeedback = $("#quiz-feedback");
$$(".quiz-options button").forEach((button) => button.addEventListener("click", () => {
  $$(".quiz-options button").forEach((choice) => choice.setAttribute("aria-pressed", String(choice === button)));
  quizFeedback.hidden = false;
  quizFeedback.textContent = button.dataset.answer === "right"
    ? "Good first step. A known, separate channel tests the request without relying on how convincing the voice sounds."
    : "A voice can sound familiar and still be copied. Pause, avoid sharing money or codes, and verify through a contact route you already trust.";
}));

// ── Age-band conversation prompts ───────────────────────────────────────────
const conversation = {
  young: ["\u201cYou can always tell me if something online feels confusing. We\u2019ll work it out together.\u201d", "Try: ask one calm, open question such as \u201cWhat happened next?\u201d and give them time."],
  middle: ["\u201cIf someone sends something that worries you, you can show me. We can decide what to do as a team.\u201d", "Try: ask what the child already tried and who else they would feel comfortable talking to."],
  teen: ["\u201cI\u2019ll listen first. We can figure out your options together, and I\u2019ll explain before I involve anyone else.\u201d", "Try: ask what outcome matters most to them, then make a safety plan that respects their voice."],
};
$$(".age-tabs button").forEach((button) => button.addEventListener("click", () => {
  $$(".age-tabs button").forEach((tab) => tab.classList.toggle("active", tab === button));
  const words = conversation[button.dataset.age];
  $("#conversation-line").textContent = words[0];
  $("#conversation-prompt").textContent = words[1];
}));

// ── Decision playbook ───────────────────────────────────────────────────────
const scenarios = {
  message: { title: "A suspicious voice or message", steps: ["Do not pay, share a code, or click a link in the message.", "Contact the person through a number or channel you already trust.", "If a child is involved, tell them you will check together and they are not in trouble."], note: "If money was sent, contact your bank promptly and use your country's official cybercrime reporting route." },
  impersonation: { title: "Someone impersonated a trusted person", steps: ["Pause the conversation and do not follow urgent instructions.", "Call the real person using a saved number or contact them another way.", "Report and block the impersonating account after preserving its username and link, if safe."], note: "If account access may be lost, use the platform's official account recovery page." },
  bullying: { title: "A fake image or video is spreading", steps: ["Reassure your child that creating or sharing it is not their fault.", "Do not repost it to prove it is fake. Save URLs, usernames, timestamps, and report IDs when safe.", "Report the content to the platform and involve a trusted school contact if classmates are involved."], note: "If it is sexual or intimate imagery involving a child, do not download, copy, or send it to KinShield. Use a child protection reporting channel." },
  threat: { title: "My child is being threatened or bullied", steps: ["Ask whether the person can reach your child offline or whether danger is immediate.", "Listen without blame. Keep the child with a trusted adult if they feel unsafe.", "Keep relevant messages, account names, links, times, and platform report references without forwarding harmful imagery."], note: "Contact local emergency services if there is immediate physical danger. Report threats through the platform and appropriate local services." },
  extortion: { title: "Someone demands money or more images", steps: ["Do not pay, send more images, negotiate, or threaten the sender.", "Tell your child they are not to blame and stay with them while you make a plan.", "Keep non-content details such as usernames, profile links, URLs, and timestamps. Report the account and seek child protection or law enforcement help."], note: "If immediate danger exists, contact local emergency services. Do not download or redistribute intimate imagery." },
  exploitation: { title: "I suspect sexual exploitation", steps: ["Do not ask the child to resend, download, screenshot, or show intimate imagery.", "Preserve non-content details only when safe and lawful: account names, URLs, timestamps, and report IDs.", "Report through the platform and a relevant regional child protection or law enforcement channel; stay with the child and reassure them."], note: "Do not upload such material to KinShield. If there is immediate danger, contact local emergency services now." },
  unsure: { title: "I am unsure whether it is authentic", steps: ["Do not forward it or act on an urgent request yet.", "Look for the original source, date, and context; contact the named person via a known channel.", "Treat a detector result as one limited signal. Ask a trusted person or official source if stakes are high."], note: "You can use the KinShield video checker for supported videos, with consent. It cannot prove authenticity, identity, or intent." },
};
$$(".decision-options button").forEach((button) => button.addEventListener("click", () => {
  $$(".decision-options button").forEach((option) => option.classList.toggle("active", option === button));
  const item = scenarios[button.dataset.scenario];
  $("#scenario-title").textContent = item.title;
  const list = $("#scenario-steps");
  list.replaceChildren(...item.steps.map((step) => { const li = document.createElement("li"); li.textContent = step; return li; }));
  $("#scenario-note").textContent = item.note;
  $("#scenario-label").textContent = button.dataset.scenario === "exploitation" ? "SENSITIVE · TEXT ONLY" : "FIRST RESPONSE";
}));

// ── Detector result examples ────────────────────────────────────────────────
const resultExamples = {
  signals: ["The video received stronger AI-associated signals", "This is a prompt to check the source and context. It is not proof the whole video is fake or that someone acted maliciously.", "Next: verify with the original speaker through a known channel."],
  none: ["No strong manipulation signals were returned", "A low score cannot confirm a video is authentic. The system may miss a manipulation, especially after edits or compression.", "Next: check where the video first appeared and whether the event is independently confirmed."],
  unclear: ["The check could not provide a useful result", "The file may not have returned enough analyzable frames, or the signals may not support a clear interpretation.", "Next: rely on context and a separate trusted source. Do not treat \u201cinconclusive\u201d as a verdict."],
};
$$(".result-tabs button").forEach((button) => button.addEventListener("click", () => {
  $$(".result-tabs button").forEach((tab) => tab.classList.toggle("active", tab === button));
  const [title, copy, next] = resultExamples[button.dataset.result];
  $("#result-title").textContent = title;
  $("#result-copy").textContent = copy;
  $("#result-next").textContent = next;
}));

// ── Copy helpers ────────────────────────────────────────────────────────────
const checklistText = () => {
  const title = $("#scenario-title").textContent;
  const steps = $$("#scenario-steps li").map((li) => `- ${li.textContent}`).join("\n");
  return `KinShield first steps \u2014 ${title}\n${steps}\n\nIf someone is in immediate physical danger, contact local emergency services. Do not forward harmful imagery involving a child.`;
};
async function copyText(value, status) {
  try {
    await navigator.clipboard.writeText(value);
    status.textContent = "Copied. You can paste this into your notes.";
  } catch {
    const field = document.createElement("textarea");
    field.value = value; field.style.position = "fixed"; field.style.opacity = "0";
    document.body.append(field); field.select();
    const copied = document.execCommand("copy"); field.remove();
    status.textContent = copied ? "Copied. You can paste this into your notes." : "Copy is unavailable here; you can select and copy the checklist.";
  }
}
$("#copy-checklist").addEventListener("click", () => copyText(checklistText(), $("#copy-status")));
$("#copy-family").addEventListener("click", () => copyText("Family conversation checklist\n- Who are two trusted adults you can contact?\n- What is one safe way to verify a family emergency?\n- What can we do if someone threatens to share something?\n- Asking for help starts with curiosity, not punishment.", $("#family-copy-status")));

// ═══════════════════════════════════════════════════════════════════════════
// SECTION QUIZZES + CERTIFICATE
// ═══════════════════════════════════════════════════════════════════════════

// State: track correct/total per question globally
const TOTAL_QUESTIONS = 21; // 7 sections × 3 questions
let quizAnswered = 0; // number of questions answered
let quizCorrect = 0;  // number answered correctly

// Each question can only be answered once
const answeredQuestions = new Set();

function updateQuizSidebar() {
  const sideBlock = $("#sidebar-quiz-score");
  if (quizAnswered === 0) { sideBlock.hidden = true; return; }
  sideBlock.hidden = false;
  $("#quiz-answered-count").textContent = `${quizAnswered} of ${TOTAL_QUESTIONS}`;
  const fillEl = $("#quiz-progress-fill");
  fillEl.style.width = `${(quizAnswered / TOTAL_QUESTIONS) * 100}%`;
  const pct = Math.round((quizCorrect / quizAnswered) * 100);
  $("#quiz-score-display").textContent = `${pct}%`;
}

function maybeShowCertificate() {
  if (quizAnswered < TOTAL_QUESTIONS) return; // not all answered yet
  const pct = Math.round((quizCorrect / TOTAL_QUESTIONS) * 100);
  const certSection = $("#certificate");
  certSection.hidden = false;
  certSection.scrollIntoView({ behavior: "smooth", block: "start" });

  const headline = $("#cert-headline");
  const subline = $("#cert-subline");
  const summary = $("#cert-score-summary");

  if (pct >= 80) {
    headline.textContent = "You've earned your KinShield certificate!";
    subline.textContent = "Outstanding! Enter your name below to download your certificate of completion.";
    summary.innerHTML = `<span class="cert-score-badge cert-pass">${pct}% \u2014 Passed</span> You answered ${quizCorrect} of ${TOTAL_QUESTIONS} questions correctly.`;
    $("#cert-name-row-wrap").hidden = false;
    drawCertificatePreview($("#cert-name").value.trim() || "Parent / Guardian");
  } else {
    headline.textContent = "Keep learning — you\u2019re getting there.";
    subline.textContent = `You scored ${pct}%. A score of 80% or above unlocks your certificate. Review the lessons above and try again.`;
    summary.innerHTML = `<span class="cert-score-badge cert-fail">${pct}% \u2014 ${quizCorrect} of ${TOTAL_QUESTIONS} correct</span> Revisit any section to refresh your knowledge.`;
    $("#cert-name-row-wrap").hidden = true;
  }
}

// ── Per-section quiz wiring ─────────────────────────────────────────────────
$$(".section-quiz").forEach((quizBlock) => {
  const section = quizBlock.dataset.section;
  const questions = $$(".sq-question", quizBlock);

  questions.forEach((qEl, qIndex) => {
    const key = `${section}-${qIndex}`;
    const options = $$(".sq-options button", qEl);
    const feedback = $(".sq-feedback", qEl);

    options.forEach((btn) => {
      btn.addEventListener("click", () => {
        if (answeredQuestions.has(key)) return; // already answered
        answeredQuestions.add(key);

        const correct = btn.dataset.correct === "true";
        quizAnswered++;
        if (correct) quizCorrect++;

        // Style all buttons
        options.forEach((b) => {
          b.disabled = true;
          if (b.dataset.correct === "true") {
            b.classList.add("sq-correct");
          } else if (b === btn && !correct) {
            b.classList.add("sq-wrong");
          }
        });

        feedback.hidden = false;
        feedback.textContent = correct
          ? "\u2713 Correct! " + btn.textContent
          : "\u2717 Not quite. The correct answer is highlighted in green above.";
        feedback.className = `sq-feedback ${correct ? "sq-fb-correct" : "sq-fb-wrong"}`;

        updateQuizSidebar();
        updateSectionScore(quizBlock, section);
        maybeShowCertificate();
      });
    });
  });
});

function updateSectionScore(quizBlock, section) {
  const total = 3;
  let correct = 0;
  for (let i = 0; i < total; i++) {
    const key = `${section}-${i}`;
    if (!answeredQuestions.has(key)) return; // not all answered yet
    // Check if this was answered correctly by looking at the button state
    const qEl = $(`.sq-question[data-q="${i}"]`, quizBlock);
    if ($(".sq-correct", qEl)) correct++;
  }
  // All 3 answered
  const scoreRow = $(".sq-score-row", quizBlock);
  const pct = Math.round((correct / total) * 100);
  $(".sq-score-val", scoreRow).textContent = `${correct} / ${total} (${pct}%)`;
  scoreRow.hidden = false;
}

// ── Certificate canvas drawing ──────────────────────────────────────────────
function drawCertificatePreview(name) {
  const canvas = $("#cert-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  // Background
  ctx.fillStyle = "#f8f7f3";
  ctx.fillRect(0, 0, W, H);

  // Outer border
  ctx.strokeStyle = "#748e7f";
  ctx.lineWidth = 3;
  ctx.strokeRect(14, 14, W - 28, H - 28);

  // Inner border (double-line effect)
  ctx.strokeStyle = "#c8d8cc";
  ctx.lineWidth = 1;
  ctx.strokeRect(22, 22, W - 44, H - 44);

  // Top corner ornaments
  const drawCorner = (x, y, sx, sy) => {
    ctx.strokeStyle = "#748e7f";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, y + 30 * sy);
    ctx.lineTo(x, y);
    ctx.lineTo(x + 30 * sx, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x + 8 * sx, y + 8 * sy, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#748e7f";
    ctx.fill();
  };
  drawCorner(30, 30, 1, 1);
  drawCorner(W - 30, 30, -1, 1);
  drawCorner(30, H - 30, 1, -1);
  drawCorner(W - 30, H - 30, -1, -1);

  // Brand mark ✳
  ctx.fillStyle = "#748e7f";
  ctx.font = "bold 22px 'DM Sans', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("✳  kinshield", W / 2, 68);

  // Thin rule
  ctx.strokeStyle = "#c8d8cc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(60, 82); ctx.lineTo(W - 60, 82);
  ctx.stroke();

  // "CERTIFICATE OF COMPLETION" eyebrow
  ctx.fillStyle = "#829b88";
  ctx.font = "600 9px 'DM Sans', sans-serif";
  ctx.letterSpacing = "2px";
  ctx.textAlign = "center";
  ctx.fillText("C E R T I F I C A T E   O F   C O M P L E T I O N", W / 2, 105);

  // Main title
  ctx.fillStyle = "#223047";
  ctx.font = "500 32px 'Newsreader', Georgia, serif";
  ctx.textAlign = "center";
  ctx.fillText("AI-Certified Parent", W / 2, 158);
  ctx.font = "500 22px 'Newsreader', Georgia, serif";
  ctx.fillText("for Child Safety", W / 2, 188);

  // Thin divider
  ctx.strokeStyle = "#dfe6dc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(160, 205); ctx.lineTo(W - 160, 205);
  ctx.stroke();

  // "This certifies that"
  ctx.fillStyle = "#6b7a72";
  ctx.font = "400 13px 'Newsreader', Georgia, serif";
  ctx.textAlign = "center";
  ctx.fillText("This certifies that", W / 2, 234);

  // Name
  ctx.fillStyle = "#304b3b";
  ctx.font = "600 36px 'Newsreader', Georgia, serif";
  ctx.textAlign = "center";
  ctx.fillText(name || "Parent / Guardian", W / 2, 282);

  // Underline for name
  const nameW = ctx.measureText(name || "Parent / Guardian").width;
  ctx.strokeStyle = "#748e7f";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(W / 2 - nameW / 2 - 8, 292);
  ctx.lineTo(W / 2 + nameW / 2 + 8, 292);
  ctx.stroke();

  // Body copy
  ctx.fillStyle = "#6b7a72";
  ctx.font = "400 12px 'Newsreader', Georgia, serif";
  ctx.textAlign = "center";
  const bodyLines = [
    "has successfully completed the KinShield Learning Centre",
    "and demonstrated knowledge of AI deepfakes, online safety,",
    "and child-protective responses."
  ];
  bodyLines.forEach((line, i) => ctx.fillText(line, W / 2, 322 + i * 20));

  // Score badge area
  const pct = Math.round((quizCorrect / TOTAL_QUESTIONS) * 100);
  ctx.fillStyle = "#eaf0ea";
  roundRect(ctx, W / 2 - 80, 383, 160, 36, 6);
  ctx.fill();
  ctx.fillStyle = "#304b3b";
  ctx.font = "600 13px 'DM Sans', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`Quiz score: ${pct}%  ·  ${quizCorrect}/${TOTAL_QUESTIONS} correct`, W / 2, 406);

  // Date
  const today = new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
  ctx.fillStyle = "#829b88";
  ctx.font = "400 11px 'DM Sans', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`Issued on ${today}`, W / 2, 447);

  // Bottom rule
  ctx.strokeStyle = "#c8d8cc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(60, 462); ctx.lineTo(W - 60, 462);
  ctx.stroke();

  // Footer note
  ctx.fillStyle = "#a8b5ad";
  ctx.font = "400 9px 'DM Sans', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("kinshield.app  ·  Educational reference — not emergency or legal advice", W / 2, 482);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// Redraw preview when name changes
const certNameInput = $("#cert-name");
if (certNameInput) {
  certNameInput.addEventListener("input", () => {
    drawCertificatePreview(certNameInput.value.trim() || "Parent / Guardian");
  });
}

// Download button
const certDownloadBtn = $("#cert-download-btn");
if (certDownloadBtn) {
  certDownloadBtn.addEventListener("click", () => {
    const name = (certNameInput?.value.trim()) || "Parent Guardian";
    drawCertificatePreview(name || "Parent / Guardian");
    const canvas = $("#cert-canvas");
    const link = document.createElement("a");
    link.download = `KinShield-Certificate-${name.replace(/\s+/g, "-")}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  });
}


