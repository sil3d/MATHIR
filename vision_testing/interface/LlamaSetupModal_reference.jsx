import React, { useEffect, useState, useRef } from "react";
import styled, { keyframes } from "styled-components";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, Download, CheckCircle, X, Zap, AlertTriangle, RotateCcw } from "lucide-react";
import { toast } from "react-hot-toast";
import CONFIG from "../config/apiConfig";
import { getLlamaSetupTranslation } from "../translations";
import { safeParseJson } from "../utils/safeJson.js";

const API = CONFIG.API_BASE;

// ─── Animations ───────────────────────────────────────────────────────────────
const pulse = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
`;

const scanLine = keyframes`
  0% { transform: translateY(0); opacity: 0.8; }
  100% { transform: translateY(120px); opacity: 0; }
`;

const shimmer = keyframes`
  0% { background-position: -200% center; }
  100% { background-position: 200% center; }
`;

// ─── Styled components ────────────────────────────────────────────────────────
const Overlay = styled(motion.div)`
  position: fixed;
  inset: 0;
  background: rgba(20, 20, 28, 0.75);
  backdrop-filter: blur(6px);
  z-index: 10005;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
`;

const Modal = styled(motion.div)`
  background: #28282f;
  border: 1px solid color-mix(in srgb, var(--accent) 20%);
  border-radius: 16px;
  width: 100%;
  max-width: 480px;
  padding: 32px;
  box-shadow: 0 24px 48px rgba(0, 0, 0, 0.6), 0 0 40px color-mix(in srgb, var(--accent) 8%);
  position: relative;
`;

const CloseBtn = styled.button`
  position: absolute;
  top: 16px;
  right: 16px;
  background: rgba(230, 230, 232, 0.06);
  border: none;
  border-radius: 8px;
  color: rgba(230, 230, 232, 0.5);
  width: 32px;
  height: 32px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 150ms ease;
  &:hover { background: rgba(230, 230, 232, 0.1); color: var(--text-color); }
`;

const IconRing = styled.div`
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: color-mix(in srgb, var(--accent) 10%);
  border: 1px solid color-mix(in srgb, var(--accent) 30%);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 20px;
  position: relative;
  overflow: hidden;

  &::after {
    content: "";
    position: absolute;
    width: 100%;
    height: 3px;
    background: linear-gradient(to right, transparent, color-mix(in srgb, var(--accent) 80%), transparent);
    animation: ${scanLine} 1.6s ease-in-out infinite;
  }
`;

const Title = styled.h2`
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--text-color);
  text-align: center;
  margin: 0 0 6px;
`;

const Subtitle = styled.p`
  font-size: 0.8rem;
  color: rgba(230, 230, 232, 0.5);
  text-align: center;
  margin: 0 0 24px;
`;

const HardwareCard = styled.div`
  background: color-mix(in srgb, var(--accent) 4%);
  border: 1px solid color-mix(in srgb, var(--accent) 15%);
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const HardwareRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.8rem;
  color: rgba(230, 230, 232, 0.6);

  span:last-child {
    color: var(--accent);
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
  }
`;

const ProgressBar = styled.div`
  background: rgba(230, 230, 232, 0.08);
  border-radius: 6px;
  height: 6px;
  overflow: hidden;
  margin-bottom: 8px;
`;

const ProgressFill = styled.div`
  height: 100%;
  border-radius: 6px;
  background: linear-gradient(90deg, var(--accent), #5a95a3);
  background-size: 200% auto;
  animation: ${shimmer} 2s linear infinite;
  transition: width 0.4s ease;
  width: ${p => p.$pct}%;
`;

const StatusText = styled.p`
  font-size: 0.78rem;
  color: rgba(230, 230, 232, 0.55);
  text-align: center;
  margin: 0 0 4px;
  min-height: 18px;
  animation: ${p => p.$pulse ? pulse : "none"} 1.5s ease-in-out infinite;
`;

const BackgroundNote = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  background: color-mix(in srgb, var(--accent) 6%);
  border: 1px solid color-mix(in srgb, var(--accent) 15%);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 0.78rem;
  color: rgba(230, 230, 232, 0.6);
  margin-top: 16px;
`;

const DismissBtn = styled.button`
  width: 100%;
  margin-top: 16px;
  padding: 10px;
  background: color-mix(in srgb, var(--accent) 8%);
  border: 1px solid color-mix(in srgb, var(--accent) 20%);
  border-radius: 8px;
  color: rgba(230, 230, 232, 0.7);
  font-size: 0.82rem;
  cursor: pointer;
  transition: all 150ms ease;
  font-family: 'Plus Jakarta Sans', sans-serif;
  &:hover {
    background: color-mix(in srgb, var(--accent) 14%);
    color: var(--text-color);
  }
`;

const SuccessIcon = styled.div`
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: color-mix(in srgb, var(--accent) 15%);
  border: 1px solid color-mix(in srgb, var(--accent) 40%);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 20px;
  color: var(--accent);
`;

const WhyBox = styled.div`
  background: rgba(230, 230, 232, 0.04);
  border: 1px solid rgba(230, 230, 232, 0.1);
  border-radius: 10px;
  padding: 12px 14px;
  margin-bottom: 20px;
`;

const WhyTitle = styled.p`
  font-size: 0.75rem;
  font-weight: 600;
  color: rgba(230, 230, 232, 0.5);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin: 0 0 6px;
`;

const WhyBody = styled.p`
  font-size: 0.78rem;
  color: rgba(230, 230, 232, 0.55);
  line-height: 1.55;
  margin: 0;
`;

const RestartBtn = styled.button`
  width: 100%;
  margin-top: 16px;
  padding: 11px;
  background: color-mix(in srgb, var(--accent) 15%);
  border: 1px solid color-mix(in srgb, var(--accent) 40%);
  border-radius: 8px;
  color: var(--accent);
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  transition: all 150ms ease;
  font-family: 'Plus Jakarta Sans', sans-serif;
  &:hover {
    background: color-mix(in srgb, var(--accent) 25%);
    border-color: color-mix(in srgb, var(--accent) 60%);
    color: var(--text-color);
  }
`;

// ─── Main component ───────────────────────────────────────────────────────────
export default function LlamaSetupModal({ lang = "en", onClose }) {
  const t = (key) => getLlamaSetupTranslation(key, lang);
  const [hardware, setHardware] = useState(null);
  const [progress, setProgress] = useState({ status: "idle", percent: 0, message_en: "", message_fr: "" });
  const [phase, setPhase] = useState("detecting"); // detecting | downloading | done | error
  const sseRef = useRef(null);

  // Step 1: fetch hardware info
  useEffect(() => {
    fetch(`${API}/api/llama-server/hardware`)
      .then(r => r.json())
      .then(data => {
        setHardware(data);
        // If binary already available, skip straight to done
        if (data.binary_available) {
          setPhase("done");
          setProgress({ status: "ready", percent: 100, message_en: "Already installed", message_fr: "Déjà installé" });
          return;
        }
        // Start download
        setPhase("downloading");
        startSetup();
      })
      .catch(() => setPhase("downloading") && startSetup());
  }, []);

  const startSetup = () => {
    fetch(`${API}/api/llama-server/setup`, { method: "POST" })
      .then(() => listenProgress())
      .catch(() => setPhase("error"));
  };

  const listenProgress = () => {
    const sse = new EventSource(`${API}/api/llama-server/progress`);
    sseRef.current = sse;

    sse.onmessage = (e) => {
      try {
        const data = safeParseJson(e.data, null, "llama-setup-sse");
        if (!data) return;
        setProgress(data);

        if (data.status === "ready" || data.status === "running") {
          setPhase("done");
          sse.close();
          toast.success(
            lang === "fr"
              ? "Moteur IA installé — redémarrez l'app !"
              : "AI engine installed — please restart the app!",
            { icon: "⚡", duration: 6000 }
          );
        } else if (data.status === "error") {
          setPhase("error");
          sse.close();
        }
      } catch {}
    };

    sse.onerror = () => {
      sse.close();
    };
  };

  useEffect(() => {
    return () => sseRef.current?.close();
  }, []);

  const handleRetry = () => {
    setPhase("downloading");
    setProgress({ status: "idle", percent: 0, message_en: "", message_fr: "" });
    startSetup();
  };

  const msgText = lang === "fr" ? progress.message_fr : progress.message_en;

  return (
    <AnimatePresence>
      <Overlay
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <Modal
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          transition={{ duration: 0.25 }}
        >
          {/* Close (minimize) button */}
          <CloseBtn onClick={onClose} title={t("dismiss")}>
            <X size={14} />
          </CloseBtn>

          {/* ── Detecting phase ── */}
          {phase === "detecting" && (
            <>
              <IconRing>
                <Cpu size={28} color="var(--accent)" />
              </IconRing>
              <Title>{t("title")}</Title>
              <Subtitle>{t("subtitle")}</Subtitle>
              <WhyBox>
                <WhyTitle>{t("whyTitle")}</WhyTitle>
                <WhyBody>{t("whyBody")}</WhyBody>
              </WhyBox>
              <StatusText $pulse>{t("detecting")}</StatusText>
            </>
          )}

          {/* ── Downloading phase ── */}
          {phase === "downloading" && (
            <>
              <IconRing>
                <Download size={28} color="var(--accent)" />
              </IconRing>
              <Title>{t("downloadingTitle")}</Title>
              <Subtitle>{t("downloadingSubtitle")}</Subtitle>

              {hardware && (
                <HardwareCard>
                  <HardwareRow>
                    <span>{t("gpu")}</span>
                    <span>{hardware.gpu_label || hardware.gpu_type}</span>
                  </HardwareRow>
                  <HardwareRow>
                    <span>{t("ram")}</span>
                    <span>{hardware.ram_total_gb} GB</span>
                  </HardwareRow>
                  <HardwareRow>
                    <span>{t("backend")}</span>
                    <span>{hardware.backend_label || hardware.backend}</span>
                  </HardwareRow>
                </HardwareCard>
              )}

              <ProgressBar>
                <ProgressFill $pct={progress.percent} />
              </ProgressBar>
              <StatusText $pulse={progress.status !== "ready"}>
                {msgText || t("detecting")}
              </StatusText>
              {progress.status === "downloading" && progress.message_en && progress.message_en.includes("(") && (
                <StatusText style={{ fontSize: "0.72rem", color: "color-mix(in srgb, var(--accent) 70%)", marginTop: 2 }}>
                  {/* Show asset name e.g. llama-b8470-bin-win-cuda-12-x64.zip */}
                  {(lang === "fr" ? progress.message_fr : progress.message_en)
                    .match(/\(([^)]+)\)/)?.[1] || ""}
                </StatusText>
              )}

              <BackgroundNote>
                <Zap size={14} color="var(--accent)" style={{ flexShrink: 0 }} />
                {t("backgroundNote")}
              </BackgroundNote>

              <DismissBtn onClick={onClose}>{t("dismiss")}</DismissBtn>
            </>
          )}

          {/* ── Done phase ── */}
          {phase === "done" && (
            <>
              <SuccessIcon>
                <CheckCircle size={32} />
              </SuccessIcon>
              <Title>{t("doneTitle")}</Title>
              <Subtitle>{t("doneNote")}</Subtitle>

{/* Hardware details hidden on done screen — user just needs to know it's ready */}

              <ProgressBar>
                <ProgressFill $pct={100} />
              </ProgressBar>

              <StatusText style={{ marginTop: 4, color: "rgba(230,230,232,0.4)", fontSize: "0.76rem" }}>
                {t("restartNote")}
              </StatusText>

              <RestartBtn onClick={() => window.api?.send("restart_app") || window.location.reload()}>
                <RotateCcw size={15} />
                {t("restartBtn")}
              </RestartBtn>
              <DismissBtn onClick={onClose} style={{ marginTop: 8 }}>{t("dismissDone")}</DismissBtn>
            </>
          )}

          {/* ── Error phase ── */}
          {phase === "error" && (
            <>
              <SuccessIcon style={{ background: "rgba(230,230,232,0.06)", borderColor: "rgba(230,230,232,0.2)" }}>
                <AlertTriangle size={32} color="rgba(230,230,232,0.6)" />
              </SuccessIcon>
              <Title>{t("errorTitle")}</Title>
              <Subtitle>{t("errorNote")}</Subtitle>
              {msgText && (
                <StatusText style={{ color: "rgba(230,230,232,0.4)", fontSize: "0.72rem", marginBottom: 16 }}>
                  {msgText}
                </StatusText>
              )}
              <DismissBtn
                onClick={handleRetry}
                style={{ background: "color-mix(in srgb, var(--accent) 12%)", borderColor: "color-mix(in srgb, var(--accent) 30%)", color: "var(--accent)" }}
              >
                {t("retry")}
              </DismissBtn>
              <DismissBtn onClick={onClose} style={{ marginTop: 8 }}>{t("dismiss")}</DismissBtn>
            </>
          )}
        </Modal>
      </Overlay>
    </AnimatePresence>
  );
}
