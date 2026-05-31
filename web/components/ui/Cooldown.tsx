"use client";

import { useState, useEffect, useCallback } from "react";

const COOLDOWN_SECS = 5;

/**
 * Returns a countdown that starts at COOLDOWN_SECS and ticks to 0.
 * Call `startCooldown()` after each AI request fires.
 */
export function useCooldown() {
  const [secondsLeft, setSecondsLeft] = useState(0);

  // Each decrement schedules the next via setTimeout — avoids stale closure issues.
  useEffect(() => {
    if (secondsLeft <= 0) return;
    const id = setTimeout(
      () => setSecondsLeft((s) => Math.max(0, s - 1)),
      1000,
    );
    return () => clearTimeout(id);
  }, [secondsLeft]);

  const startCooldown = useCallback(() => {
    setSecondsLeft(COOLDOWN_SECS);
  }, []);

  return { secondsLeft, startCooldown, onCooldown: secondsLeft > 0 };
}
