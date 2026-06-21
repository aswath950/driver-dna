import { describe, it, expect, afterEach, vi } from "vitest";
import {
  FEATURES,
  isFeatureEnabled,
  enabledFeatures,
  firstEnabledFeature,
  featureForPath,
} from "./features";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("enabledFeatures / isFeatureEnabled", () => {
  it("enables all features when the var is unset", () => {
    vi.stubEnv("NEXT_PUBLIC_DISABLED_FEATURES", "");
    expect(enabledFeatures()).toEqual(FEATURES);
    expect(isFeatureEnabled("radar")).toBe(true);
    expect(isFeatureEnabled("pipeline")).toBe(true);
  });

  it("hides a disabled feature and keeps the rest", () => {
    vi.stubEnv("NEXT_PUBLIC_DISABLED_FEATURES", "mystery-driver");
    expect(isFeatureEnabled("mystery-driver")).toBe(false);
    expect(enabledFeatures().map((f) => f.key)).toEqual([
      "radar",
      "race",
      "pipeline",
    ]);
  });

  it("parses multiple keys and tolerates whitespace", () => {
    vi.stubEnv("NEXT_PUBLIC_DISABLED_FEATURES", " mystery-driver , pipeline ");
    expect(enabledFeatures().map((f) => f.key)).toEqual(["radar", "race"]);
  });
});

describe("firstEnabledFeature", () => {
  it("returns the first non-disabled feature in registry order", () => {
    vi.stubEnv("NEXT_PUBLIC_DISABLED_FEATURES", "radar");
    expect(firstEnabledFeature()?.key).toBe("mystery-driver");
  });

  it("returns null when everything is disabled", () => {
    vi.stubEnv(
      "NEXT_PUBLIC_DISABLED_FEATURES",
      "radar,mystery-driver,race,pipeline",
    );
    expect(firstEnabledFeature()).toBeNull();
  });
});

describe("featureForPath", () => {
  it("maps the landing and race routes to race", () => {
    expect(featureForPath("/")).toBe("race");
    expect(featureForPath("/race")).toBe("race");
    expect(featureForPath("/race/42")).toBe("race");
    expect(featureForPath("/event/abc")).toBe("race");
  });

  it("maps the other feature routes to their keys", () => {
    expect(featureForPath("/radar")).toBe("radar");
    expect(featureForPath("/mystery-driver")).toBe("mystery-driver");
    expect(featureForPath("/pipeline/anything")).toBe("pipeline");
  });

  it("returns null for unknown paths", () => {
    expect(featureForPath("/api/v1/healthz")).toBeNull();
    expect(featureForPath("/some/other")).toBeNull();
  });
});
