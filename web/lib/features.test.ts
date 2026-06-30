import { describe, it, expect, afterEach, vi } from "vitest";
import {
  FEATURES,
  isFeatureEnabled,
  enabledFeatures,
  firstEnabledFeature,
  featureForPath,
  disabledFromFlags,
} from "./features";
import { readDisabledFeatures } from "./env";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("disabledFromFlags", () => {
  it("returns an empty set when nothing is FALSE (unset/TRUE/other = enabled)", () => {
    expect(disabledFromFlags({}).size).toBe(0);
    expect(
      disabledFromFlags({
        radar: "TRUE",
        "mystery-driver": undefined,
        race: "",
        pipeline: "anything",
      }).size,
    ).toBe(0);
  });

  it("disables only features whose flag is FALSE (case-insensitive, trimmed)", () => {
    expect([
      ...disabledFromFlags({ "mystery-driver": "FALSE", pipeline: " false " }),
    ]).toEqual(["mystery-driver", "pipeline"]);
  });
});

describe("enabledFeatures / isFeatureEnabled", () => {
  it("enables all features when nothing is disabled", () => {
    const disabled = disabledFromFlags({});
    expect(enabledFeatures(disabled)).toEqual(FEATURES);
    expect(isFeatureEnabled("radar", disabled)).toBe(true);
    expect(isFeatureEnabled("pipeline", disabled)).toBe(true);
  });

  it("hides a disabled feature and keeps the rest", () => {
    const disabled = disabledFromFlags({ "mystery-driver": "FALSE" });
    expect(isFeatureEnabled("mystery-driver", disabled)).toBe(false);
    expect(enabledFeatures(disabled).map((f) => f.key)).toEqual([
      "radar",
      "race",
      "pipeline",
    ]);
  });

  it("hides multiple disabled features", () => {
    const disabled = disabledFromFlags({ "mystery-driver": "FALSE", pipeline: "FALSE" });
    expect(enabledFeatures(disabled).map((f) => f.key)).toEqual(["radar", "race"]);
  });
});

describe("firstEnabledFeature", () => {
  it("returns the first non-disabled feature in registry order", () => {
    expect(firstEnabledFeature(disabledFromFlags({ radar: "FALSE" }))?.key).toBe(
      "mystery-driver",
    );
  });

  it("returns null when everything is disabled", () => {
    expect(
      firstEnabledFeature(
        disabledFromFlags({
          radar: "FALSE",
          "mystery-driver": "FALSE",
          race: "FALSE",
          pipeline: "FALSE",
        }),
      ),
    ).toBeNull();
  });
});

describe("readDisabledFeatures (server, runtime env)", () => {
  it("reads the per-feature FEATURE_* vars at call time", () => {
    vi.stubEnv("FEATURE_MYSTERY_DRIVER", "FALSE");
    vi.stubEnv("FEATURE_PIPELINE", "FALSE");
    expect([...readDisabledFeatures()]).toEqual(["mystery-driver", "pipeline"]);
  });

  it("treats unset / TRUE vars as enabled", () => {
    vi.stubEnv("FEATURE_RADAR", "TRUE");
    expect(readDisabledFeatures().size).toBe(0);
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
