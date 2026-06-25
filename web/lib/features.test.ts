import { describe, it, expect, afterEach, vi } from "vitest";
import {
  FEATURES,
  isFeatureEnabled,
  enabledFeatures,
  firstEnabledFeature,
  featureForPath,
  parseDisabledFeatures,
} from "./features";
import { readDisabledFeatures } from "./env";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("parseDisabledFeatures", () => {
  it("returns an empty set for undefined / empty", () => {
    expect(parseDisabledFeatures(undefined).size).toBe(0);
    expect(parseDisabledFeatures("").size).toBe(0);
  });

  it("parses multiple keys and tolerates whitespace / empty entries", () => {
    expect([...parseDisabledFeatures(" mystery-driver , pipeline , ")]).toEqual([
      "mystery-driver",
      "pipeline",
    ]);
  });
});

describe("enabledFeatures / isFeatureEnabled", () => {
  it("enables all features when nothing is disabled", () => {
    const disabled = parseDisabledFeatures("");
    expect(enabledFeatures(disabled)).toEqual(FEATURES);
    expect(isFeatureEnabled("radar", disabled)).toBe(true);
    expect(isFeatureEnabled("pipeline", disabled)).toBe(true);
  });

  it("hides a disabled feature and keeps the rest", () => {
    const disabled = parseDisabledFeatures("mystery-driver");
    expect(isFeatureEnabled("mystery-driver", disabled)).toBe(false);
    expect(enabledFeatures(disabled).map((f) => f.key)).toEqual([
      "radar",
      "race",
      "pipeline",
    ]);
  });

  it("hides multiple disabled features", () => {
    const disabled = parseDisabledFeatures("mystery-driver,pipeline");
    expect(enabledFeatures(disabled).map((f) => f.key)).toEqual(["radar", "race"]);
  });
});

describe("firstEnabledFeature", () => {
  it("returns the first non-disabled feature in registry order", () => {
    expect(firstEnabledFeature(parseDisabledFeatures("radar"))?.key).toBe(
      "mystery-driver",
    );
  });

  it("returns null when everything is disabled", () => {
    expect(
      firstEnabledFeature(
        parseDisabledFeatures("radar,mystery-driver,race,pipeline"),
      ),
    ).toBeNull();
  });
});

describe("readDisabledFeatures (server, runtime env)", () => {
  it("reads DISABLED_FEATURES from the environment at call time", () => {
    vi.stubEnv("DISABLED_FEATURES", "mystery-driver, pipeline");
    expect([...readDisabledFeatures()]).toEqual(["mystery-driver", "pipeline"]);
  });

  it("treats an unset var as no features disabled", () => {
    vi.stubEnv("DISABLED_FEATURES", "");
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
