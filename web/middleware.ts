import { NextResponse, type NextRequest } from "next/server";
import {
  featureForPath,
  isFeatureEnabled,
  firstEnabledFeature,
} from "@/lib/features";
import { readDisabledFeatures } from "@/lib/env";

/**
 * Route-level enforcement for the operator feature kill-switch.
 *
 * Hiding a disabled feature from the nav is not enough — its route is still
 * reachable by typing the URL. When a request targets a disabled feature
 * (including the "/" landing when Race Dashboard is off), redirect it to the
 * first still-enabled feature so there are no dead-end URLs.
 */
export function middleware(req: NextRequest) {
  const key = featureForPath(req.nextUrl.pathname);
  if (!key) return NextResponse.next();

  // Read the runtime kill-switch once per request (server/edge env, not inlined).
  const disabled = readDisabledFeatures();
  if (isFeatureEnabled(key, disabled)) return NextResponse.next();

  const target = firstEnabledFeature(disabled);
  // Guard against a redirect loop if literally everything is disabled.
  if (!target || req.nextUrl.pathname === target.href) return NextResponse.next();

  return NextResponse.redirect(new URL(target.href, req.url));
}

export const config = {
  // Only run on the feature routes + landing; skip _next, static assets, api, favicon.
  // Each feature lists the bare path and its subpaths so a disabled top-level
  // route (e.g. "/radar") is intercepted, not just its children.
  matcher: [
    "/",
    "/radar",
    "/radar/:path*",
    "/mystery-driver",
    "/mystery-driver/:path*",
    "/race",
    "/race/:path*",
    "/event",
    "/event/:path*",
    "/pipeline",
    "/pipeline/:path*",
  ],
};
