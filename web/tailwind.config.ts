import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        f1: {
          red: "#E8002D",
          bg1: "#111111",
          bg2: "#1a1a1a",
        },
      },
      fontFamily: {
        sans: ["Space Grotesk", "system-ui", "sans-serif"],
      },
      borderRadius: {
        none: "0",
      },
    },
  },
  plugins: [],
};

export default config;
