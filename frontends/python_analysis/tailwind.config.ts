import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#4F46E5",
          foreground: "#FFFFFF"
        },
        muted: {
          DEFAULT: "#F3F4F6",
          foreground: "#6B7280"
        }
      }
    }
  },
  plugins: []
};

export default config;

