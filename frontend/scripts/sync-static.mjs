import { cpSync, existsSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..", "..");
const frontendOut = resolve(projectRoot, "frontend", "out");
const backendStatic = resolve(projectRoot, "backend", "static_frontend");

if (!existsSync(frontendOut)) {
  console.error(`No Next export found at ${frontendOut}. Run 'npm run build:static' first.`);
  process.exit(1);
}

rmSync(backendStatic, { recursive: true, force: true });
cpSync(frontendOut, backendStatic, { recursive: true });

console.log(`Copied static frontend -> ${backendStatic}`);
