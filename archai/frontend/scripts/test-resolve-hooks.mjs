// Module resolution for `npm test` (node --test), which runs the TypeScript
// sources directly: Node strips the types but resolves imports by its own
// rules, not the bundler's. Two differences matter here:
//   - "@/..." is a tsconfig path alias (src/...), unknown to Node;
//   - relative imports in the sources have no extension ("./workspaceSegmentation"),
//     which Node's ESM loader does not complete.
// So "@/x" maps to src/x, and a relative or aliased specifier that does not
// resolve as written is retried with ".ts" and "/index.ts". Nothing else
// changes; packages resolve as usual. Registered by scripts/test-loader.mjs.

const SRC = new URL("../src/", import.meta.url);
const HAS_EXTENSION = /\.[cm]?[jt]sx?$|\.json$/;

export async function resolve(specifier, context, nextResolve) {
  let target = specifier;
  if (target.startsWith("@/")) {
    target = new URL(target.slice(2), SRC).href;
  }
  const local = target.startsWith(".") || target.startsWith("/") || target.startsWith("file:");
  try {
    return await nextResolve(target, context);
  } catch (error) {
    if (!local || HAS_EXTENSION.test(target)) {
      throw error;
    }
    for (const suffix of [".ts", "/index.ts"]) {
      try {
        return await nextResolve(target + suffix, context);
      } catch {
        // Try the next form; the original error is reported if none resolves.
      }
    }
    throw error;
  }
}
