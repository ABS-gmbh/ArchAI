// Loaded with `node --import ./scripts/test-loader.mjs --test ...` (see the
// "test" script in package.json): registers the resolve hooks the unit tests
// need to import the TypeScript sources as they are.
import { register } from "node:module";

register("./test-resolve-hooks.mjs", import.meta.url);
