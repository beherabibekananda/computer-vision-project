import { createServer } from 'vite';

async function main() {
  console.log("Loading Vite from frontend directory...");
  try {
    const server = await createServer({
      root: process.cwd(),
      server: {
        port: 5173,
        host: '0.0.0.0'
      }
    });
    console.log("Starting server listen...");
    await server.listen();
    console.log(" Vite dev server is running!");
    server.printUrls();
  } catch (e) {
    console.error("FATAL ERROR IN VITE:", e);
  }
}

main();
