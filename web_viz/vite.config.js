import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  server: {
    // Serve experiments/ folder from project root so web viz
    // reads data directly without copying to public/
    fs: {
      allow: ['..'],
    },
  },
  plugins: [
    {
      name: 'serve-experiments',
      configureServer(server) {
        const experimentsDir = path.resolve(__dirname, '..', 'experiments');
        server.middlewares.use('/experiments', (req, res, next) => {
          const filePath = path.join(experimentsDir, req.url);
          res.setHeader('Access-Control-Allow-Origin', '*');
          server.middlewares.stack.find(s => s.handle.name === 'viteServeStaticMiddleware');
          import('fs').then(fs => {
            if (fs.existsSync(filePath)) {
              const content = fs.readFileSync(filePath, 'utf-8');
              res.setHeader('Content-Type', 'application/json');
              res.end(content);
            } else {
              res.statusCode = 404;
              res.end('Not found');
            }
          });
        });
      },
    },
  ],
});
