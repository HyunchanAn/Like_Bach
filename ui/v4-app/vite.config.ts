import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { spawn, exec } from 'child_process'
import path from 'path'
import fs from 'fs'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'start-backend-middleware',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url === '/api-start-backend') {
            const projectRoot = path.resolve(__dirname, '../../')
            const hasVenv = fs.existsSync(path.join(projectRoot, '.venv', 'Scripts', 'activate.bat'))
            const command = hasVenv
              ? 'call .venv\\Scripts\\activate.bat && python src/v5/api.py'
              : 'python src/v5/api.py'
            
            res.setHeader('Content-Type', 'application/json')
            try {
              const subprocess = spawn('cmd.exe', [
                '/c',
                `start /min "Like_Bach_Backend_Process" cmd /c "set PYTHONIOENCODING=utf-8 && ${command}"`
              ], {
                cwd: projectRoot,
                detached: true,
                stdio: 'ignore',
                shell: true
              })
              subprocess.unref()
              
              console.log('Backend start request triggered via spawn successfully.')
              res.statusCode = 200
              res.end(JSON.stringify({ success: true }))
            } catch (error: any) {
              console.error('Failed to start backend via spawn:', error)
              res.statusCode = 500
              res.end(JSON.stringify({ success: false, error: error.message }))
            }
          } else if (req.url === '/api-stop-backend') {
            res.setHeader('Content-Type', 'application/json')
            try {
              exec('taskkill /fi "windowtitle eq Like_Bach_Backend_Process*" /t /f', () => {
                exec('for /f "tokens=5" %a in (\'netstat -ano ^| findstr :8000 ^| findstr LISTENING\') do ( taskkill /pid %a /t /f )', () => {
                  res.statusCode = 200
                  res.end(JSON.stringify({ success: true }))
                })
              })
            } catch (error: any) {
              console.error('Failed to stop backend via spawn:', error)
              res.statusCode = 500
              res.end(JSON.stringify({ success: false, error: error.message }))
            }
          } else {
            next()
          }
        })
      }
    }
  ],
})
