import { useEffect, useRef, useState } from 'react'
import { Scrfd } from './scrfd/model'
import { drawFaces } from './drawFaces'
import './App.css'

const MODEL_URL = '/models/scrfd.onnx'

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const offscreenCanvasRef = useRef<HTMLCanvasElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [isReady, setIsReady] = useState(false)
  const modelRef = useRef<Scrfd | null>(null)
  const rafRef = useRef<number>(0)
  const processingRef = useRef(false)

  useEffect(() => {
    let cancelled = false

    async function init() {
      try {
        const model = await Scrfd.fromUrl(MODEL_URL)
        if (cancelled) return
        modelRef.current = model
        setIsReady(true)
      } catch (e) {
        if (cancelled) return
        setError(e instanceof Error ? e.message : 'Failed to load model')
      }
    }

    init()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    let stream: MediaStream | null = null

    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'user' }, audio: false })
      .then((s) => {
        stream = s
        video.srcObject = s
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : 'Failed to access webcam')
      })

    return () => {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  useEffect(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    const offscreenCanvas = offscreenCanvasRef.current
    const model = modelRef.current

    if (!video || !canvas || !offscreenCanvas || !model) return

    const handlePlay = () => {
      function loop() {
        rafRef.current = requestAnimationFrame(loop)

        if (processingRef.current) return
        if (video.readyState < 2) return
        if (video.videoWidth === 0 || video.videoHeight === 0) return

        processingRef.current = true

        const w = video.videoWidth
        const h = video.videoHeight

        offscreenCanvas.width = w
        offscreenCanvas.height = h
        const offCtx = offscreenCanvas.getContext('2d')
        if (!offCtx) {
          processingRef.current = false
          return
        }
        offCtx.drawImage(video, 0, 0)

        const t0 = performance.now()
        model
          .detect(offscreenCanvas)
          .then((faces) => {
            const elapsed = performance.now() - t0
            console.log(`Inference: ${elapsed.toFixed(1)} ms`)
            // Keep the captured frame (don't redraw video – it would be several frames ahead)
            drawFaces(offCtx, faces)

            canvas.width = w
            canvas.height = h
            const ctx = canvas.getContext('2d')
            if (ctx) {
              ctx.drawImage(offscreenCanvas, 0, 0)
            }
          })
          .catch((e) => {
            console.warn('Detection failed:', e)
          })
          .finally(() => {
            processingRef.current = false
          })
      }

      rafRef.current = requestAnimationFrame(loop)
    }

    video.addEventListener('play', handlePlay)
    if (video.readyState >= 2) handlePlay()

    return () => {
      video.removeEventListener('play', handlePlay)
      cancelAnimationFrame(rafRef.current)
    }
  }, [isReady])

  if (error) {
    return (
      <div className="error">
        <p>{error}</p>
      </div>
    )
  }

  return (
    <div className="dashboard">
      <div className="panel">
        <video
          ref={videoRef}
          className="frame"
          autoPlay
          playsInline
          muted
        />
      </div>
      <div className="panel">
        <canvas ref={canvasRef} className="frame" />
      </div>
      <canvas ref={offscreenCanvasRef} style={{ display: 'none' }} />
    </div>
  )
}

export default App
