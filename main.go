package main

import (
	"flag"
	"time"

	rl "github.com/gen2brain/raylib-go/raylib"
)

func main() {
	speed := flag.Int("speed", 100, "Game speed in milliseconds (lower = faster)")
	flag.Parse()

	rl.InitWindow(900, 800, "Snake Game")
	rl.SetWindowMinSize(600, 600)
	rl.SetWindowState(rl.FlagWindowResizable)
	defer rl.CloseWindow()

	rl.SetTargetFPS(60)

	// Calculate initial grid dimensions
	width := (rl.GetScreenWidth() - 20) / 20
	height := (rl.GetScreenHeight() - 190) / 20
	g := NewGame(width, height)

	renderer := NewRenderer()
	ticker := time.NewTicker(time.Duration(*speed) * time.Millisecond)
	defer ticker.Stop()

	// Main game loop
	for !rl.WindowShouldClose() {
		// Handle input
		snake := g.GetSnake()
		if !snake.Dead {
			if rl.IsKeyPressed(rl.KeyUp) || rl.IsKeyPressed(rl.KeyW) {
				snake.SetDirection(Point{X: 0, Y: -1})
			}
			if rl.IsKeyPressed(rl.KeyDown) || rl.IsKeyPressed(rl.KeyS) {
				snake.SetDirection(Point{X: 0, Y: 1})
			}
			if rl.IsKeyPressed(rl.KeyLeft) || rl.IsKeyPressed(rl.KeyA) {
				snake.SetDirection(Point{X: -1, Y: 0})
			}
			if rl.IsKeyPressed(rl.KeyRight) || rl.IsKeyPressed(rl.KeyD) {
				snake.SetDirection(Point{X: 1, Y: 0})
			}
		}

		// Handle quit
		if rl.IsKeyPressed(rl.KeyQ) || rl.IsKeyPressed(rl.KeyEscape) {
			break
		}

		// Handle window resize
		if rl.IsWindowResized() {
			renderer.UpdateDimensions()
			width := (rl.GetScreenWidth() - 20) / 20
			height := (rl.GetScreenHeight() - 190) / 20
			g.Grid.Width = width
			g.Grid.Height = height
		}

		// Update game state at fixed interval
		select {
		case <-ticker.C:
			g.Update()
		default:
		}

		// Render
		renderer.Draw(g)
	}
}
